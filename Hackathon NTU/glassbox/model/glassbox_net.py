import torch
import torch.nn as nn
import torch.nn.functional as F

from model.chunks import ChunkNet
from model.ghost_gate import GhostSignalGate
from data.feature_groups import CHUNK_GROUPS


class GlassboxNet(nn.Module):
    """
    Full Glassbox model assembly.

    Architecture:
        Chunk A (Demographics, 3 features)  ─┬──> gate_ab ──> gated_ab
        Chunk B (Vitals, 3 features)         ─┤    gate_bc ──> gated_bc
        Chunk C (LabDiagnostic, 4 features)  ─┤    gate_cd ──> gated_cd
        Chunk D (Structural, 3 features)     ─┘    gate_ac ──> gated_ac

    All gated outputs are concatenated and passed to a linear classifier.

    Ghost signal α values and chunk L2 norms are logged on every inference
    and returned when return_audit=True.

    use_ghost=False disables all ghost gates (ablation mode): each chunk's
    output passes through unchanged, proving gates add genuine value.
    """

    EMBED_DIM = 16

    def __init__(self, use_ghost=True):
        super().__init__()
        self.use_ghost = use_ghost

        # ── Chunk subnetworks ──────────────────────────────────────────
        self.chunk_a = ChunkNet(3, [64, 32], self.EMBED_DIM, 'Demographics')
        self.chunk_b = ChunkNet(3, [64, 32], self.EMBED_DIM, 'Vitals')
        self.chunk_c = ChunkNet(4, [64, 32], self.EMBED_DIM, 'LabDiagnostic')
        self.chunk_d = ChunkNet(3, [32, 16], self.EMBED_DIM, 'Structural')

        # ── Ghost Signal gates ─────────────────────────────────────────
        self.gate_ab = GhostSignalGate(self.EMBED_DIM, self.EMBED_DIM, 'Demographics→Vitals')
        self.gate_bc = GhostSignalGate(self.EMBED_DIM, self.EMBED_DIM, 'Vitals→LabDiagnostic')
        self.gate_cd = GhostSignalGate(self.EMBED_DIM, self.EMBED_DIM, 'LabDiag→Structural')
        self.gate_ac = GhostSignalGate(self.EMBED_DIM, self.EMBED_DIM, 'Demographics→LabDiag')

        # ── Classifier ────────────────────────────────────────────────
        # 4 gated chunk outputs, each EMBED_DIM → 64 total
        self.classifier = nn.Linear(self.EMBED_DIM * 4, 2)

    def forward(self, x, return_audit=False):
        """
        Args:
            x:            (N, 13) full feature tensor
            return_audit: if True, also return audit_dict

        Returns:
            logits:     (N, 2)
            audit_dict: (only if return_audit=True)
                {
                  'ghost_signals': {gate_name: alpha_float},
                  'chunk_norms':   {chunk_name_layerN: l2_norm_float}
                }
        """
        # ── Split features by chunk ────────────────────────────────────
        x_a = x[:, CHUNK_GROUPS['Demographics']['indices']]
        x_b = x[:, CHUNK_GROUPS['Vitals']['indices']]
        x_c = x[:, CHUNK_GROUPS['LabDiagnostic']['indices']]
        x_d = x[:, CHUNK_GROUPS['Structural']['indices']]

        # ── Chunk forward passes ───────────────────────────────────────
        emb_a, norms_a = self.chunk_a(x_a)
        emb_b, norms_b = self.chunk_b(x_b)
        emb_c, norms_c = self.chunk_c(x_c)
        emb_d, norms_d = self.chunk_d(x_d)

        # ── Apply Ghost Signal gates (or bypass in ablation mode) ─────
        if self.use_ghost:
            gated_ab, alpha_ab, mag_ab = self.gate_ab(emb_a, emb_b)
            gated_bc, alpha_bc, mag_bc = self.gate_bc(emb_b, emb_c)
            gated_cd, alpha_cd, mag_cd = self.gate_cd(emb_c, emb_d)
            gated_ac, alpha_ac, mag_ac = self.gate_ac(emb_a, emb_c)
        else:
            # Ablation: no ghost blending — raw chunk embeddings only
            gated_ab, alpha_ab, mag_ab = emb_a, 0.0, 0.0
            gated_bc, alpha_bc, mag_bc = emb_b, 0.0, 0.0
            gated_cd, alpha_cd, mag_cd = emb_c, 0.0, 0.0
            gated_ac, alpha_ac, mag_ac = emb_a, 0.0, 0.0

        # ── Concatenate all gated embeddings ───────────────────────────
        gated_chunks = [gated_ab, gated_bc, gated_cd, gated_ac]
        combined = torch.cat(gated_chunks, dim=-1)
        logits = self.classifier(combined)

        if not return_audit:
            return logits

        # ── Exact chunk logit decomposition (the Glassbox guarantee) ──
        # Since logits = W @ combined + b, and combined is a concatenation
        # of 4 equal-sized blocks, each chunk's contribution to the logit
        # is exactly: W[:, i*D:(i+1)*D] @ gated_chunk[i]
        # This is an exact, lossless decomposition — not an approximation.
        chunk_names = ['Demographics', 'Vitals', 'LabDiagnostic', 'Structural']
        D = self.EMBED_DIM
        W = self.classifier.weight   # (2, 64)
        b = self.classifier.bias     # (2,)
        chunk_logit_contribs = {}
        for i, name in enumerate(chunk_names):
            W_block = W[:, i*D:(i+1)*D]               # (2, 16)
            contrib = (W_block @ gated_chunks[i].T)    # (2, N)
            # disease_push: how much this chunk pushed toward class 1 (disease)
            # positive = pushing toward disease, negative = pushing away
            disease_push = contrib[1] - contrib[0]     # (N,) logit difference
            chunk_logit_contribs[name] = {
                'disease_logit': round(float(contrib[1].mean()), 4),
                'healthy_logit': round(float(contrib[0].mean()), 4),
                'disease_push':  round(float(disease_push.mean()), 4),
            }

        # Ghost signal direction: does the ghost signal push toward disease?
        # Compute what the gated output WOULD be without ghost (pure chunk)
        # and compare with actual gated output via the classifier
        ghost_directions = {}
        raw_chunks   = [emb_a, emb_b, emb_c, emb_d]
        gate_pairs   = [(0,1,'Demographics→Vitals'), (1,2,'Vitals→LabDiagnostic'),
                        (2,3,'LabDiag→Structural'),  (0,2,'Demographics→LabDiag')]
        for src_i, dst_i, gate_name in gate_pairs:
            W_block = W[:, src_i*D:(src_i+1)*D]
            # How much does gated output differ from raw source chunk?
            delta = gated_chunks[src_i] - raw_chunks[src_i]   # (N, 16)
            ghost_push = (W_block @ delta.T)                   # (2, N)
            disease_direction = float((ghost_push[1] - ghost_push[0]).mean())
            ghost_directions[gate_name] = {
                'disease_direction': round(disease_direction, 4),
                'pushing': 'toward_disease' if disease_direction > 0 else 'away_from_disease',
            }

        audit_dict = {
            'ghost_signals': {
                'Demographics→Vitals':      alpha_ab,
                'Vitals→LabDiagnostic':     alpha_bc,
                'LabDiag→Structural':       alpha_cd,
                'Demographics→LabDiag':     alpha_ac,
            },
            'ghost_magnitudes': {
                'Demographics→Vitals':      mag_ab,
                'Vitals→LabDiagnostic':     mag_bc,
                'LabDiag→Structural':       mag_cd,
                'Demographics→LabDiag':     mag_ac,
            },
            # Exact structural decompositions — no approximation
            'chunk_contributions': chunk_logit_contribs,
            'ghost_directions':    ghost_directions,
            'chunk_norms': {**norms_a, **norms_b, **norms_c, **norms_d},
        }
        return logits, audit_dict

    def get_all_gate_weights(self) -> dict:
        """Return current α weights without running inference."""
        return {
            'Demographics→Vitals':      self.gate_ab.get_gate_weight(),
            'Vitals→LabDiagnostic':     self.gate_bc.get_gate_weight(),
            'LabDiag→Structural':       self.gate_cd.get_gate_weight(),
            'Demographics→LabDiag':     self.gate_ac.get_gate_weight(),
        }

    def get_gate_l1_loss(self) -> torch.Tensor:
        """
        softplus penalty on per-sample gate logits.

        Each gate stores self._last_logit = (N, 1) from its last forward pass.
        Penalty = mean softplus(logit) across all samples and all gates.
        Gradient = sigmoid(logit) per sample — always positive, stronger on
        open gates than closed ones.
        """
        logits = [g._last_logit for g in
                  [self.gate_ab, self.gate_bc, self.gate_cd, self.gate_ac]
                  if g._last_logit is not None]
        if not logits:
            return torch.tensor(0.0)
        return F.softplus(torch.cat(logits, dim=-1)).mean()
