"""
PneumoniaMNIST — Spatial Chunk Decomposition on Chest X-rays.

Architecture:
  1. Shared shallow CNN backbone extracts a 32-channel feature map (28×28→14×14)
  2. Feature map is split into 4 spatial quadrants → 4 anatomical region embeddings
  3. GlassboxNetV2 applies ghost gates + order decomp on those region embeddings

Spatial chunks (after backbone → 14×14 feature map, then 2×2 spatial split):
  Chunk0: Upper-Left  — left upper lobe
  Chunk1: Upper-Right — right upper lobe
  Chunk2: Lower-Left  — left lower lobe
  Chunk3: Lower-Right — right lower lobe + heart/mediastinum

Ghost gates model cross-region dependencies:
  Which lung zones influence each other's interpretation?
  Bilateral pneumonia activates symmetric gates; unilateral does not.

Penalty fix:
  P(α) = softplus(α) = log(1+exp(α)), gradient = sigmoid(α) > 0 always
  Strong pressure on open gates, gentle on closed gates.
  α initialized at -3 (gate starts nearly closed, sigmoid(-3)≈0.05).

Run from glassbox/:
    python experiments/train_pneumonia.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, roc_auc_score

from medmnist import PneumoniaMNIST

from model.glassbox_net_v2 import GlassboxNetV2

EPOCHS        = 300
LR            = 5e-4
LAMBDA_GHOST  = 0.005   # softplus penalty; small λ gives gates room to differentiate
EMBED_DIM     = 32
BATCH_SIZE    = 64
SEED          = 42

# After the CNN backbone (Conv 3×3 stride 2), the 28×28 image becomes 14×14.
# We split this 14×14 feature map into 4 quadrants of 7×7.
BACKBONE_CHANNELS = 32
PATCH_H, PATCH_W  = 7, 7
CHUNK_DIM         = BACKBONE_CHANNELS * PATCH_H * PATCH_W   # 32*7*7 = 1568

CHUNK_NAMES = [
    'UpperLeft  (L upper lobe)',
    'UpperRight (R upper lobe)',
    'LowerLeft  (L lower lobe)',
    'LowerRight (R lower+heart)',
]


class SpatialGlassbox(nn.Module):
    """
    CNN backbone + GlassboxNetV2 on spatial chunks.

    Backbone: Conv2d(1→32, 3×3, stride=2, pad=1) + BN + ReLU
              → (N, 32, 14, 14) feature map
    Chunking: split into 4 quadrants of (N, 32, 7, 7) → flatten → (N, 1568) each
    Glassbox: 4 chunks of 1568 dims, embed_dim=32, ghost+order
    """

    def __init__(self, use_ghost=True, use_order_decomp=True):
        super().__init__()

        # Shared spatial feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(1, BACKBONE_CHANNELS, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(BACKBONE_CHANNELS),
            nn.ReLU(),
        )

        chunk_sizes = [CHUNK_DIM] * 4   # [1568, 1568, 1568, 1568]
        self.glassbox = GlassboxNetV2(
            chunk_sizes, embed_dim=EMBED_DIM,
            use_ghost=use_ghost, use_order_decomp=use_order_decomp
        )

    def _extract_chunks(self, imgs):
        """imgs: (N,1,28,28) → (N, 4*CHUNK_DIM) via backbone + spatial split."""
        feat = self.backbone(imgs)                          # (N, 32, 14, 14)
        q0 = feat[:, :, :7,   :7 ].reshape(feat.size(0), -1)   # upper-left
        q1 = feat[:, :, :7,   7: ].reshape(feat.size(0), -1)   # upper-right
        q2 = feat[:, :, 7:,   :7 ].reshape(feat.size(0), -1)   # lower-left
        q3 = feat[:, :, 7:,   7: ].reshape(feat.size(0), -1)   # lower-right
        return torch.cat([q0, q1, q2, q3], dim=1)          # (N, 4*1568)

    def forward(self, imgs, return_audit=False):
        x = self._extract_chunks(imgs)
        return self.glassbox(x, return_audit=return_audit)

    def get_all_gate_weights(self):
        return self.glassbox.get_all_gate_weights()

    def get_order_weights(self):
        return self.glassbox.get_order_weights()

    def get_gate_l1_loss(self):
        return self.glassbox.get_gate_l1_loss()


def load_pneumonia():
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[.5], std=[.5])])
    train_ds = PneumoniaMNIST(split='train', transform=transform, download=True)
    val_ds   = PneumoniaMNIST(split='val',   transform=transform, download=True)
    test_ds  = PneumoniaMNIST(split='test',  transform=transform, download=True)
    combined = torch.utils.data.ConcatDataset([train_ds, val_ds])
    train_loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, test_ds


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for imgs, labels in loader:
        labels = labels.squeeze(1).long()
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_probs.extend(probs.numpy())
        all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds), roc_auc_score(all_labels, all_probs)


def train_model(model, train_loader, test_loader, is_glassbox=False, silent=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, labels in train_loader:
            labels = labels.squeeze(1).long()
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if is_glassbox:
                loss = loss + LAMBDA_GHOST * model.get_gate_l1_loss()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if not silent and (epoch % 50 == 0 or epoch == 1):
            acc, auc = evaluate(model, test_loader)
            print(f"  epoch {epoch:>3}  acc={acc:.3f}  auc={auc:.3f}")

    return evaluate(model, test_loader)


class BaselineCNN(nn.Module):
    """Black-box CNN baseline — same backbone, flat classifier."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("  GHOSTGATE — Spatial Chunk Decomposition on Chest X-rays")
    print("  Dataset:  PneumoniaMNIST (normal vs pneumonia)")
    print("  Backbone: Conv2d(1→32, 3×3 stride-2) → 14×14 feature map")
    print("  Chunks:   4 quadrants of 7×7×32 = 1568 dims (anatomical zones)")
    print("  Penalty:  softplus(α), α init=-3 (gate starts closed)")
    print("=" * 70)

    train_loader, test_loader, test_ds = load_pneumonia()
    n_train = sum(len(ds) for ds in train_loader.dataset.datasets)
    print(f"\n  Train+Val: {n_train}   Test: {len(test_ds)}")

    # ── Glassbox: backbone + spatial chunks + ghost + order decomp ────────────
    print("\n── Glassbox (backbone + spatial chunks + ghost + order decomp) ────────")
    glassbox = SpatialGlassbox(use_ghost=True, use_order_decomp=True)
    gb_acc, gb_auc = train_model(glassbox, train_loader, test_loader, is_glassbox=True)

    print("\n  Converged ghost gate weights (α) — softplus penalty, α init=-3:")
    gate_weights = glassbox.get_all_gate_weights()
    sorted_gates = sorted(gate_weights.items(), key=lambda kv: kv[1], reverse=True)
    for gate, w in sorted_gates:
        i, j = int(gate[1]), int(gate[4])
        bar = '▓' * int(w * 40)
        print(f"    {gate:<8}  α={w:.4f}  {bar}")
        print(f"             [{CHUNK_NAMES[i][:22]} → {CHUNK_NAMES[j][:22]}]")

    print("\n  Converged order gate weights (β):")
    order_weights = glassbox.get_order_weights()
    for chunk_key, beta in order_weights.items():
        idx  = int(chunk_key.replace('Chunk', ''))
        zone = CHUNK_NAMES[idx]
        dom  = "nonlinear" if beta > 0.5 else "linear"
        print(f"    {chunk_key}  β={beta:.4f}  [{zone}]  → {dom}-dominant")

    # ── Ghost-only ablation ───────────────────────────────────────────────────
    print("\n── Ghost-only ablation (no order decomp) ────────────────────────────")
    ghost_only = SpatialGlassbox(use_ghost=True, use_order_decomp=False)
    go_acc, go_auc = train_model(ghost_only, train_loader, test_loader,
                                 is_glassbox=True, silent=True)

    # ── No-gates ablation ─────────────────────────────────────────────────────
    print("\n── No-gates ablation (raw spatial chunks only) ──────────────────────")
    ablation = SpatialGlassbox(use_ghost=False, use_order_decomp=False)
    ab_acc, ab_auc = train_model(ablation, train_loader, test_loader, silent=True)

    # ── Baseline CNN ──────────────────────────────────────────────────────────
    print("\n── Baseline black-box CNN ───────────────────────────────────────────")
    baseline = BaselineCNN()
    bl_acc, bl_auc = train_model(baseline, train_loader, test_loader)

    # ── Results ───────────────────────────────────────────────────────────────
    full_lift  = gb_acc - ab_acc
    order_lift = gb_acc - go_acc
    acc_gap    = abs(gb_acc - bl_acc)

    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — PneumoniaMNIST (Spatial CNN Chunks)")
    print("=" * 70)
    print(f"  Glassbox (backbone+ghost+order): acc={gb_acc:.3f}  auc={gb_auc:.3f}  [Full spec]")
    print(f"  Ghost-only (no order decomp):    acc={go_acc:.3f}  auc={go_auc:.3f}  [Partial]")
    print(f"  No-gates (raw spatial chunks):   acc={ab_acc:.3f}  auc={ab_auc:.3f}  [Ablation]")
    print(f"  Baseline CNN (black-box):        acc={bl_acc:.3f}  auc={bl_auc:.3f}  [Black-box]")
    print(f"  Full lift vs no-gates: {full_lift:+.3f}  ({'gates help ✓' if full_lift > 0 else 'no lift'})")
    print(f"  Order decomp lift:     {order_lift:+.3f}  ({'order decomp helps ✓' if order_lift > 0 else 'no lift'})")
    print(f"  Accuracy gap vs CNN:   {acc_gap:.3f}  "
          f"({'within 2% target' if acc_gap <= 0.02 else 'outside target'})")

    # ── Sample structural audit ────────────────────────────────────────────────
    print("\n── Sample Structural Audit ──────────────────────────────────────────")
    glassbox.eval()
    test_iter    = iter(test_loader)
    imgs, labels = next(test_iter)
    labels_long  = labels.squeeze(1).long()

    with torch.no_grad():
        logits_all = glassbox(imgs)
        preds_all  = logits_all.argmax(dim=1)
        wrong      = (preds_all != labels_long).nonzero(as_tuple=True)[0]
        idx        = wrong[0].item() if len(wrong) > 0 else 0

        x0         = imgs[idx:idx+1]
        true_label = labels_long[idx].item()
        logits, audit = glassbox(x0, return_audit=True)
        pred       = logits.argmax(dim=1).item()
        names      = ['Normal', 'Pneumonia']
        result     = 'WRONG' if pred != true_label else 'correct'

    print(f"  Prediction: {names[pred]}  |  Ground Truth: {names[true_label]}  [{result}]")

    print(f"\n  [Layer 1] Spatial chunk contributions:")
    for ci, zone in enumerate(CHUNK_NAMES):
        v    = audit['chunk_contributions'].get(f'Chunk{ci}', {})
        push = v.get('disease_push', 0)
        bar  = '▓' * min(int(abs(push) * 2), 25)
        direction = 'toward pneumonia' if push > 0 else 'toward normal'
        print(f"    {zone}  push={push:+.4f}  {bar}  [{direction}]")

    most_active = max(audit['ghost_signals'].items(), key=lambda kv: kv[1])
    print(f"\n  [Layer 2] Ghost gate activations (most active: {most_active[0]} α={most_active[1]:.4f}):")
    for gate, alpha in sorted(audit['ghost_signals'].items(), key=lambda kv: kv[1], reverse=True):
        i, j = int(gate[1]), int(gate[4])
        print(f"    {gate}  α={alpha:.4f}  [{CHUNK_NAMES[i][:18]} → {CHUNK_NAMES[j][:18]}]")

    print(f"\n  [Layer 3] Order decomposition:")
    for cname, od in audit['order_decomp'].items():
        idx_c = int(cname.replace('Chunk', ''))
        zone  = CHUNK_NAMES[idx_c]
        bar_l = '█' * int(od['linear_frac'] * 20)
        bar_n = '░' * (20 - int(od['linear_frac'] * 20))
        print(f"    {zone}  β={od['beta']:.4f}  "
              f"linear={od['linear_frac']*100:.0f}%  |{bar_l}{bar_n}|  [{od['dominant']}]")

    # Save
    results = {
        'dataset':    'pneumonia_mnist',
        'glassbox':   {'acc': gb_acc,  'auc': gb_auc},
        'ghost_only': {'acc': go_acc,  'auc': go_auc},
        'ablation':   {'acc': ab_acc,  'auc': ab_auc},
        'baseline':   {'acc': bl_acc,  'auc': bl_auc},
        'full_lift':   round(full_lift,  4),
        'order_lift':  round(order_lift, 4),
        'acc_gap':     round(acc_gap,    4),
        'gate_weights':  gate_weights,
        'order_weights': order_weights,
    }
    out = os.path.join(os.path.dirname(__file__), 'pneumonia_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
