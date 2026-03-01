import torch
import torch.nn as nn


class ChunkNet(nn.Module):
    """
    A deep MLP subnetwork representing one semantically-grouped feature chunk.

    When use_order_decomp=False (default):
        Single nonlinear pathway → embedding (original behaviour)

    When use_order_decomp=True (Order Decomposition):
        Two parallel pathways:
          • 1st-order path:  x → Linear → emb_linear   (no activation)
          • nth-order path:  x → MLP   → emb_nonlinear (deep, nonlinear)
        Combined via a learned gate β:
          embedding = emb_linear + β · emb_nonlinear

        β → 0 : chunk is essentially a linear model of its features
        β large: chunk relies on complex, high-order feature interactions

    The β values are exposed in activation_norms and consumed by the audit engine.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 chunk_name: str, use_order_decomp: bool = False):
        super().__init__()
        self.chunk_name = chunk_name
        self.use_order_decomp = use_order_decomp

        # ── nth-order (nonlinear) path ──────────────────────────────────────────
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # ── 1st-order (linear) path + order gate ────────────────────────────────
        if use_order_decomp:
            self.linear_path = nn.Linear(input_dim, output_dim, bias=True)
            # β: learned order gate — sigmoid(0) = 0.5, gradient free to move
            self.order_gate = nn.Parameter(torch.tensor(0.0))

        # Keep references to Linear layers for per-layer norm tracking
        self._linear_indices = [
            i for i, m in enumerate(self.network) if isinstance(m, nn.Linear)
        ]

    def _run_network(self, x, activation_norms: dict):
        """Run nonlinear path, recording hidden-layer norms."""
        h = x
        layer_num = 0
        for module in self.network:
            h = module(h)
            if isinstance(module, nn.ReLU):
                norm_val = h.norm(dim=-1).mean().item()
                activation_norms[f'{self.chunk_name}_layer{layer_num}'] = norm_val
                layer_num += 1
        return h

    def forward(self, x):
        """
        Returns:
            embedding:        (N, output_dim)
            activation_norms: dict with layer norms and (if order decomp) order gate info
        """
        activation_norms = {}

        if self.use_order_decomp:
            emb_linear    = self.linear_path(x)                      # 1st-order
            emb_nonlinear = self._run_network(x, activation_norms)   # nth-order

            beta = torch.sigmoid(self.order_gate)
            embedding = emb_linear + beta * emb_nonlinear

            # Expose order decomp diagnostics
            activation_norms[f'{self.chunk_name}_order_beta']       = beta.item()
            activation_norms[f'{self.chunk_name}_linear_norm']      = emb_linear.norm(dim=-1).mean().item()
            activation_norms[f'{self.chunk_name}_nonlinear_norm']   = emb_nonlinear.norm(dim=-1).mean().item()
        else:
            embedding = self._run_network(x, activation_norms)

        return embedding, activation_norms
