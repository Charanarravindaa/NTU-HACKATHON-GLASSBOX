"""
Auto-Discover Experiment — the purest form of the GhostGate spec.

Instead of manually grouping features by domain knowledge, this experiment:
  1. Takes a flat 30-feature dataset (Wisconsin Breast Cancer)
  2. Clusters features into chunks using hierarchical correlation clustering
  3. Trains GlassboxNetV2 on the discovered chunks, with:
       • Ghost Signal Gates  (cross-chunk α gates)
       • Order Decomposition (per-chunk β gates: 1st-order vs nth-order)
  4. Shows what the model discovered — and whether it makes sense

This validates the original spec's vision:
  "group features into clusters based on their mutual information and correlation"
  "order decomposition: separate 1st-order from nth-order signal contributions"

Run from glassbox/:
    python experiments/train_autodiscover.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from model.glassbox_net_v2 import GlassboxNetV2

EPOCHS       = 300
LR           = 1e-3
LAMBDA_GHOST = 0.02
N_CHUNKS     = 4
SEED         = 42


# ── Step 1: Auto-discover feature chunks ──────────────────────────────────────
def compute_feature_chunks(X: np.ndarray, feature_names: list, n_chunks: int) -> dict:
    """
    Cluster features into n_chunks groups using hierarchical clustering
    on the absolute Pearson correlation matrix.

    Features that co-vary (high |corr|) are placed in the same chunk —
    they share a "functional context" the model can exploit together.
    """
    corr = np.abs(np.corrcoef(X.T))          # (n_features, n_features)
    np.fill_diagonal(corr, 1.0)
    dist = 1.0 - corr                         # distance: high corr = near

    # Condense the symmetric matrix to a vector for scipy
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='ward')
    labels = fcluster(Z, n_chunks, criterion='maxclust')   # 1-indexed

    chunks = {}
    for fi, c in enumerate(labels):
        chunks.setdefault(c, []).append(fi)

    # Return sorted by chunk id, rename to 0-indexed
    return {
        f'Chunk{i}': sorted(indices)
        for i, (_, indices) in enumerate(sorted(chunks.items()))
    }


def print_chunks(chunks: dict, feature_names: list):
    print("\n  Auto-discovered feature chunks (by correlation clustering):")
    for name, indices in chunks.items():
        feat_list = ', '.join(feature_names[i] for i in indices)
        print(f"    {name} [{len(indices)} features]: {feat_list}")


# ── Step 2: Data loading ───────────────────────────────────────────────────────
def load_breast_cancer_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = np.array(data.feature_names)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, feature_names, scaler


def make_loader(X, y, batch_size=32, shuffle=True):
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), torch.LongTensor(y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for X_b, y_b in loader:
        logits = model(X_b)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_probs.extend(probs.numpy())
        all_labels.extend(y_b.numpy())
    return (accuracy_score(all_labels, all_preds),
            roc_auc_score(all_labels, all_probs))


def train_model(model, train_loader, test_loader, is_glassbox=False, silent=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            if is_glassbox:
                loss = loss + LAMBDA_GHOST * model.get_gate_l1_loss()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if not silent and (epoch % 50 == 0 or epoch == 1):
            acc, auc = evaluate(model, test_loader)
            print(f"  epoch {epoch:>3}  acc={acc:.3f}  auc={auc:.3f}")

    return evaluate(model, test_loader)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("  GHOSTGATE — Auto-Discover + Order Decomposition")
    print("  Dataset: Wisconsin Breast Cancer (569 samples, 30 features)")
    print("  No manual grouping — chunks discovered from correlation structure")
    print("  Order Decomposition — per-chunk 1st-order vs nth-order β gate")
    print("=" * 70)

    # Load data
    X_tr, X_te, y_tr, y_te, feature_names, scaler = load_breast_cancer_data()
    print(f"\n  Train: {len(X_tr)}   Test: {len(X_te)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Prevalence (benign): {y_te.mean():.1%}")

    # Discover chunks from training data correlation
    print(f"\n  Computing feature correlation structure...")
    chunks = compute_feature_chunks(X_tr, feature_names, n_chunks=N_CHUNKS)
    print_chunks(chunks, feature_names)

    # Build chunk_sizes list in chunk order
    chunk_sizes = [len(indices) for indices in chunks.values()]

    # Reorder X columns to match chunk layout
    col_order = [i for indices in chunks.values() for i in indices]
    X_tr_reordered = X_tr[:, col_order]
    X_te_reordered = X_te[:, col_order]

    train_loader = make_loader(X_tr_reordered, y_tr)
    test_loader  = make_loader(X_te_reordered, y_te, shuffle=False)

    # ── Train Glassbox: ghost gates + order decomposition ─────────────────────
    print("\n── Glassbox (auto-discovered + ghost gates + order decomp) ──────────")
    glassbox = GlassboxNetV2(chunk_sizes, embed_dim=16,
                             use_ghost=True, use_order_decomp=True)
    gb_acc, gb_auc = train_model(glassbox, train_loader, test_loader, is_glassbox=True)

    print("\n  Converged ghost gate weights (α):")
    for gate, w in glassbox.get_all_gate_weights().items():
        print(f"    {gate:<12} α = {w:.4f}")

    print("\n  Converged order gate weights (β — 1st-order vs nth-order):")
    for chunk, beta in glassbox.get_order_weights().items():
        complexity = "nonlinear-dominant" if beta > 0.5 else "linear-dominant"
        print(f"    {chunk:<12} β = {beta:.4f}  [{complexity}]")

    # ── Ablation: no ghost gates, no order decomp ─────────────────────────────
    print("\n── Ablation (same chunks, no ghost gates, no order decomp) ──────────")
    ablation = GlassboxNetV2(chunk_sizes, embed_dim=16,
                             use_ghost=False, use_order_decomp=False)
    ab_acc, ab_auc = train_model(ablation, train_loader, test_loader)

    # ── Ablation: ghost gates only, no order decomp ───────────────────────────
    print("\n── Ablation (ghost gates only, no order decomp) ─────────────────────")
    ghost_only = GlassboxNetV2(chunk_sizes, embed_dim=16,
                               use_ghost=True, use_order_decomp=False)
    go_acc, go_auc = train_model(ghost_only, train_loader, test_loader,
                                 is_glassbox=True, silent=True)

    # ── Baseline black-box MLP ────────────────────────────────────────────────
    print("\n── Baseline black-box MLP ───────────────────────────────────────────")
    baseline = nn.Sequential(
        nn.Linear(sum(chunk_sizes), 128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
        nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU(),
        nn.Linear(32, 2),
    )
    bl_acc, bl_auc = train_model(baseline, train_loader, test_loader)

    # ── Results ───────────────────────────────────────────────────────────────
    gate_lift   = gb_acc - ab_acc
    order_lift  = gb_acc - go_acc
    acc_gap     = abs(gb_acc - bl_acc)

    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — Breast Cancer (Auto-Discovered Chunks)")
    print("=" * 70)
    print(f"  Glassbox (chunks + ghost + order): acc={gb_acc:.3f}  auc={gb_auc:.3f}  [Full spec]")
    print(f"  Ghost-only (no order decomp):      acc={go_acc:.3f}  auc={go_auc:.3f}  [Partial]")
    print(f"  No gates ablation:                 acc={ab_acc:.3f}  auc={ab_auc:.3f}  [Raw chunks]")
    print(f"  Baseline MLP:                      acc={bl_acc:.3f}  auc={bl_auc:.3f}  [Black-box]")
    print(f"  Full spec vs no-gates lift: {gate_lift:+.3f}  ({'gates help ✓' if gate_lift > 0 else 'no lift'})")
    print(f"  Order decomp lift:          {order_lift:+.3f}  ({'order decomp helps ✓' if order_lift > 0 else 'no lift'})")
    print(f"  Accuracy gap vs baseline:   {acc_gap:.3f}  "
          f"({'within 2% target' if acc_gap <= 0.02 else 'outside target'})")

    # ── Sample structural audit ────────────────────────────────────────────────
    print("\n── Sample Structural Audit (first test patient) ─────────────────────")
    glassbox.eval()
    with torch.no_grad():
        x0 = torch.FloatTensor(X_te_reordered[0:1])
        logits, audit = glassbox(x0, return_audit=True)
        pred  = logits.argmax(dim=1).item()
        truth = int(y_te[0])
        labels = ['Malignant', 'Benign']
        print(f"  Prediction: {labels[pred]}  |  Ground Truth: {labels[truth]}")

        print(f"\n  [Layer 1] Chunk contributions (which cluster matters most):")
        for ci, (cname, cindices) in enumerate(chunks.items()):
            key = f'Chunk{ci}'
            v = audit['chunk_contributions'].get(key, {})
            feat_summary = feature_names[cindices[0]].split(' ')[0]
            push = v.get('disease_push', 0)
            bar = '▓' * int(abs(push) * 2) if abs(push) < 20 else '▓' * 20
            direction = 'toward disease' if push > 0 else 'toward healthy'
            print(f"    {cname} ({feat_summary}...) "
                  f"disease_push={push:+.4f}  {bar}  [{direction}]")

        print(f"\n  [Layer 2] Ghost gate activations (cross-chunk dependencies):")
        for gate, alpha in audit['ghost_signals'].items():
            print(f"    {gate:<15}  α={alpha:.4f}")

        print(f"\n  [Layer 3] Order decomposition (linear vs complex within each chunk):")
        if audit['order_decomp']:
            for cname, od in audit['order_decomp'].items():
                bar_lin = '█' * int(od['linear_frac'] * 20)
                bar_nln = '░' * (20 - int(od['linear_frac'] * 20))
                print(f"    {cname}  β={od['beta']:.4f}  "
                      f"linear={od['linear_frac']*100:.0f}%  "
                      f"|{bar_lin}{bar_nln}|  [{od['dominant']}]")
        else:
            print("    (order decomp not enabled)")

    # ── What did the model discover? ──────────────────────────────────────────
    print("\n── What the model discovered ────────────────────────────────────────")
    print("  Checking if discovered chunks align with known feature structure...")
    print("  (Breast Cancer has 10 measurements × 3 stats: mean/error/worst)")
    for cname, indices in chunks.items():
        feat_labels = [feature_names[i] for i in indices]
        has_mean  = sum(1 for f in feat_labels if 'mean' in f)
        has_error = sum(1 for f in feat_labels if 'error' in f)
        has_worst = sum(1 for f in feat_labels if 'worst' in f)
        dominant = max([('mean', has_mean), ('error', has_error), ('worst', has_worst)],
                       key=lambda x: x[1])
        purity = dominant[1] / len(indices) * 100
        beta = glassbox.get_order_weights().get(cname, None)
        order_str = f"  β={beta:.3f} ({'nonlinear' if beta and beta > 0.5 else 'linear'})" if beta is not None else ""
        print(f"  {cname}: {dominant[0]}-dominant ({purity:.0f}% purity){order_str}")

    # Save results
    results = {
        'dataset': 'breast_cancer',
        'chunks':  {k: [int(i) for i in v] for k, v in chunks.items()},
        'chunk_names': {k: [feature_names[i] for i in v] for k, v in chunks.items()},
        'glassbox':   {'acc': gb_acc,  'auc': gb_auc},
        'ghost_only': {'acc': go_acc,  'auc': go_auc},
        'ablation':   {'acc': ab_acc,  'auc': ab_auc},
        'baseline':   {'acc': bl_acc,  'auc': bl_auc},
        'gate_lift':   round(gate_lift,  4),
        'order_lift':  round(order_lift, 4),
        'gate_weights':  glassbox.get_all_gate_weights(),
        'order_weights': glassbox.get_order_weights(),
    }
    out = os.path.join(os.path.dirname(__file__), 'autodiscover_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
