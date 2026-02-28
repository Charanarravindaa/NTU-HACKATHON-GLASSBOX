"""
Multi-seed consistency test for Glassbox Ghost Signal gates.

Trains GlassboxNet 5× with different random seeds and verifies that:
  1. Gate weights converge to consistent values across seeds
  2. LabDiag→Structural is consistently the most active gate
  3. Accuracy is within ±3% across runs

Run from glassbox/:
    python experiments/seed_stability.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score

from data.loader import load_heart_disease
from model.glassbox_net import GlassboxNet

SEEDS   = [0, 1, 2, 3, 4]
EPOCHS  = 300
LR      = 1e-3
LAMBDA  = 0.03

GATE_NAMES = [
    'Demographics→Vitals',
    'Vitals→LabDiagnostic',
    'LabDiag→Structural',
    'Demographics→LabDiag',
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one(seed: int) -> dict:
    set_seed(seed)
    train_loader, test_loader, scaler, X_test, y_test = load_heart_disease(batch_size=32)

    model = GlassboxNet(use_ghost=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b) + LAMBDA * model.get_gate_l1_loss()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            logits = model(X_b)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())
            all_labels.extend(y_b.numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    weights = model.get_all_gate_weights()

    return {
        'seed':    seed,
        'acc':     acc,
        'auc':     auc,
        'weights': weights,
    }


def main():
    print("=" * 70)
    print("  GLASSBOX — Multi-Seed Consistency Test")
    print(f"  Seeds: {SEEDS}   Epochs: {EPOCHS}   λ_ghost: {LAMBDA}")
    print("=" * 70)

    results = []
    for seed in SEEDS:
        print(f"\n  Training seed={seed}...", end='', flush=True)
        r = train_one(seed)
        results.append(r)
        print(f"  acc={r['acc']:.3f}  auc={r['auc']:.3f}  "
              f"LabDiag→Structural α={r['weights']['LabDiag→Structural']:.4f}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GATE WEIGHT CONSISTENCY ACROSS SEEDS")
    print("=" * 70)
    header = f"  {'Gate':<30} " + "  ".join(f"s{s}" for s in SEEDS) + "   mean   std   CV%"
    print(header)
    print("-" * 70)

    for gate in GATE_NAMES:
        vals = [r['weights'][gate] for r in results]
        mean = np.mean(vals)
        std  = np.std(vals)
        cv   = std / (mean + 1e-8) * 100
        row = f"  {gate:<30} " + "  ".join(f"{v:.3f}" for v in vals)
        row += f"   {mean:.3f}  {std:.3f}  {cv:.1f}%"
        stable = "✓" if cv < 5.0 else "✗"
        print(row + f"  {stable}")

    # ── Accuracy table ─────────────────────────────────────────────────────
    accs = [r['acc'] for r in results]
    aucs = [r['auc'] for r in results]
    print(f"\n  Accuracy: {[f'{a:.3f}' for a in accs]}")
    print(f"  AUC:      {[f'{a:.3f}' for a in aucs]}")
    print(f"  Acc  mean={np.mean(accs):.3f}  std={np.std(accs):.3f}  "
          f"range=[{min(accs):.3f}, {max(accs):.3f}]")
    print(f"  AUC  mean={np.mean(aucs):.3f}  std={np.std(aucs):.3f}")

    # ── Consistency verdict ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    dominant_per_seed = []
    for r in results:
        dominant = max(r['weights'], key=r['weights'].get)
        dominant_per_seed.append(dominant)

    dominant_gate = max(set(dominant_per_seed), key=dominant_per_seed.count)
    agreement = dominant_per_seed.count(dominant_gate) / len(SEEDS) * 100

    print(f"  Dominant gate across seeds: '{dominant_gate}'  ({agreement:.0f}% agreement)")
    print(f"  Acc variation:  ±{np.std(accs)*100:.1f}%  "
          f"({'stable ✓' if np.std(accs) < 0.03 else 'variable ✗'})")

    all_cvs = []
    for gate in GATE_NAMES:
        vals = [r['weights'][gate] for r in results]
        cv = np.std(vals) / (np.mean(vals) + 1e-8) * 100
        all_cvs.append(cv)
    print(f"  Gate weight CV: {np.mean(all_cvs):.1f}% avg  "
          f"({'consistent ✓' if np.mean(all_cvs) < 5.0 else 'inconsistent ✗'})")


if __name__ == '__main__':
    main()
