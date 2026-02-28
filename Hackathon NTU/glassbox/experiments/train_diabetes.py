"""
Second-dataset validation: Pima Indians Diabetes Dataset.

Proves that the Glassbox architecture (GhostSignalGates + structural audit)
generalises beyond UCI Heart Disease.

Dataset: 768 samples, 8 features, binary classification (diabetic vs not)
Chunks:
  Metabolic    (Glucose, Insulin, BMI)
  Physiological (BloodPressure, SkinThickness, DiabetesPedigree)
  Demographic   (Pregnancies, Age)

Run from glassbox/:
    python experiments/train_diabetes.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from model.glassbox_net_v2 import GlassboxNetV2
from data.diabetes_groups import (
    DIABETES_FEATURE_NAMES, DIABETES_CHUNK_GROUPS
)

EPOCHS       = 300
LR           = 1e-3
LAMBDA_GHOST = 0.03
SEED         = 42


def load_pima():
    """Load Pima Indians Diabetes from sklearn/OpenML. Falls back to local CSV."""
    # OpenML 'diabetes' v1 uses short col names: preg,plas,pres,skin,insu,mass,pedi,age
    OPENML_COLS = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']
    try:
        data = fetch_openml(name='diabetes', version=1, as_frame=True, parser='auto')
        X = data.data[OPENML_COLS].values.astype(float)
        y = (data.target == 'tested_positive').astype(int).values
    except Exception:
        # Fallback: try local CSV (user can drop pima.csv in experiments/)
        csv_path = os.path.join(os.path.dirname(__file__), 'pima.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                "Could not fetch Pima dataset from OpenML and no local pima.csv found.\n"
                "Download from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
                " and save as experiments/pima.csv"
            )
        import csv
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append([float(row[n]) for n in DIABETES_FEATURE_NAMES] + [int(row['Outcome'])])
        arr = np.array(rows)
        X, y = arr[:, :-1], arr[:, -1].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, scaler


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


def train_model(model, train_loader, test_loader, is_glassbox=False):
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
        if epoch % 50 == 0 or epoch == 1:
            acc, auc = evaluate(model, test_loader)
            print(f"  epoch {epoch:>3}  acc={acc:.3f}  auc={auc:.3f}")

    return evaluate(model, test_loader)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 65)
    print("  GLASSBOX — Second Dataset Validation")
    print("  Dataset: Pima Indians Diabetes (768 samples, 8 features)")
    print("=" * 65)

    print("\nLoading Pima dataset...")
    X_tr, X_te, y_tr, y_te, scaler = load_pima()
    print(f"  Train: {len(X_tr)}   Test: {len(X_te)}   "
          f"Prevalence: {y_te.mean():.1%}")

    train_loader = make_loader(X_tr, y_tr)
    test_loader  = make_loader(X_te, y_te, shuffle=False)

    # Chunk sizes: Metabolic=3, Physiological=3, Demographic=2
    chunk_sizes = [3, 3, 2]

    # ── Train Glassbox (with ghost gates) ─────────────────────────────
    print("\n── Glassbox (with ghost gates) ─────────────────────────────")
    glassbox = GlassboxNetV2(chunk_sizes, embed_dim=16, use_ghost=True)
    gb_acc, gb_auc = train_model(glassbox, train_loader, test_loader, is_glassbox=True)

    print("\n  Gate weights at convergence:")
    for gate, w in glassbox.get_all_gate_weights().items():
        print(f"    {gate:<15} α = {w:.4f}")

    # ── Train Ablation (no ghost gates) ───────────────────────────────
    print("\n── Ablation (no ghost gates) ────────────────────────────────")
    ablation = GlassboxNetV2(chunk_sizes, embed_dim=16, use_ghost=False)
    ab_acc, ab_auc = train_model(ablation, train_loader, test_loader, is_glassbox=False)

    # ── Baseline MLP ──────────────────────────────────────────────────
    print("\n── Baseline black-box MLP ───────────────────────────────────")
    n_features = sum(chunk_sizes)   # 8
    baseline = nn.Sequential(
        nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
        nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU(),
        nn.Linear(32, 2),
    )
    bl_acc, bl_auc = train_model(baseline, train_loader, test_loader)

    # ── Summary ───────────────────────────────────────────────────────
    gate_lift = gb_acc - ab_acc
    acc_gap   = abs(gb_acc - bl_acc)

    print("\n" + "=" * 65)
    print("  PIMA DIABETES — FINAL BENCHMARK")
    print("=" * 65)
    print(f"  Glassbox (with gates): acc={gb_acc:.3f}  auc={gb_auc:.3f}  [Structurally interpretable]")
    print(f"  No-Ghost ablation:     acc={ab_acc:.3f}  auc={ab_auc:.3f}  [Ablation]")
    print(f"  Baseline MLP:          acc={bl_acc:.3f}  auc={bl_auc:.3f}  [Black-box]")
    print(f"  Ghost gate lift:  {gate_lift:+.3f}  ({'gates help ✓' if gate_lift > 0 else 'no lift'})")
    print(f"  Accuracy gap vs baseline: {acc_gap:.3f}  "
          f"({'within 2% target' if acc_gap <= 0.02 else 'outside target'})")

    # ── Sample structural audit ────────────────────────────────────────
    print("\n── Sample Structural Audit (first test patient) ─────────────")
    glassbox.eval()
    with torch.no_grad():
        x0 = torch.FloatTensor(X_te[0:1])
        logits, audit = glassbox(x0, return_audit=True)
        pred = logits.argmax(dim=1).item()
        label = 'Diabetic' if pred == 1 else 'Not Diabetic'
        truth = 'Diabetic' if y_te[0] == 1 else 'Not Diabetic'
        print(f"  Prediction: {label}  |  Ground Truth: {truth}")
        print(f"  Chunk contributions:")
        chunk_labels = {
            'Chunk0': 'Metabolic   (Glucose/Insulin/BMI)',
            'Chunk1': 'Physiological (BP/Skin/Pedigree)',
            'Chunk2': 'Demographic  (Pregnancies/Age)',
        }
        for k, v in audit['chunk_contributions'].items():
            print(f"    {chunk_labels.get(k, k):<38}  disease_push={v['disease_push']:+.4f}")
        print(f"  Ghost gate activations:")
        for gate, alpha in audit['ghost_signals'].items():
            print(f"    {gate:<15}  α={alpha:.4f}")

    # Save results
    results = {
        'dataset': 'pima_diabetes',
        'glassbox': {'acc': gb_acc, 'auc': gb_auc},
        'ablation': {'acc': ab_acc, 'auc': ab_auc},
        'baseline': {'acc': bl_acc, 'auc': bl_auc},
        'gate_lift': round(gate_lift, 4),
        'acc_gap':   round(acc_gap, 4),
        'gate_weights': glassbox.get_all_gate_weights(),
    }
    out = os.path.join(os.path.dirname(__file__), 'diabetes_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
