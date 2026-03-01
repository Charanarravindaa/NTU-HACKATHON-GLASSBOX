"""
Evaluation utilities: accuracy, AUC, ghost signal stability, chunk importance.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
from collections import defaultdict

from model.glassbox_net import GlassboxNet
from model.audit import StructuralAudit
from data.loader import load_heart_disease
from data.feature_groups import FEATURE_NAMES, CHUNK_GROUPS


@torch.no_grad()
def full_evaluation(model_path: str, scaler=None):
    """Load a saved Glassbox model and run full evaluation."""
    _, test_loader, scaler_new, X_test, y_test = load_heart_disease()
    if scaler is None:
        scaler = scaler_new

    model = GlassboxNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    alpha_series = defaultdict(list)
    chunk_norms_all = defaultdict(list)

    for i in range(len(X_test)):
        x = torch.FloatTensor(X_test[i:i+1])
        logits, audit_dict = model(x, return_audit=True)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_probs.append(prob)
        all_labels.append(int(y_test[i]))
        for gate, alpha in audit_dict['ghost_signals'].items():
            alpha_series[gate].append(alpha)
        for layer, norm in audit_dict['chunk_norms'].items():
            chunk_norms_all[layer].append(norm)

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Disease', 'Disease']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    auditor = StructuralAudit()
    stability = auditor.compute_stability(dict(alpha_series))
    print("\nGhost Signal Stability:")
    for gate, stats in stability.items():
        status = "STABLE" if stats['stable'] else "UNSTABLE"
        print(f"  {gate:<30} mean={stats['mean']:.3f}  std={stats['std']:.3f}  [{status}]")

    print("\nChunk Importance (mean L2 norm):")
    chunk_means = {}
    for chunk_name in CHUNK_GROUPS:
        layer_keys = [k for k in chunk_norms_all if k.startswith(chunk_name)]
        if layer_keys:
            avg = np.mean([np.mean(chunk_norms_all[k]) for k in layer_keys])
            chunk_means[chunk_name] = avg
    for name, val in sorted(chunk_means.items(), key=lambda x: -x[1]):
        bar = '█' * int(val * 5)
        print(f"  {name:<20} {val:.4f}  {bar}")

    return acc, auc, stability, chunk_means


if __name__ == '__main__':
    artefacts = os.path.join(os.path.dirname(__file__), '..', 'artefacts')
    model_path = os.path.join(artefacts, 'glassbox.pt')
    if os.path.exists(model_path):
        full_evaluation(model_path)
    else:
        print("No saved model found. Run training/train.py first.")
