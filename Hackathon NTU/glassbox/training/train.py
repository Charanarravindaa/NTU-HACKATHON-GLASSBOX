"""
Training loop for Glassbox and a baseline black-box MLP.

Trains both models on the UCI Heart Disease dataset and logs:
  - Per-epoch: loss, accuracy, AUC-ROC
  - Post-training: ghost signal stability, chunk importance, wrong prediction audit
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import defaultdict

from data.loader import load_heart_disease
from model.glassbox_net import GlassboxNet
from model.audit import StructuralAudit
from data.feature_groups import FEATURE_NAMES


# ── Baseline black-box MLP ─────────────────────────────────────────────────────
class BaselineMLP(nn.Module):
    """Equivalent parameter count MLP for benchmark comparison."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 32),  nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.net(x)


# ── Training helpers ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, is_glassbox=False, lambda_ghost=0.0):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        if is_glassbox and lambda_ghost > 0.0:
            loss = loss + lambda_ghost * model.get_gate_l1_loss()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, is_glassbox=False):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0., [], [], []
    for X_batch, y_batch in loader:
        if is_glassbox:
            logits = model(X_batch)
        else:
            logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_probs.extend(probs.numpy())
        all_labels.extend(y_batch.numpy())
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / n, acc, auc


# ── Post-training analysis ─────────────────────────────────────────────────────
@torch.no_grad()
def collect_ghost_stability(model, X_test_tensor):
    """Collect α values across full test set and compute stability."""
    model.eval()
    alpha_series = defaultdict(list)
    for i in range(len(X_test_tensor)):
        x = X_test_tensor[i:i+1]
        _, audit_dict = model(x, return_audit=True)
        for gate, alpha in audit_dict['ghost_signals'].items():
            alpha_series[gate].append(alpha)
    auditor = StructuralAudit()
    return auditor.compute_stability(dict(alpha_series))


@torch.no_grad()
def collect_wrong_predictions(model, X_test, y_test):
    """Find all wrong predictions and run structural audit on each."""
    model.eval()
    wrong_cases = []
    full_history = []  # store full audit_dicts for fitting auditor stats

    for i in range(len(X_test)):
        x = torch.FloatTensor(X_test[i:i+1])
        logits, audit_dict = model(x, return_audit=True)
        full_history.append(audit_dict)
        pred = logits.argmax(dim=1).item()
        gt = int(y_test[i])
        if pred != gt:
            features = {FEATURE_NAMES[j]: float(X_test[i, j]) for j in range(13)}
            wrong_cases.append({
                'index': i,
                'prediction': pred,
                'ground_truth': gt,
                'audit_dict': audit_dict,
                'features': features,
            })

    auditor = StructuralAudit(norm_history=full_history)
    reports = []
    for case in wrong_cases:
        report = auditor.run(
            case['audit_dict'],
            case['prediction'],
            case['ground_truth'],
            case['features'],
        )
        report['index'] = case['index']
        reports.append(report)

    return reports, auditor


# ── Main training entry point ──────────────────────────────────────────────────
def train(epochs=300, lr=1e-3, save_dir=None, lambda_ghost=0.03):
    print("=" * 60)
    print("  GLASSBOX — Training Run")
    print(f"  lambda_ghost = {lambda_ghost} (L1 gate regularisation)")
    print("=" * 60)

    train_loader, test_loader, scaler, X_test, y_test = load_heart_disease(batch_size=32)
    X_test_tensor = torch.FloatTensor(X_test)

    glassbox = GlassboxNet(use_ghost=True)
    baseline = BaselineMLP()

    criterion = nn.CrossEntropyLoss()
    opt_gb = optim.AdamW(glassbox.parameters(), lr=lr, weight_decay=1e-4)
    opt_bl = optim.AdamW(baseline.parameters(), lr=lr, weight_decay=1e-4)

    sched_gb = optim.lr_scheduler.CosineAnnealingLR(opt_gb, T_max=epochs)
    sched_bl = optim.lr_scheduler.CosineAnnealingLR(opt_bl, T_max=epochs)

    history = {'glassbox': [], 'baseline': []}

    print(f"\n{'Epoch':>5} | {'GB Loss':>8} {'GB Acc':>7} {'GB AUC':>7} | "
          f"{'BL Loss':>8} {'BL Acc':>7} {'BL AUC':>7}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        gb_tr_loss, _ = train_epoch(glassbox, train_loader, opt_gb, criterion,
                                    is_glassbox=True, lambda_ghost=lambda_ghost)
        bl_tr_loss, _ = train_epoch(baseline, train_loader, opt_bl, criterion)

        gb_loss, gb_acc, gb_auc = eval_epoch(glassbox, test_loader, criterion, is_glassbox=True)
        bl_loss, bl_acc, bl_auc = eval_epoch(baseline, test_loader, criterion)

        sched_gb.step()
        sched_bl.step()

        history['glassbox'].append({'epoch': epoch, 'loss': gb_loss, 'acc': gb_acc, 'auc': gb_auc})
        history['baseline'].append({'epoch': epoch, 'loss': bl_loss, 'acc': bl_acc, 'auc': bl_auc})

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>5} | {gb_loss:>8.4f} {gb_acc:>7.3f} {gb_auc:>7.3f} | "
                  f"{bl_loss:>8.4f} {bl_acc:>7.3f} {bl_auc:>7.3f}")

    # ── Ablation: train without ghost gates ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ABLATION — No-Ghost Baseline (proves gates add value)")
    print("=" * 60)

    ablation = GlassboxNet(use_ghost=False)
    opt_ab = optim.AdamW(ablation.parameters(), lr=lr, weight_decay=1e-4)
    sched_ab = optim.lr_scheduler.CosineAnnealingLR(opt_ab, T_max=epochs)

    print(f"\n{'Epoch':>5} | {'AB Loss':>8} {'AB Acc':>7} {'AB AUC':>7}")
    print("-" * 35)
    for epoch in range(1, epochs + 1):
        train_epoch(ablation, train_loader, opt_ab, criterion)
        ab_loss, ab_acc, ab_auc = eval_epoch(ablation, test_loader, criterion)
        sched_ab.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>5} | {ab_loss:>8.4f} {ab_acc:>7.3f} {ab_auc:>7.3f}")

    final_ab = {'acc': ab_acc, 'auc': ab_auc}

    # ── Post-training metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  POST-TRAINING ANALYSIS")
    print("=" * 60)

    stability = collect_ghost_stability(glassbox, X_test_tensor)
    print("\n Ghost Signal Stability (std < 0.15 = stable):")
    for gate, stats in stability.items():
        status = "✓ STABLE" if stats['stable'] else "✗ UNSTABLE"
        print(f"  {gate:<30} mean={stats['mean']:.3f}  std={stats['std']:.3f}  {status}")

    wrong_reports, auditor = collect_wrong_predictions(glassbox, X_test, y_test)
    ghost_fault_count = sum(1 for r in wrong_reports
                            if r['verdict']['cause'] == 'Ghost Signal Anomaly')
    fault_attr_rate = ghost_fault_count / max(1, len(wrong_reports))

    print(f"\n Wrong predictions: {len(wrong_reports)}")
    print(f" Fault attribution rate: {fault_attr_rate:.1%} "
          f"(target >80% via ghost signal detection)")

    print("\n Gate weights at convergence:")
    for gate, w in glassbox.get_all_gate_weights().items():
        print(f"  {gate:<30} α = {w:.4f}")

    final_gb = history['glassbox'][-1]
    final_bl = history['baseline'][-1]
    print("\n FINAL BENCHMARK:")
    print(f"  Glassbox (with gates): acc={final_gb['acc']:.3f}  auc={final_gb['auc']:.3f}  [Structurally interpretable]")
    print(f"  Glassbox (no gates):   acc={final_ab['acc']:.3f}  auc={final_ab['auc']:.3f}  [Ablation — no ghost]")
    print(f"  Baseline MLP:          acc={final_bl['acc']:.3f}  auc={final_bl['auc']:.3f}  [Black-box]")
    gate_lift = final_gb['acc'] - final_ab['acc']
    acc_gap   = abs(final_gb['acc'] - final_bl['acc'])
    print(f"  Ghost gate lift:  {gate_lift:+.3f} acc vs no-ghost ({'gates help ✓' if gate_lift > 0 else 'no lift — consider tuning λ'})")
    print(f"  Accuracy gap vs baseline: {acc_gap:.3f} ({'within target' if acc_gap <= 0.02 else 'outside target'})")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'artefacts')
    os.makedirs(save_dir, exist_ok=True)

    torch.save(glassbox.state_dict(), os.path.join(save_dir, 'glassbox.pt'))
    torch.save(baseline.state_dict(), os.path.join(save_dir, 'baseline.pt'))

    import pickle
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save wrong prediction cases (for demo dashboard) — enriched format
    demo_cases = []
    for r in wrong_reports[:10]:
        idx = r['index']
        verdict = r['verdict']
        demo_cases.append({
            'index':              idx,
            'prediction':         r['prediction'],
            'ground_truth':       r['ground_truth'],
            'prediction_label':   r['prediction_label'],
            'ground_truth_label': r['ground_truth_label'],
            'verdict_message':    verdict['message'],
            'verdict_cause':      verdict['cause'],
            'verdict_confidence': verdict.get('confidence', 0.5),
            'dominant_chunk':     verdict.get('dominant_chunk'),
            'features':           {FEATURE_NAMES[j]: float(X_test[idx, j]) for j in range(13)},
            'chunk_contributions': r.get('chunk_contributions', {}),
            'ghost_signals':      r['ghost_signal_analysis'],
        })

    # Add a few correct predictions too
    correct_cases = []
    with torch.no_grad():
        for i in range(len(X_test)):
            if len(correct_cases) >= 5:
                break
            x = torch.FloatTensor(X_test[i:i+1])
            logits, audit_dict = glassbox(x, return_audit=True)
            pred = logits.argmax(dim=1).item()
            if pred == int(y_test[i]):
                correct_cases.append({
                    'index': i,
                    'prediction': pred,
                    'ground_truth': int(y_test[i]),
                    'prediction_label': 'Heart Disease' if pred == 1 else 'No Disease',
                    'ground_truth_label': 'Heart Disease' if int(y_test[i]) == 1 else 'No Disease',
                    'verdict_message': 'Prediction correct — no fault.',
                    'verdict_cause': 'None',
                    'ghost_signals': {k: {'alpha': round(v, 4), 'anomalous': v > 0.65,
                                         'strength': 'weak' if v < 0.3 else 'moderate'}
                                     for k, v in audit_dict['ghost_signals'].items()},
                })

    all_demo = {
        'wrong': demo_cases,
        'correct': correct_cases,
        'stability': stability,
        'final_metrics': {
            'glassbox':  {'acc': final_gb['acc'], 'auc': final_gb['auc']},
            'baseline':  {'acc': final_bl['acc'], 'auc': final_bl['auc']},
            'ablation':  {'acc': final_ab['acc'], 'auc': final_ab['auc']},
            'gate_lift': round(final_gb['acc'] - final_ab['acc'], 4),
        },
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.bool_): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(os.path.join(save_dir, 'demo_cases.json'), 'w') as f:
        json.dump(all_demo, f, indent=2, cls=_NumpyEncoder)

    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2, cls=_NumpyEncoder)

    print(f"\n Artefacts saved to: {save_dir}")
    return glassbox, baseline, ablation, scaler, history, stability, wrong_reports


if __name__ == '__main__':
    train()
