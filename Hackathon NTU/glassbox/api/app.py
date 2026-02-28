"""
FastAPI backend for Glassbox.

Endpoints:
  GET  /health
  POST /predict
  POST /audit
  GET  /ghost_signals
  GET  /demo_cases
  POST /compare_shap
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pickle
import numpy as np
import torch
from typing import List
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model.glassbox_net import GlassboxNet
from model.audit import StructuralAudit
from data.feature_groups import FEATURE_NAMES, CHUNK_GROUPS

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Glassbox API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artefacts at startup ──────────────────────────────────────────────────
ARTEFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artefacts')

_model: GlassboxNet = None
_scaler = None
_demo_data: dict = {}
_auditor: StructuralAudit = None
_norm_history: list = []


def _load_artefacts():
    global _model, _scaler, _demo_data, _auditor

    model_path = os.path.join(ARTEFACT_DIR, 'glassbox.pt')
    scaler_path = os.path.join(ARTEFACT_DIR, 'scaler.pkl')
    demo_path   = os.path.join(ARTEFACT_DIR, 'demo_cases.json')

    if not os.path.exists(model_path):
        print("WARNING: No trained model found. Run training/train.py first.")
        return

    _model = GlassboxNet()
    _model.load_state_dict(torch.load(model_path, map_location='cpu'))
    _model.eval()

    with open(scaler_path, 'rb') as f:
        _scaler = pickle.load(f)

    with open(demo_path) as f:
        _demo_data = json.load(f)

    # Build norm history from X_test for fitted z-score auditor
    norm_history = []
    X_test_list = _demo_data.get("X_test", [])
    if X_test_list and _model is not None:
        import torch as _torch
        _model.eval()
        with _torch.no_grad():
            for row in X_test_list[:60]:   # use up to 60 samples for fitting
                x = _torch.FloatTensor([row])
                _, ad = _model(x, return_audit=True)
                norm_history.append(ad)
    _auditor = StructuralAudit(norm_history=norm_history if norm_history else None)
    print(f"Glassbox model and artefacts loaded (auditor fitted on {len(norm_history)} samples).")


@app.on_event("startup")
def startup():
    _load_artefacts()


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class FeaturesRequest(BaseModel):
    features: List[float]   # 13 values in FEATURE_NAMES order


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: List[float]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _preprocess(features: List[float]) -> torch.Tensor:
    if len(features) != 13:
        raise HTTPException(status_code=400, detail="Expected 13 features")
    arr = np.array(features).reshape(1, -1)
    if _scaler is not None:
        arr = _scaler.transform(arr)
    return torch.FloatTensor(arr)


def _require_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "artefact_dir": ARTEFACT_DIR,
    }


@app.post("/predict")
def predict(req: FeaturesRequest):
    _require_model()
    x = _preprocess(req.features)
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)[0].tolist()
    pred = int(np.argmax(probs))
    return {
        "prediction": pred,
        "prediction_label": "Heart Disease" if pred == 1 else "No Disease",
        "confidence": round(max(probs), 4),
        "probabilities": [round(p, 4) for p in probs],
    }


@app.post("/audit")
def audit(req: FeaturesRequest, ground_truth: int = -1):
    _require_model()
    x = _preprocess(req.features)
    with torch.no_grad():
        logits, audit_dict = _model(x, return_audit=True)
        probs = torch.softmax(logits, dim=1)[0].tolist()
    pred = int(np.argmax(probs))

    features_named = {FEATURE_NAMES[i]: req.features[i] for i in range(13)}
    gt = ground_truth if ground_truth in (0, 1) else pred  # fallback for demo
    report = _auditor.run(audit_dict, pred, gt, features_named)

    # Make audit_dict JSON-serialisable
    report['ghost_signal_raw'] = {k: round(v, 4) for k, v in audit_dict['ghost_signals'].items()}
    report['chunk_norms_raw']  = {k: round(v, 4) for k, v in audit_dict['chunk_norms'].items()}
    report['probabilities'] = [round(p, 4) for p in probs]
    report['confidence'] = round(max(probs), 4)
    return report


@app.get("/ghost_signals")
def ghost_signals():
    _require_model()
    weights = _model.get_all_gate_weights()
    return {
        gate: {
            "alpha": round(w, 4),
            "anomalous": w > StructuralAudit.GHOST_THRESHOLD,
            "strength": (
                "weak" if w < 0.3 else
                "moderate" if w < 0.55 else
                "strong" if w < 0.65 else
                "ANOMALOUS"
            ),
        }
        for gate, w in weights.items()
    }


@app.get("/demo_cases")
def demo_cases():
    if not _demo_data:
        raise HTTPException(status_code=503, detail="Demo data not available. Run training first.")
    wrong = sorted(
        _demo_data.get("wrong", []),
        key=lambda c: c.get("verdict_confidence", 0),
        reverse=True
    )
    return {
        "wrong_predictions": wrong[:8],
        "correct_predictions": _demo_data.get("correct", [])[:5],
        "final_metrics": _demo_data.get("final_metrics", {}),
        "stability": _demo_data.get("stability", {}),
    }


@app.post("/compare_shap")
def compare_shap(req: FeaturesRequest):
    """Run SHAP on the same input and return side-by-side with Glassbox."""
    _require_model()
    x = _preprocess(req.features)

    # Glassbox result
    with torch.no_grad():
        logits, audit_dict = _model(x, return_audit=True)
        probs = torch.softmax(logits, dim=1)[0].tolist()
    pred = int(np.argmax(probs))

    # SHAP via a wrapper that exposes the baseline model
    shap_values = None
    try:
        import shap
        import pickle

        baseline_path = os.path.join(ARTEFACT_DIR, 'baseline.pt')
        if os.path.exists(baseline_path):
            from training.train import BaselineMLP
            baseline = BaselineMLP()
            baseline.load_state_dict(torch.load(baseline_path, map_location='cpu'))
            baseline.eval()

            # Use X_test subset as background
            X_bg = np.array(_demo_data.get("X_test", [])[:50])
            if len(X_bg) > 0:
                def predict_fn(X_np):
                    with torch.no_grad():
                        t = torch.FloatTensor(X_np)
                        logits = baseline(t)
                        return torch.softmax(logits, dim=1).numpy()

                explainer = shap.KernelExplainer(predict_fn, X_bg[:20])
                x_np = np.array(req.features).reshape(1, -1)
                if _scaler:
                    x_np = _scaler.transform(x_np)
                sv = explainer.shap_values(x_np, nsamples=100)
                # sv is list of 2 arrays (one per class) — use class 1
                shap_vals = sv[1][0].tolist() if isinstance(sv, list) else sv[0].tolist()
                shap_values = {FEATURE_NAMES[i]: round(shap_vals[i], 4) for i in range(13)}
    except Exception as e:
        shap_values = {"error": f"SHAP computation failed: {str(e)}"}

    return {
        "glassbox": {
            "prediction": pred,
            "prediction_label": "Heart Disease" if pred == 1 else "No Disease",
            "confidence": round(max(probs), 4),
            "ghost_signals": {k: round(v, 4) for k, v in audit_dict['ghost_signals'].items()},
            "explanation_type": "Structural (Ghost Signal gates — exact interaction pathways)",
        },
        "shap": {
            "feature_importances": shap_values,
            "explanation_type": "Post-hoc approximation (Shapley values — feature importance only)",
            "limitation": (
                "SHAP shows WHICH features mattered — not HOW they interacted. "
                "It cannot identify the cross-chunk pathway that caused the prediction."
            ),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
