import numpy as np
from typing import Optional


class StructuralAudit:
    """
    The Structural Audit Engine — Glassbox's showpiece.

    When a wrong prediction occurs, this engine produces a human-readable
    fault report by analysing:
      1. Ghost Signal α values — anomalous cross-chunk influence
      2. Chunk activation L2 norms — noisy or over-active chunks

    The verdict identifies: what caused the fault, where it occurred,
    and how confident the audit is.
    """

    GHOST_THRESHOLD   = 0.65  # α above this → anomalous (fallback for global weight)
    ALPHA_THRESHOLD_Z = 1.5  # per-sample α z-score above this → patient-specific anomaly
    MAG_THRESHOLD_Z   = 1.0  # ghost magnitude z-score above this → anomalous per-sample
    NORM_THRESHOLD_Z  = 2.5  # L2 norm z-score above this → noisy chunk

    # Narrative templates for the demo "aha moment"
    GHOST_NARRATIVES = {
        'Vitals→LabDiagnostic': (
            "High max heart rate (thalach) created an unexpected interaction with "
            "lab markers (oldpeak/restecg), masking the ST-depression signal."
        ),
        'Demographics→Vitals': (
            "Patient demographics (age/sex/cp) over-influenced the cardiovascular "
            "vitals interpretation, biasing the prediction pathway."
        ),
        'LabDiag→Structural': (
            "Lab diagnostic signals (fbs/restecg/slope) abnormally coupled with "
            "structural findings (ca/thal), amplifying structural risk indicators."
        ),
        'Demographics→LabDiag': (
            "Demographic baseline (age/sex/cp) created strong priors that suppressed "
            "the lab diagnostic signal, causing under-estimation of clinical severity."
        ),
    }

    def __init__(self, norm_history: Optional[list] = None):
        """
        norm_history: list of norm dicts from training set, used to compute
                      per-layer z-scores. If None, z-score comparison is skipped.
        """
        self.norm_history = norm_history
        self._norm_means = {}
        self._norm_stds = {}
        if norm_history:
            self._fit_norm_stats(norm_history)

    def _fit_norm_stats(self, history):
        # history entries may be dicts with keys 'chunk_norms' or flat dicts
        flat_history = []
        for h in history:
            if 'chunk_norms' in h:
                flat_history.append(h['chunk_norms'])
            else:
                flat_history.append(h)
        if not flat_history:
            return
        all_keys = flat_history[0].keys()
        for key in all_keys:
            vals = [h[key] for h in flat_history if key in h]
            self._norm_means[key] = np.mean(vals)
            self._norm_stds[key] = np.std(vals) + 1e-8

        # Fit magnitude z-scores
        self._mag_means = {}
        self._mag_stds  = {}
        mag_history = [h.get('ghost_magnitudes', {}) for h in history if isinstance(h, dict)]
        mag_history = [h for h in mag_history if h]
        if mag_history:
            for key in mag_history[0].keys():
                vals = [h[key] for h in mag_history if key in h]
                self._mag_means[key] = np.mean(vals)
                self._mag_stds[key]  = np.std(vals) + 1e-8

        # Fit alpha z-scores — per-sample alpha now varies per patient
        # ghost_signals[gate] = patient's gate weight (float) for N=1 inference
        self._alpha_means = {}
        self._alpha_stds  = {}
        alpha_history = [h.get('ghost_signals', {}) for h in history if isinstance(h, dict)]
        alpha_history = [h for h in alpha_history if h]
        if alpha_history:
            for key in alpha_history[0].keys():
                vals = [h[key] for h in alpha_history
                        if key in h and isinstance(h[key], (int, float))]
                if vals:
                    self._alpha_means[key] = np.mean(vals)
                    self._alpha_stds[key]  = np.std(vals) + 1e-8

    def run(self, audit_dict: dict, prediction: int, ground_truth: int,
            input_features: dict) -> dict:
        """
        Run a full structural audit.

        The audit uses exact structural decompositions from the model:
          - chunk_contributions: each chunk's exact logit contribution (no approximation)
          - ghost_directions: whether each ghost signal pushed toward/away from disease
          - ghost_magnitudes: per-sample cross-chunk influence strength
          - chunk_norms: activation health check per layer
        """
        report = {
            'prediction':    prediction,
            'ground_truth':  ground_truth,
            'fault':         prediction != ground_truth,
            'prediction_label':   'Heart Disease' if prediction == 1 else 'No Disease',
            'ground_truth_label': 'Heart Disease' if ground_truth == 1 else 'No Disease',
            'chunk_analysis':        self._analyze_chunks(audit_dict['chunk_norms']),
            'ghost_signal_analysis': self._analyze_ghosts(
                audit_dict['ghost_signals'],
                audit_dict.get('ghost_magnitudes', {}),
                audit_dict.get('ghost_directions', {}),
            ),
            'chunk_contributions':   audit_dict.get('chunk_contributions', {}),
            'verdict': None,
        }
        report['verdict'] = self._determine_verdict(
            report, input_features, audit_dict.get('ghost_directions', {})
        )
        return report

    def _analyze_chunks(self, chunk_norms: dict) -> dict:
        analysis = {}
        for layer_key, norm_val in chunk_norms.items():
            entry = {'norm': norm_val, 'anomalous': False, 'z_score': None}
            if layer_key in self._norm_means:
                z = (norm_val - self._norm_means[layer_key]) / self._norm_stds[layer_key]
                entry['z_score'] = round(float(z), 3)
                entry['anomalous'] = bool(abs(z) > self.NORM_THRESHOLD_Z)
            analysis[layer_key] = entry
        return analysis

    def _analyze_ghosts(self, ghost_signals: dict, ghost_magnitudes: dict = None,
                        ghost_directions: dict = None) -> dict:
        if ghost_magnitudes is None:
            ghost_magnitudes = {}
        if ghost_directions is None:
            ghost_directions = {}
        analysis = {}
        for gate_name, alpha in ghost_signals.items():
            mag = ghost_magnitudes.get(gate_name, 0.0)

            # Magnitude z-score (existing)
            mag_z = None
            mag_anomalous = False
            if hasattr(self, '_mag_means') and gate_name in getattr(self, '_mag_means', {}):
                mag_z = float((mag - self._mag_means[gate_name]) / self._mag_stds[gate_name])
                mag_anomalous = bool(mag_z > self.MAG_THRESHOLD_Z)

            # Alpha z-score (new) — "was this gate unusually open for THIS patient?"
            alpha_z = None
            alpha_anomalous = False
            if hasattr(self, '_alpha_means') and gate_name in self._alpha_means:
                alpha_z = float((float(alpha) - self._alpha_means[gate_name])
                                / self._alpha_stds[gate_name])
                alpha_anomalous = bool(alpha_z > self.ALPHA_THRESHOLD_Z)

            anomalous = bool(
                (alpha > self.GHOST_THRESHOLD) or mag_anomalous or alpha_anomalous
            )
            direction_info = ghost_directions.get(gate_name, {})
            analysis[gate_name] = {
                'alpha':             round(float(alpha), 4),
                'alpha_zscore':      round(alpha_z, 3) if alpha_z is not None else None,
                'magnitude':         round(float(mag), 4),
                'magnitude_zscore':  round(float(mag_z), 3) if mag_z is not None else None,
                'anomalous':         anomalous,
                'strength':          self._alpha_label(alpha),
                'disease_direction': float(direction_info.get('disease_direction', 0.0)),
                'pushing':           direction_info.get('pushing', 'unknown'),
            }
        return analysis

    def _alpha_label(self, alpha: float) -> str:
        if alpha < 0.3:
            return 'weak'
        elif alpha < 0.55:
            return 'moderate'
        elif alpha < self.GHOST_THRESHOLD:
            return 'strong'
        else:
            return 'ANOMALOUS'

    def _determine_verdict(self, report: dict, input_features: dict,
                           ghost_directions: dict = None) -> dict:
        """
        Three-layer attribution — from coarse to fine:

        1. WHICH CHUNK drove the wrong prediction  (chunk logit contribution)
        2. WHICH GATE caused the distortion        (ghost signal anomaly)
        3. IN WHAT DIRECTION                       (ghost direction: toward/away disease)

        Together these answer: "Which feature cluster caused this, and why?"
        All from exact structural decomposition — zero approximation.
        """
        if ghost_directions is None:
            ghost_directions = {}

        ghost_analysis    = report['ghost_signal_analysis']
        chunk_contribs    = report.get('chunk_contributions', {})
        prediction        = report['prediction']
        fault             = report['fault']

        # ── Step 1: Identify the dominant chunk by logit contribution ──
        # Always identify the dominant chunk (most active in prediction direction)
        dominant_chunk = None
        dominant_push  = None
        if chunk_contribs:
            push_sign = 1 if prediction == 1 else -1  # 1=disease predicted, -1=healthy predicted
            ranked = sorted(
                chunk_contribs.items(),
                key=lambda x: push_sign * x[1]['disease_push'],
                reverse=True,
            )
            dominant_chunk, dominant_push = ranked[0]

        # ── Step 2: Sort gates by magnitude z-score ───────────────────
        all_ghosts_sorted = sorted(
            ghost_analysis.items(),
            key=lambda x: max(
                abs(x[1].get('magnitude_zscore') or 0.0),
                abs(x[1].get('alpha_zscore') or 0.0),
            ),
            reverse=True,
        )
        anomalous_ghosts = [(n, i) for n, i in all_ghosts_sorted if i['anomalous']]

        # ── Step 3: Anomalous chunk norms ─────────────────────────────
        anomalous_chunks = [
            (name, info) for name, info in report['chunk_analysis'].items()
            if info['anomalous']
        ]

        # ── Build structural explanation ───────────────────────────────
        chunk_explanation = ""
        if dominant_chunk and dominant_push:
            push_val = dominant_push['disease_push']
            direction = "toward disease" if push_val > 0 else "away from disease"
            chunk_explanation = (
                f"The {dominant_chunk} chunk is the primary driver "
                f"(logit push={push_val:+.2f} {direction}). "
            )

        # ── No fault: correct prediction ──────────────────────────────
        if not fault:
            return {
                'cause':          'No Fault',
                'location':       'N/A',
                'dominant_chunk': dominant_chunk,
                'confidence':     0.95,
                'message':        (
                    f"Prediction is correct. {chunk_explanation}"
                    if chunk_explanation else
                    'Prediction is correct — no structural fault detected.'
                ),
            }

        # ── Confirmed ghost signal anomaly ─────────────────────────────
        if anomalous_ghosts:
            gate_name, gate_info = anomalous_ghosts[0]
            mag_z   = gate_info.get('magnitude_zscore') or 0.0
            alpha_z = gate_info.get('alpha_zscore') or 0.0
            # Use whichever z-score is larger as the primary signal
            z = max(abs(mag_z), abs(alpha_z))
            direction_info = ghost_directions.get(gate_name, {})
            push_word = direction_info.get('pushing', 'unknown').replace('_', ' ')
            push_val  = direction_info.get('disease_direction', 0.0)
            confidence = min(0.95, 0.5 + min(z, 3.0) * 0.15)
            narrative  = self.GHOST_NARRATIVES.get(gate_name, "Unexpected cross-chunk dependency.")
            alpha_note = (f"α={gate_info['alpha']:.3f} (z={alpha_z:+.2f} vs population)"
                         if alpha_z else f"α={gate_info['alpha']:.3f}")
            return {
                'cause':      'Ghost Signal Anomaly',
                'location':   gate_name,
                'dominant_chunk': dominant_chunk,
                'alpha':          gate_info['alpha'],
                'alpha_zscore':   round(alpha_z, 3),
                'magnitude_zscore': round(mag_z, 3),
                'ghost_push':     round(push_val, 4),
                'ghost_direction': push_word,
                'confidence':     round(confidence, 3),
                'message': (
                    f"{chunk_explanation}"
                    f"Ghost Signal '{gate_name}' is anomalously active "
                    f"({alpha_note}, magnitude z={mag_z:.2f}) and is pushing {push_word} "
                    f"(force={push_val:+.3f}). {narrative}"
                ),
                'all_anomalous_gates': [
                    {'gate': n, 'alpha': i['alpha'], 'z': i.get('magnitude_zscore'),
                     'pushing': ghost_directions.get(n, {}).get('pushing', '?')}
                    for n, i in anomalous_ghosts
                ],
            }

        # ── Confirmed chunk noise ──────────────────────────────────────
        if anomalous_chunks:
            chunk_names = ', '.join(k.split('_layer')[0] for k, _ in anomalous_chunks)
            return {
                'cause':      'Chunk Activation Noise',
                'location':   chunk_names,
                'dominant_chunk': dominant_chunk,
                'confidence': 0.60,
                'message': (
                    f"{chunk_explanation}"
                    f"Noisy activations detected in: {chunk_names}. "
                    f"High L2 norm indicates the chunk received conflicting signals."
                ),
            }

        # ── Suspected gate — always attribute for faults ───────────────
        if fault and all_ghosts_sorted:
            gate_name, gate_info = all_ghosts_sorted[0]
            z = gate_info.get('magnitude_zscore') or 0.0
            direction_info = ghost_directions.get(gate_name, {})
            push_word = direction_info.get('pushing', 'unknown').replace('_', ' ')
            push_val  = direction_info.get('disease_direction', 0.0)
            narrative = self.GHOST_NARRATIVES.get(gate_name, "Subtle cross-chunk dependency.")
            return {
                'cause':      'Suspected Ghost Signal',
                'location':   gate_name,
                'dominant_chunk': dominant_chunk,
                'alpha':      gate_info['alpha'],
                'magnitude_zscore': round(z, 3),
                'ghost_push': round(push_val, 4),
                'ghost_direction': push_word,
                'confidence': round(max(0.35, 0.25 + abs(z) * 0.1), 3),
                'message': (
                    f"{chunk_explanation}"
                    f"Suspected Ghost Signal contribution from '{gate_name}' "
                    f"(magnitude z={z:.2f}, pushing {push_word}). "
                    f"{narrative}"
                ),
            }

        # Fallback (fault but no ghost/norm signal detected)
        return {
            'cause':          'Unattributed Fault',
            'location':       'N/A',
            'dominant_chunk': dominant_chunk,
            'confidence':     0.35,
            'message':        (
                f"{chunk_explanation}"
                "Prediction error detected but no anomalous ghost signal found."
            ),
        }

    def compute_stability(self, alpha_series: dict) -> dict:
        """
        Compute ghost signal stability across a batch of inferences.

        alpha_series: {gate_name: [alpha1, alpha2, ...]}
        Returns: {gate_name: {'mean': ..., 'std': ..., 'stable': bool}}
        """
        results = {}
        for gate_name, alphas in alpha_series.items():
            arr = np.array(alphas)
            std_val = float(np.std(arr))
            results[gate_name] = {
                'mean':   round(float(np.mean(arr)), 4),
                'std':    round(std_val, 4),
                'stable': std_val < 0.15,
            }
        return results
