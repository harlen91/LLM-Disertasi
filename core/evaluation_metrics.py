"""
core/evaluation_metrics.py
--------------------------
Utilitas evaluasi metrik komprehensif, kalibrasi probabilitas, bootstrap confidence interval,
optimasi threshold strictly-on-validation, dan manajemen model_manifest.json untuk
eksperimen IndoBERTweet & Hybrid Dual-Stream Fusion.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import APP_VERSION, MODEL_MANIFEST_FILENAME, TAXONOMY


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Optional[float]]:
    """
    Menghitung metrik kalibrasi probabilitas biner:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Brier Score
    - Binary Log-Loss
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob_pos, dtype=float), 1e-7, 1.0 - 1e-7)

    if len(y_true) == 0:
        return {"ece": None, "mce": None, "brier_score": None, "log_loss": None}

    brier = float(brier_score_loss(y_true, y_prob))
    try:
        ll = float(log_loss(y_true, np.column_stack([1.0 - y_prob, y_prob]), labels=[0, 1]))
    except Exception:
        ll = None

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for idx in range(n_bins):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        if count > 0:
            avg_conf = float(y_prob[mask].mean())
            avg_acc = float(y_true[mask].mean())
            gap = abs(avg_acc - avg_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4) if ll is not None else None,
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    threshold: float = 0.50,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, List[Optional[float]]]:
    """Menghitung 95% Bootstrap Confidence Interval untuk metrik utama."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob_pos, dtype=float)
    n = len(y_true)
    if n < 5:
        return {"accuracy": [None, None], "macro_f1": [None, None], "roc_auc": [None, None], "pr_auc": [None, None]}

    rng = np.random.RandomState(seed)
    accs, f1s, rocs, prs = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        pred = (yp >= threshold).astype(int)
        accs.append(float(accuracy_score(yt, pred)))
        f1s.append(float(f1_score(yt, pred, average="macro", zero_division=0)))
        if len(np.unique(yt)) == 2:
            rocs.append(float(roc_auc_score(yt, yp)))
            prs.append(float(average_precision_score(yt, yp)))

    def _ci(arr: List[float]) -> List[Optional[float]]:
        if not arr:
            return [None, None]
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    return {
        "accuracy": _ci(accs),
        "macro_f1": _ci(f1s),
        "roc_auc": _ci(rocs),
        "pr_auc": _ci(prs),
    }


def find_optimal_threshold_on_val(
    val_probs_pos: np.ndarray,
    val_targets: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Mencari ambang batas probabilitas optimal STRICTLY pada Validation Set
    menggunakan Macro F1-Score sebagai metrik utama (tie-breaker Accuracy).
    """
    if grid is None:
        grid = np.arange(0.20, 0.81, 0.02)

    probs = np.asarray(val_probs_pos, dtype=float)
    targets = np.asarray(val_targets, dtype=int)

    best_tau = 0.50
    best_f1 = -1.0
    best_acc = -1.0

    for tau in grid:
        preds = (probs >= tau).astype(int)
        f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(targets, preds))
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-6 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_tau = float(tau)

    return round(best_tau, 2), round(best_f1, 4), round(best_acc, 4)


def compute_comprehensive_classification_metrics(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    threshold: float = 0.50,
    df_meta: Optional[pd.DataFrame] = None,
    seed: int = 42,
    n_bootstrap: int = 500,
) -> Dict[str, Any]:
    """Menghitung metrik evaluasi komprehensif untuk klasifikasi cyberbullying biner dan slicing taksonomi."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob_pos, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    roc_auc = None
    pr_auc = None
    if len(np.unique(y_true)) == 2:
        roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
        pr_auc = round(float(average_precision_score(y_true, y_prob)), 4)

    cal = compute_calibration_metrics(y_true, y_prob)
    cis = bootstrap_confidence_interval(
        y_true, y_prob, threshold=threshold, n_bootstrap=n_bootstrap, seed=seed
    )

    # Error analysis: False Positives & False Negatives
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

    error_analysis: Dict[str, Any] = {
        "false_positive_count": int(len(fp_indices)),
        "false_negative_count": int(len(fn_indices)),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0,
    }

    implicit_breakdown: Dict[str, Any] = {}
    type_breakdown: Dict[str, Any] = {}

    if df_meta is not None and len(df_meta) == len(y_true):
        meta = df_meta.reset_index(drop=True)

        # Slice by implicit level (E0, E1, E2)
        imp_col = "implicit_level" if "implicit_level" in meta.columns else ("target_implicit_level" if "target_implicit_level" in meta.columns else None)
        if imp_col:
            for lvl in ["E0", "E1", "E2"]:
                sub_mask = (meta[imp_col].astype(str) == lvl).values
                n_sub = int(sub_mask.sum())
                if n_sub > 0:
                    yt_sub = y_true[sub_mask]
                    yp_sub = y_pred[sub_mask]
                    implicit_breakdown[lvl] = {
                        "count": n_sub,
                        "correct": int((yt_sub == yp_sub).sum()),
                        "accuracy": round(float(accuracy_score(yt_sub, yp_sub)), 4),
                        "macro_f1": round(float(f1_score(yt_sub, yp_sub, average="macro", zero_division=0)), 4),
                    }

        # Slice by cyberbullying type
        t_col = "type" if "type" in meta.columns else ("target_type" if "target_type" in meta.columns else None)
        if t_col:
            for t_code in sorted(meta[t_col].astype(str).unique()):
                sub_mask = (meta[t_col].astype(str) == t_code).values
                n_sub = int(sub_mask.sum())
                if n_sub > 0:
                    yt_sub = y_true[sub_mask]
                    yp_sub = y_pred[sub_mask]
                    type_breakdown[t_code] = {
                        "count": n_sub,
                        "correct": int((yt_sub == yp_sub).sum()),
                        "accuracy": round(float(accuracy_score(yt_sub, yp_sub)), 4),
                    }

        # Error examples (text snippets)
        text_col = "comment" if "comment" in meta.columns else ("comment_normalized" if "comment_normalized" in meta.columns else None)
        if text_col:
            error_analysis["fp_samples"] = [
                {
                    "text": str(meta.loc[i, text_col])[:140],
                    "prob": round(float(y_prob[i]), 4),
                    "implicit_level": str(meta.loc[i, imp_col]) if imp_col else "N/A",
                }
                for i in fp_indices[:5]
            ]
            error_analysis["fn_samples"] = [
                {
                    "text": str(meta.loc[i, text_col])[:140],
                    "prob": round(float(y_prob[i]), 4),
                    "implicit_level": str(meta.loc[i, imp_col]) if imp_col else "N/A",
                    "type": str(meta.loc[i, t_col]) if t_col else "N/A",
                }
                for i in fn_indices[:5]
            ]

    return {
        "threshold": round(float(threshold), 4),
        "support": int(len(y_true)),
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "calibration": cal,
        "confidence_intervals_95": cis,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "confusion_matrix_2x2": [[tn, fp], [fn, tp]],
        "per_class": {
            "non_cyberbullying": {
                "precision": round(float(p_cls[0]), 4),
                "recall": round(float(r_cls[0]), 4),
                "f1": round(float(f1_cls[0]), 4),
                "support": int(sup_cls[0]),
            },
            "cyberbullying": {
                "precision": round(float(p_cls[1]), 4),
                "recall": round(float(r_cls[1]), 4),
                "f1": round(float(f1_cls[1]), 4),
                "support": int(sup_cls[1]),
            },
        },
        "implicitness_breakdown": implicit_breakdown,
        "type_breakdown": type_breakdown,
        "error_analysis": error_analysis,
    }


def build_model_manifest(
    model_architecture: str,
    pretrained_model: str,
    checkpoint_location: Path,
    hyperparameters: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
    optimal_threshold: float,
    dataset_version: str = "unspecified",
    target_source: str = "pseudo_labeled_llm",
    random_seed: int = 42,
    model_version: str = APP_VERSION,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Membangun struktur standar model_manifest.json untuk reproduktibilitas penelitian."""
    manifest: Dict[str, Any] = {
        "model_architecture": model_architecture,
        "pretrained_model": pretrained_model,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "target_source": target_source,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_seed": int(random_seed),
        "optimal_threshold": round(float(optimal_threshold), 4),
        "threshold_selection_split": "validation",
        "taxonomy": TAXONOMY,
        "hyperparameters": hyperparameters,
        "evaluation_metrics": evaluation_metrics,
        "checkpoint_location": str(checkpoint_location),
        "git_storage_note": "Bobot biner model (.pt/.safetensors) dikecualikan dari Git melalui .gitignore; simpan secara lokal atau pada external artifact storage.",
    }
    if extra_metadata:
        manifest.update(extra_metadata)
    return manifest


def save_model_manifest(output_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Menyimpan model_manifest.json dan threshold/training_config pendamping ke direktori model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MODEL_MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


def load_model_manifest(model_dir: Path) -> Optional[Dict[str, Any]]:
    """Memuat model_manifest.json apabila tersedia di direktori model."""
    manifest_path = model_dir / MODEL_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

