#!/usr/bin/env python3
"""
evaluate_indobertweet.py
------------------------
Skrip pengujian dan evaluasi lengkap untuk model IndoBERTweet Baseline maupun HybridFusionModel.
Menggunakan preprocessing konsisten (`clean_and_normalize_social_text`), threshold dari
`model_manifest.json` (Validation Set), serta melaporkan metrik komprehensif.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import MODELS_DIR
from core.data_understanding import clean_and_normalize_social_text
from core.evaluation_metrics import compute_comprehensive_classification_metrics, load_model_manifest
from core.fusion_model import detect_torch_device, load_hybrid_classifier_with_manifest, predict_hybrid_text

DEFAULT_MODEL_DIR = MODELS_DIR / "indobertweet_cyberbullying" / "best_model"
DEFAULT_HYBRID_WEIGHTS = MODELS_DIR / "hybrid_fusion_indobertweet" / "best_hybrid_model.pt"


def load_classifier(model_dir: Path, device: Optional[str] = None):
    """Memuat tokenizer, model baseline IndoBERTweet, device, serta manifest jika tersedia."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Direktori model tidak ditemukan di: {model_dir}. "
            "Jalankan pelatihan terlebih dahulu: 'python3 scripts/train_indobertweet.py'."
        )
    has_weights = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
    if not has_weights:
        raise FileNotFoundError(
            f"File bobot model (model.safetensors / pytorch_model.bin) tidak ditemukan di: {model_dir}."
        )

    dev = detect_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(dev)
    model.eval()
    return model, tokenizer, dev


def get_saved_threshold(model_dir: Path, default: float = 0.50) -> float:
    """Membaca optimal_threshold dari model_manifest.json atau evaluation_results.json."""
    manifest = load_model_manifest(model_dir) or load_model_manifest(model_dir.parent)
    if manifest and manifest.get("optimal_threshold") is not None:
        return float(manifest["optimal_threshold"])
    eval_file = model_dir.parent / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("optimal_threshold") is not None:
                return float(data["optimal_threshold"])
    return default


def predict_text(model, tokenizer, device: str, text: str, threshold: float = 0.50) -> Dict[str, Any]:
    """Melakukan prediksi biner dengan preprocessing sosial konsisten dan threshold terkalibrasi."""
    start_time = time.time()
    norm_text = clean_and_normalize_social_text(str(text))
    inputs = tokenizer(
        norm_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    prob_cb = float(probs[1])
    pred_id = 1 if prob_cb >= threshold else 0
    conf = prob_cb if pred_id == 1 else float(probs[0])
    latency = (time.time() - start_time) * 1000.0
    verdict = "CYBERBULLYING" if pred_id == 1 else "NON_CYBERBULLYING"

    return {
        "text": text,
        "normalized_text": norm_text,
        "prediction": verdict,
        "label": pred_id,
        "confidence": round(conf, 4),
        "prob_non_cb": round(float(probs[0]), 4),
        "prob_cb": round(prob_cb, 4),
        "latency_ms": round(latency, 2),
        "threshold": round(float(threshold), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Inferensi & Evaluasi Model IndoBERTweet / Hybrid Fusion")
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "hybrid"])
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.model_type == "hybrid":
        w_path = Path(args.model_dir)
        if w_path == DEFAULT_MODEL_DIR:
            w_path = DEFAULT_HYBRID_WEIGHTS
        elif w_path.is_dir():
            w_path = w_path / "best_hybrid_model.pt"
        model, tokenizer, device, manifest = load_hybrid_classifier_with_manifest(w_path)
        tau = args.threshold if args.threshold is not None else float((manifest or {}).get("optimal_threshold", 0.50))

        def _pred_fn(txt: str) -> Dict[str, Any]:
            res = predict_hybrid_text(model, tokenizer, device, txt, annotation_dict=None, threshold=tau, manifest=manifest)
            return {
                "text": txt,
                "prediction": "CYBERBULLYING" if res["cyberbullying"] == "YES" else "NON_CYBERBULLYING",
                "label": 1 if res["cyberbullying"] == "YES" else 0,
                "confidence": res["confidence"],
                "prob_non_cb": round(1.0 - res["prob_cyberbullying"], 4),
                "prob_cb": res["prob_cyberbullying"],
                "latency_ms": res["latency_ms"],
                "threshold": tau,
            }
    else:
        model_dir = Path(args.model_dir)
        model, tokenizer, device = load_classifier(model_dir)
        tau = args.threshold if args.threshold is not None else get_saved_threshold(model_dir, default=0.50)

        def _pred_fn(txt: str) -> Dict[str, Any]:
            return predict_text(model, tokenizer, device, txt, threshold=tau)

    if args.text:
        res = _pred_fn(args.text)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.test_csv:
        csv_path = Path(args.test_csv)
        df = pd.read_csv(csv_path)
        text_col = "comment_normalized" if "comment_normalized" in df.columns else "comment"
        probs_pos = [_pred_fn(str(t))["prob_cb"] for t in df[text_col]]
        metrics = compute_comprehensive_classification_metrics(
            y_true=df["label"].to_numpy(),
            y_prob_pos=np.array(probs_pos, dtype=float),
            threshold=tau,
            df_meta=df,
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    elif args.interactive:
        print(f"Interactive mode ({args.model_type}, tau={tau:.2f}). Ketik 'exit' untuk keluar.")
        while True:
            try:
                user_input = input("\nTeks: ").strip()
                if not user_input or user_input.lower() in {"exit", "keluar", "q"}:
                    break
                print(json.dumps(_pred_fn(user_input), indent=2, ensure_ascii=False))
            except (KeyboardInterrupt, EOFError):
                break
    else:
        sample = "makannya segentong buset mirip ursula"
        print(json.dumps(_pred_fn(sample), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
