#!/usr/bin/env python3
"""
evaluate_indobertweet.py
------------------------
Skrip pengujian dan inferensi cepat menggunakan model IndoBERTweet yang telah dilatih.
Mendukung:
1. Uji kalimat tunggal via argumen CLI.
2. Mode interaktif terminal (masukkan teks satu per satu).
3. Evaluasi komprehensif berkas CSV uji.
"""

import sys
import time
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup Project Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "output" / "models" / "indobertweet_cyberbullying" / "best_model"


def load_classifier(model_dir: Path):
    """Memuat tokenizer dan model IndoBERTweet terbaik yang telah disimpan."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Direktori model tidak ditemukan di: {model_dir}\n"
            f"Pastikan Anda telah menjalankan pelatihan: 'python3 scripts/train_indobertweet.py'."
        )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"📦 Memuat model dari: {model_dir.name} (Perangkat: {device})...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_text(model, tokenizer, device, text: str, threshold: float = 0.50):
    """Melakukan prediksi klasifikasi cyberbullying untuk sebuah teks dengan decision threshold."""
    start_time = time.time()
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    prob_cb = float(probs[1])
    pred_id = 1 if prob_cb >= threshold else 0
    conf = float(probs[pred_id])
    latency = (time.time() - start_time) * 1000

    labels = {0: "NON_CYBERBULLYING", 1: "CYBERBULLYING"}
    verdict = labels[pred_id]

    return {
        "text": text,
        "prediction": verdict,
        "label": pred_id,
        "confidence": round(conf, 4),
        "prob_non_cb": round(float(probs[0]), 4),
        "prob_cb": round(prob_cb, 4),
        "latency_ms": round(latency, 2),
        "threshold": threshold
    }


def main():
    parser = argparse.ArgumentParser(description="Inferensi & Evaluasi IndoBERTweet")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Path direktori model terbaik")
    parser.add_argument("--text", type=str, default=None, help="Teks yang ingin diuji secara langsung")
    parser.add_argument("--test_csv", type=str, default=None, help="Path ke berkas CSV yang ingin diuji")
    parser.add_argument("--interactive", action="store_true", help="Buka sesi terminal interaktif")

    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    model, tokenizer, device = load_classifier(model_dir)

    if args.text:
        res = predict_text(model, tokenizer, device, args.text)
        print("\n" + "=" * 60)
        print("🔍 HASIL PREDIKSI INDOBERTWEET")
        print("=" * 60)
        print(f"📝 Teks         : \"{res['text']}\"")
        icon = "🚨" if res["label"] == 1 else "✅"
        print(f"🏷️  Prediksi     : {icon} {res['prediction']} (Label {res['label']})")
        print(f"📊 Confidence   : {res['confidence'] * 100:.2f}%")
        print(f"📈 Probabilitas : [Non-CB: {res['prob_non_cb']:.4f} | CB: {res['prob_cb']:.4f}]")
        print(f"⏱️  Kecepatan    : {res['latency_ms']} ms")
        print("=" * 60)

    elif args.test_csv:
        csv_path = Path(args.test_csv)
        print(f"\n📑 Mengevaluasi berkas CSV: {csv_path.name}...")
        df = pd.read_csv(csv_path)
        correct = 0
        total = len(df)
        preds = []

        for _, row in df.iterrows():
            res = predict_text(model, tokenizer, device, str(row["comment"]))
            preds.append(res["label"])
            if "label" in row and row["label"] == res["label"]:
                correct += 1

        df["indobertweet_pred"] = preds
        acc = (correct / total * 100) if total > 0 else 0
        print(f"✅ Akurasi pada {total} sampel: {acc:.2f}%")

    elif args.interactive:
        print("\n" + "=" * 60)
        print("💬 SESI INTERAKTIF INDOBERTWEET CYBERBULLYING DETECTOR")
        print("Ketik komentar untuk diuji (Ketik 'exit' atau 'keluar' untuk berhenti):")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nMasukkan teks: ").strip()
                if not user_input or user_input.lower() in ["exit", "keluar", "q"]:
                    print("Selesai.")
                    break
                res = predict_text(model, tokenizer, device, user_input)
                icon = "🚨" if res["label"] == 1 else "✅"
                print(f"-> {icon} {res['prediction']} (Conf: {res['confidence']*100:.1f}% | {res['latency_ms']} ms)")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        # Default test contoh
        sample = "makannya segentong buset mirip ursula"
        print(f"\n💡 Contoh pengujian: \"{sample}\"")
        res = predict_text(model, tokenizer, device, sample)
        print(f"Hasil: {res['prediction']} (Confidence: {res['confidence']*100:.2f}%)")


if __name__ == "__main__":
    main()
