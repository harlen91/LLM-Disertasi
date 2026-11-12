#!/usr/bin/env python3
"""
train_indobertweet.py
---------------------
Fine-tuning pipeline untuk model Transformer Encoder IndoBERTweet
(indolem/indobertweet-base-uncased) menggunakan dataset IICD yang telah
dikurasi dan dianotasi oleh LLM SumoPod (LLaMA-3).

Fitur:
1. Akselerasi komputasi otomatis (NVIDIA CUDA / Apple Silicon MPS / CPU).
2. Tokenisasi teks komentar Twitter/medsos (max_length=128).
3. Optimasi AdamW dengan linear learning rate warmup.
4. Evaluasi komprehensif: Macro-F1, Precision, Recall, Confusion Matrix.
5. Penyimpanan model terbaik berdasarkan metrik evaluasi tertinggi.
6. Evaluasi independen pada data uji (test.csv) dengan ekspor tabel metrik riset.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)

# Setup Project Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_FINAL_DIR = PROJECT_DIR / "output" / "final"
DEFAULT_MODEL_DIR = PROJECT_DIR / "output" / "models" / "indobertweet_cyberbullying"


class CyberbullyingDataset(Dataset):
    """PyTorch Dataset untuk klasifikasi komentar cyberbullying dengan confidence LLaMA-3."""
    def __init__(self, texts, labels, tokenizer, max_length: int = 128, confidences=None):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.confidences = list(confidences) if confidences is not None else [0.90] * len(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        conf = float(self.confidences[idx]) if idx < len(self.confidences) else 0.90

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["confidences"] = torch.tensor(conf, dtype=torch.float32)
        return item


class FocalLoss(torch.nn.Module):
    """
    Focal Loss dengan Distilasi Soft Label Terbobot Confidence LLM.
    Mengatasi isu False Positives dan sampel sulit dengan:
    1. Focusing parameter gamma (meredam loss sampel mudah, memperbesar penalti hard samples).
    2. Alpha class weighting (penyeimbang kelas positif/negatif).
    3. Soft labels berbobot confidence LLaMA-3 (mencegah overconfidence pada teks informal/ambigu).
    """
    def __init__(self, alpha: float = 0.55, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, confidences: torch.Tensor = None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        if confidences is not None:
            conf = torch.clamp(confidences, 0.70, 0.98)
            smooth_targets = torch.zeros_like(logits)
            smooth_targets[:, 1] = torch.where(targets == 1, conf, 1.0 - conf)
            smooth_targets[:, 0] = torch.where(targets == 0, conf, 1.0 - conf)
        else:
            num_classes = logits.size(-1)
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        pt = torch.sum(probs * smooth_targets, dim=-1)
        focal_weight = (1.0 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        loss_per_sample = -torch.sum(smooth_targets * log_probs, dim=-1)
        weighted_loss = alpha_weight * focal_weight * loss_per_sample
        return weighted_loss.mean()


class FocalLossTrainer(Trainer):
    """HuggingFace Trainer yang menggunakan FocalLoss dan Confidence-Weighted Label Smoothing."""
    def __init__(self, *args, focal_alpha: float = 0.55, focal_gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        confidences = inputs.get("confidences", None)
        model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "confidences"]}
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss = self.focal_loss_fn(logits, labels, confidences)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Menghitung metrik performa multi-dimensi untuk naskah publikasi ilmiah."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "weighted_f1": round(float(f1_w), 4)
    }


def detect_device() -> str:
    """Deteksi perangkat keras terbaik yang tersedia."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"  ⚡ Perangkat: CUDA GPU ({device_name})")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("  🍎 Perangkat: Apple Silicon MPS (Metal Performance Shaders)")
        return "mps"
    else:
        print("  🖥️ Perangkat: CPU (Standard)")
        return "cpu"


def train_model(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    model_name: str = "indolem/indobertweet-base-uncased",
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    seed: int = 42,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.55,
    focal_gamma: float = 2.0
):
    print("=" * 75)
    print("🚀 FINE-TUNING INDOBERTWEET TRANSFORMER ENCODER UNTUK IICD CYBERBULLYING")
    print("=" * 75)

    set_seed(seed)
    device = detect_device()

    # 1. Memuat Partisi Dataset
    print("\n📂 [1/5] Memuat Partisi Dataset...")
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Dataset partisi tidak ditemukan di {train_path.parent}. "
            f"Jalankan 'python3 scripts/curate_dataset.py' terlebih dahulu."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path) if test_path and test_path.exists() else None

    print(f"  • Data Latih (Train)     : {len(train_df):,} baris")
    print(f"  • Data Validasi (Val)    : {len(val_df):,} baris")
    if test_df is not None:
        print(f"  • Data Uji (Test)        : {len(test_df):,} baris")

    # 2. Inisialisasi Tokenizer & Model
    print(f"\n🧠 [2/5] Mengunduh/Memuat Model & Tokenizer: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id2label = {0: "NON_CYBERBULLYING", 1: "CYBERBULLYING"}
    label2id = {"NON_CYBERBULLYING": 0, "CYBERBULLYING": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # 3. Pembuatan PyTorch Dataset dengan Confidence LLM
    print("\n📦 [3/5] Mengonversi Data ke PyTorch Tensor dengan Bobot Confidence...")
    train_conf = train_df["confidence"] if "confidence" in train_df.columns else None
    val_conf = val_df["confidence"] if "confidence" in val_df.columns else None
    test_conf = test_df["confidence"] if test_df is not None and "confidence" in test_df.columns else None

    train_dataset = CyberbullyingDataset(train_df["comment"], train_df["label"], tokenizer, max_length, confidences=train_conf)
    val_dataset = CyberbullyingDataset(val_df["comment"], val_df["label"], tokenizer, max_length, confidences=val_conf)
    test_dataset = CyberbullyingDataset(test_df["comment"], test_df["label"], tokenizer, max_length, confidences=test_conf) if test_df is not None else None

    # 4. Konfigurasi Hyperparameter Training
    print("\n⚙️ [4/5] Mengonfigurasi Hyperparameter Pelatihan...")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best_model"

    # Penyesuaian training arguments yang kompatibel dengan Hugging Face terbaru
    eval_arg_name = "eval_strategy" if hasattr(TrainingArguments, "eval_strategy") else "evaluation_strategy"
    eval_kwargs = {eval_arg_name: "epoch"}

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=str(output_dir / "logs"),
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=seed,
        dataloader_pin_memory=False if device == "mps" else True,
        report_to="none",
        **eval_kwargs
    )

    if use_focal_loss:
        print(f"  🎯 Menggunakan FocalLossTrainer (gamma={focal_gamma}, alpha={focal_alpha}) dengan Confidence Distillation.")
        trainer = FocalLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )
    else:
        print("  ℹ️ Menggunakan Standard Cross-Entropy Trainer.")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

    # 5. Eksekusi Pelatihan
    print("\n🚀 [5/5] Memulai Proses Fine-Tuning Transformer...")
    print(f"  • Epochs         : {epochs}")
    print(f"  • Batch Size     : {batch_size}")
    print(f"  • Learning Rate  : {learning_rate}")
    print(f"  • Max Sequence   : {max_length} tokens\n")

    train_result = trainer.train()

    # Evaluasi pada validation set
    print("\n📊 Mengevaluasi performa model terbaik pada Validation Set:")
    val_metrics = trainer.evaluate(val_dataset)
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  • {k:22}: {v:.4f}")

    # Simpan model & tokenizer terbaik
    print(f"\n💾 Menyimpan model terbaik ke: {best_model_dir}...")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    # Optimasi Decision Threshold pada Validation Set
    val_predictions = trainer.predict(val_dataset)
    val_probs = torch.softmax(torch.tensor(val_predictions.predictions), dim=-1).numpy()[:, 1]
    val_y_true = val_predictions.label_ids

    best_thresh = 0.50
    best_val_f1 = val_metrics.get("eval_macro_f1", 0.0)
    for th in np.arange(0.30, 0.70, 0.02):
        th_preds = (val_probs >= th).astype(int)
        _, _, th_f1, _ = precision_recall_fscore_support(val_y_true, th_preds, average="macro", zero_division=0)
        if th_f1 > best_val_f1:
            best_val_f1 = th_f1
            best_thresh = round(float(th), 2)

    print(f"\n🎯 Optimal Decision Threshold pada Validation Set: {best_thresh:.2f} (Val Macro-F1: {best_val_f1:.4f})")

    # 6. Evaluasi Independen pada Test Set (Gold Evaluation)
    test_report_dict = {}
    if test_dataset is not None:
        print("\n" + "=" * 75)
        print("🎯 EVALUASI AKHIR PADA INDEPENDENT TEST SET (UNSEEN DATA)")
        print("=" * 75)

        predictions_output = trainer.predict(test_dataset)
        y_true = predictions_output.label_ids
        test_probs = torch.softmax(torch.tensor(predictions_output.predictions), dim=-1).numpy()[:, 1]

        # 1. Metrik Standar (Threshold 0.50)
        y_pred_def = np.argmax(predictions_output.predictions, axis=1)
        test_metrics_def = compute_metrics((predictions_output.predictions, y_true))

        # 2. Metrik Optimal (Threshold best_thresh)
        y_pred_opt = (test_probs >= best_thresh).astype(int)
        acc_opt = accuracy_score(y_true, y_pred_opt)
        p_opt, r_opt, f1_opt, _ = precision_recall_fscore_support(y_true, y_pred_opt, average="macro", zero_division=0)
        _, _, f1_w_opt, _ = precision_recall_fscore_support(y_true, y_pred_opt, average="weighted", zero_division=0)

        test_metrics_opt = {
            "accuracy": round(float(acc_opt), 4),
            "macro_f1": round(float(f1_opt), 4),
            "macro_precision": round(float(p_opt), 4),
            "macro_recall": round(float(r_opt), 4),
            "weighted_f1": round(float(f1_w_opt), 4)
        }

        print(f"  • Standard (Threshold 0.50) -> Akurasi: {test_metrics_def['accuracy']*100:.2f}% | Macro-F1: {test_metrics_def['macro_f1']:.4f}")
        print(f"  • Optimized (Threshold {best_thresh:.2f}) -> Akurasi: {test_metrics_opt['accuracy']*100:.2f}% | Macro-F1: {test_metrics_opt['macro_f1']:.4f}")

        # Classification Report pada Threshold Optimal
        cls_report_text = classification_report(
            y_true, y_pred_opt,
            target_names=["Non-Cyberbullying (0)", "Cyberbullying (1)"],
            digits=4
        )
        print("\n📄 Detailed Classification Report (Optimal Threshold):")
        print(cls_report_text)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_opt)
        print("🔲 Confusion Matrix:")
        print(f"  [TN={cm[0][0]:3d} | FP={cm[0][1]:3d}]")
        print(f"  [FN={cm[1][0]:3d} | TP={cm[1][1]:3d}]")

        # Simpan Laporan Evaluasi ke JSON
        report_file = output_dir / "evaluation_results.json"
        summary = {
            "model_name": model_name,
            "epochs_trained": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimal_threshold": best_thresh,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "validation_metrics": val_metrics,
            "test_metrics_default": test_metrics_def,
            "test_metrics_optimized": test_metrics_opt,
            "confusion_matrix": cm.tolist()
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n📑 Ringkasan metrik berhasil diekspor ke: {report_file}")


    print("\n" + "=" * 75)
    print("🎉 PELATIHAN INDOBERTWEET BERHASIL SELESAI!")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning IndoBERTweet pada dataset IICD")
    parser.add_argument("--train_csv", type=str, default=str(DEFAULT_FINAL_DIR / "train.csv"))
    parser.add_argument("--val_csv", type=str, default=str(DEFAULT_FINAL_DIR / "val.csv"))
    parser.add_argument("--test_csv", type=str, default=str(DEFAULT_FINAL_DIR / "test.csv"))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model_name", type=str, default="indolem/indobertweet-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_model(
        train_path=Path(args.train_csv),
        val_path=Path(args.val_csv),
        test_path=Path(args.test_csv),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
