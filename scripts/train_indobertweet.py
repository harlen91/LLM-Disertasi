#!/usr/bin/env python3
"""
train_indobertweet.py
---------------------
Fine-tuning pipeline untuk model baseline Transformer Encoder IndoBERTweet
(indolem/indobertweet-base-uncased) pada dataset IICD terkurasi.

Perbaikan Metodologis & Reproduktibilitas:
1. Pemeriksaan otomatis bebas kebocoran data lintas train / val / test.
2. Konsistensi preprocessing training & inference (`comment_normalized`).
3. Tuning threshold keputusan STRICTLY pada Validation Set (bukan Test Set).
4. Checkpoint resume (`--resume_from_checkpoint`) dan Early Stopping.
5. Metrik lengkap (Accuracy, Precision, Recall, Macro/Weighted F1, ROC-AUC, PR-AUC,
   Calibration ECE/MCE/Brier/LogLoss, Bootstrap 95% CI, E0/E1/E2, FP/FN error analysis).
6. Penyimpanan `model_manifest.json` lengkap untuk audit & inferensi.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import DEFAULT_RANDOM_SEED, INDOBERTWEET_PRETRAINED_MODEL
from core.data_understanding import clean_and_normalize_social_text
from core.evaluation_metrics import (
    build_model_manifest,
    compute_comprehensive_classification_metrics,
    find_optimal_threshold_on_val,
    save_model_manifest,
)
from scripts.curate_dataset import verify_no_cross_split_leakage

DEFAULT_FINAL_DIR = PROJECT_DIR / "output" / "final"
DEFAULT_MODEL_DIR = PROJECT_DIR / "output" / "models" / "indobertweet_cyberbullying"


class CyberbullyingDataset(Dataset):
    """PyTorch Dataset untuk baseline teks-only IndoBERTweet dengan preprocessing konsisten."""

    def __init__(self, texts, labels, tokenizer, max_length: int = 128, confidences=None):
        self.texts = [clean_and_normalize_social_text(str(t)) for t in texts]
        self.labels = [int(y) for y in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.confidences = list(confidences) if confidences is not None else [0.90] * len(self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        raw_conf = self.confidences[idx] if idx < len(self.confidences) else 0.90
        conf = 0.90 if pd.isna(raw_conf) else float(raw_conf)

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["confidences"] = torch.tensor(conf, dtype=torch.float32)
        return item


class FocalLoss(torch.nn.Module):
    """Focal Loss dengan soft label smoothing berbobot confidence (opsional)."""

    def __init__(self, alpha: float = 0.55, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        return (alpha_weight * focal_weight * loss_per_sample).mean()


class FocalLossTrainer(Trainer):
    """HuggingFace Trainer dengan FocalLoss terbobot confidence."""

    def __init__(self, *args, focal_alpha: float = 0.55, focal_gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        confidences = inputs.get("confidences", None)
        model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "confidences"]}
        outputs = model(**model_inputs)
        loss = self.focal_loss_fn(outputs.logits, labels, confidences)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Menghitung metrik cepat per epoch selama training."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_m), 4),
        "macro_precision": round(float(p_m), 4),
        "macro_recall": round(float(r_m), 4),
        "weighted_f1": round(float(f1_w), 4),
    }


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_latest_hf_checkpoint(checkpoints_dir: Path) -> Optional[str]:
    if not checkpoints_dir.exists():
        return None
    ckpts = sorted(
        [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(ckpts[-1]) if ckpts else None


def train_model(
    train_path: Path,
    val_path: Path,
    test_path: Optional[Path],
    output_dir: Path,
    model_name: str = INDOBERTWEET_PRETRAINED_MODEL,
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    seed: int = DEFAULT_RANDOM_SEED,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.55,
    focal_gamma: float = 2.0,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 75)
    print("🚀 FINE-TUNING BASELINE INDOBERTWEET (TEXT-ONLY)")
    print("=" * 75)

    set_seed(seed)
    device = detect_device()
    print(f"  ⚙️ Device: {device} | Seed: {seed}")

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Dataset partisi tidak ditemukan di {train_path.parent}. Jalankan scripts/curate_dataset.py."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path) if test_path and test_path.exists() else None

    if test_df is not None:
        leakage_info = verify_no_cross_split_leakage(train_df, val_df, test_df)
        print(f"  ✅ Cross-split leakage verification: {leakage_info}")

    # Load dataset manifest metadata if present
    ds_manifest_path = train_path.parent / "dataset_manifest.json"
    dataset_version = "unspecified"
    target_source = str(train_df["target_source"].iloc[0]) if "target_source" in train_df.columns and len(train_df) > 0 else "pseudo_labeled_llm"
    if ds_manifest_path.exists():
        with open(ds_manifest_path, "r", encoding="utf-8") as f:
            ds_m = json.load(f)
            dataset_version = ds_m.get("dataset_version", dataset_version)
            target_source = ds_m.get("label_source", target_source)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NON_CYBERBULLYING", 1: "CYBERBULLYING"},
        label2id={"NON_CYBERBULLYING": 0, "CYBERBULLYING": 1},
    )

    text_col = "comment_normalized" if "comment_normalized" in train_df.columns else "comment"
    train_conf = train_df["confidence"] if "confidence" in train_df.columns else None
    val_conf = val_df["confidence"] if "confidence" in val_df.columns else None
    test_conf = test_df["confidence"] if test_df is not None and "confidence" in test_df.columns else None

    train_dataset = CyberbullyingDataset(train_df[text_col], train_df["label"], tokenizer, max_length, confidences=train_conf)
    val_dataset = CyberbullyingDataset(val_df[text_col], val_df["label"], tokenizer, max_length, confidences=val_conf)
    test_dataset = (
        CyberbullyingDataset(test_df[text_col], test_df["label"], tokenizer, max_length, confidences=test_conf)
        if test_df is not None
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best_model"

    eval_arg_name = "eval_strategy" if hasattr(TrainingArguments, "eval_strategy") else "evaluation_strategy"
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
        **{eval_arg_name: "epoch"},
    )

    trainer_cls = FocalLossTrainer if use_focal_loss else Trainer
    trainer_extra = {"focal_alpha": focal_alpha, "focal_gamma": focal_gamma} if use_focal_loss else {}

    # Compatibility across transformers versions (`processing_class` vs `tokenizer`)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
        **trainer_extra,
    }
    if "processing_class" in Trainer.__init__.__code__.co_varnames:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = trainer_cls(**trainer_kwargs)

    resume_arg: Any = None
    if resume_from_checkpoint == "auto":
        resume_arg = _resolve_latest_hf_checkpoint(checkpoints_dir)
    elif resume_from_checkpoint:
        resume_arg = resume_from_checkpoint

    trainer.train(resume_from_checkpoint=resume_arg)

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    # Optimasi threshold STRICTLY pada Validation Set
    val_predictions = trainer.predict(val_dataset)
    val_probs = torch.softmax(torch.tensor(val_predictions.predictions), dim=-1).numpy()[:, 1]
    val_y_true = val_predictions.label_ids
    best_thresh, best_val_f1, best_val_acc = find_optimal_threshold_on_val(val_probs, val_y_true)
    val_full_metrics = compute_comprehensive_classification_metrics(
        y_true=val_y_true,
        y_prob_pos=val_probs,
        threshold=best_thresh,
        df_meta=val_df,
        seed=seed,
    )
    print(f"\n🎯 Threshold optimal pada Validation Set: tau={best_thresh:.2f} (Val Macro-F1={best_val_f1:.4f}, Val Acc={best_val_acc*100:.2f}%)")

    test_metrics_full: Dict[str, Any] = {}
    if test_dataset is not None and test_df is not None:
        test_predictions = trainer.predict(test_dataset)
        test_probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=-1).numpy()[:, 1]
        test_y_true = test_predictions.label_ids
        test_metrics_full = compute_comprehensive_classification_metrics(
            y_true=test_y_true,
            y_prob_pos=test_probs,
            threshold=best_thresh,
            df_meta=test_df,
            seed=seed,
        )
        print(
            f"📊 Test Set (tau={best_thresh:.2f}) -> Acc={test_metrics_full['accuracy']*100:.2f}% | "
            f"Macro-F1={test_metrics_full['macro_f1']:.4f} | ROC-AUC={test_metrics_full['roc_auc']} | "
            f"PR-AUC={test_metrics_full['pr_auc']} | ECE={test_metrics_full['calibration']['ece']}"
        )

    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "use_focal_loss": use_focal_loss,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "text_column": text_col,
    }
    summary = {
        "model_architecture": "IndoBERTweet Baseline (Text-Only Binary SequenceClassifier)",
        "pretrained_model": model_name,
        "dataset_version": dataset_version,
        "target_source": target_source,
        "optimal_threshold": best_thresh,
        "threshold_selection_split": "validation",
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)) if test_df is not None else 0,
        "validation_metrics": val_full_metrics,
        "test_metrics_optimized": test_metrics_full,
        "hyperparameters": hyperparameters,
    }

    report_file = output_dir / "evaluation_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest = build_model_manifest(
        model_architecture="IndoBERTweet Baseline SequenceClassifier",
        pretrained_model=model_name,
        checkpoint_location=best_model_dir,
        hyperparameters=hyperparameters,
        evaluation_metrics=test_metrics_full or val_full_metrics,
        optimal_threshold=best_thresh,
        dataset_version=dataset_version,
        target_source=target_source,
        random_seed=seed,
    )
    save_model_manifest(best_model_dir, manifest)
    save_model_manifest(output_dir, manifest)
    print(f"💾 Model Manifest tersimpan di: {best_model_dir / 'model_manifest.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning IndoBERTweet Baseline pada dataset IICD")
    parser.add_argument("--train_csv", type=str, default=str(DEFAULT_FINAL_DIR / "train.csv"))
    parser.add_argument("--val_csv", type=str, default=str(DEFAULT_FINAL_DIR / "val.csv"))
    parser.add_argument("--test_csv", type=str, default=str(DEFAULT_FINAL_DIR / "test.csv"))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model_name", type=str, default=INDOBERTWEET_PRETRAINED_MODEL)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--resume", type=str, default=None, help="'auto' atau path checkpoint HuggingFace")

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
        seed=args.seed,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
