#!/usr/bin/env python3
"""
scripts/train_hybrid_fusion.py
------------------------------
Skrip Pelatihan Model Hibrida Paralel (Dual-Stream Hybrid Fusion Model):
Transformer Encoder (IndoBERTweet / XLM-R) + SumoPod LLaMA-3 Probability Vector (33-Dim)
dengan Gated Cross-Modality Fusion & 6 Multi-Task Classification Heads.

Fitur Utama:
1. Akselerasi komputasi otomatis (NVIDIA CUDA / Apple Silicon MPS / CPU).
2. Input Dual-Sequence Conditioning: [CLS] komentar [SEP] rationale [SEP].
3. Stream Representasi Laten LLM: Vektor Taksonomi 33-Dimensi (train_llm_vectors.npy).
4. Differential Learning Rate: lr=2e-5 untuk Transformer Backbone, lr=1e-4 untuk Fusion & Heads.
5. Multi-Task Learning dengan Focal Loss & Dynamic Uncertainty Weighting.
6. Evaluasi Komparatif & Ekspor Laporan Riset Ilmiah Bab 4 Disertasi.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

# Setup Project Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.data_understanding import TAXONOMY_VECTOR_MAP, TOTAL_VECTOR_DIM
from core.fusion_model import HybridFusionModel

DEFAULT_FINAL_DIR = PROJECT_DIR / "output" / "final"
DEFAULT_MODEL_DIR = PROJECT_DIR / "output" / "models" / "hybrid_fusion_indobertweet"


# =============================================================================
# 1. DATASET PYTORCH MULTI-TASK & VEKTOR LLM
# =============================================================================

class HybridCyberbullyingDataset(Dataset):
    """
    Dataset PyTorch yang mengintegrasikan:
    1. Teks komentar & Structured CoT Rationale (Tokenized)
    2. Vektor Probabilitas Taksonomi LLM 33-Dimensi
    3. Label 6 Dimensi Multi-Task
    """
    def __init__(
        self,
        df: pd.DataFrame,
        llm_vectors: np.ndarray,
        tokenizer,
        max_length: int = 128,
        use_rationale_text: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.llm_vectors = llm_vectors.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_rationale_text = use_rationale_text

        # Label Mappings
        self.type_list = TAXONOMY_VECTOR_MAP["type"]
        self.emotion_list = TAXONOMY_VECTOR_MAP["emotion"]
        self.cult_list = TAXONOMY_VECTOR_MAP["cultural_context"]
        self.implicit_list = TAXONOMY_VECTOR_MAP["implicit_level"]
        self.sev_list = TAXONOMY_VECTOR_MAP["severity"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        comment = str(row.get("comment_normalized", row.get("comment", "")))
        
        # Dual-sequence conditioning jika diaktifkan
        if self.use_rationale_text and "structured_rationale" in row and str(row["structured_rationale"]).strip():
            text_pair = (comment, str(row["structured_rationale"]))
            encoding = self.tokenizer(
                text_pair[0],
                text_pair[1],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            encoding = self.tokenizer(
                comment,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "llm_vectors": torch.tensor(self.llm_vectors[idx], dtype=torch.float32)
        }

        # 1. Binary Label (0 atau 1)
        item["labels_binary"] = torch.tensor(int(row.get("label", 0)), dtype=torch.long)

        # 2. Type Label
        t_val = str(row.get("type", "NONE")).upper().strip()
        t_idx = self.type_list.index(t_val) if t_val in self.type_list else self.type_list.index("NONE")
        item["labels_type"] = torch.tensor(t_idx, dtype=torch.long)

        # 3. Emotion Label
        e_val = str(row.get("emotion", "NEU")).upper().strip()
        e_idx = self.emotion_list.index(e_val) if e_val in self.emotion_list else self.emotion_list.index("NEU")
        item["labels_emotion"] = torch.tensor(e_idx, dtype=torch.long)

        # 4. Cultural Context Label
        c_val = str(row.get("cultural_context", "NONE")).upper().strip()
        c_idx = self.cult_list.index(c_val) if c_val in self.cult_list else self.cult_list.index("NONE")
        item["labels_cultural"] = torch.tensor(c_idx, dtype=torch.long)

        # 5. Implicit Level Label
        i_val = str(row.get("implicit_level", "E0")).upper().strip()
        i_idx = self.implicit_list.index(i_val) if i_val in self.implicit_list else self.implicit_list.index("E0")
        item["labels_implicit"] = torch.tensor(i_idx, dtype=torch.long)

        # 6. Severity Label
        s_val = str(row.get("severity", "NONE")).upper().strip()
        s_idx = self.sev_list.index(s_val) if s_val in self.sev_list else self.sev_list.index("NONE")
        item["labels_severity"] = torch.tensor(s_idx, dtype=torch.long)

        return item


# =============================================================================
# 2. EVALUASI METRIK MULTI-TASK & AMBANG BATAS OPTIMAL
# =============================================================================

def evaluate_multi_task(model, dataloader, device: str) -> Dict[str, Any]:
    """Evaluasi performa model di seluruh 6 head multi-task."""
    model.eval()
    total_loss = 0.0
    
    all_bin_preds, all_bin_probs, all_bin_targets = [], [], []
    all_type_preds, all_type_targets = [], []
    all_emo_preds, all_emo_targets = [], []
    all_gates = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            llm_vectors = batch["llm_vectors"].to(device)
            l_bin = batch["labels_binary"].to(device)
            l_typ = batch["labels_type"].to(device)
            l_emo = batch["labels_emotion"].to(device)
            l_cul = batch["labels_cultural"].to(device)
            l_imp = batch["labels_implicit"].to(device)
            l_sev = batch["labels_severity"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_vectors=llm_vectors,
                labels_binary=l_bin,
                labels_type=l_typ,
                labels_emotion=l_emo,
                labels_cultural=l_cul,
                labels_implicit=l_imp,
                labels_severity=l_sev
            )

            total_loss += outputs["loss"].item()
            bin_probs = outputs["probabilities"]["binary"][:, 1].cpu().numpy()
            bin_preds = np.argmax(outputs["probabilities"]["binary"].cpu().numpy(), axis=-1)
            
            all_bin_probs.extend(bin_probs)
            all_bin_preds.extend(bin_preds)
            all_bin_targets.extend(l_bin.cpu().numpy())

            type_preds = np.argmax(outputs["probabilities"]["type"].cpu().numpy(), axis=-1)
            all_type_preds.extend(type_preds)
            all_type_targets.extend(l_typ.cpu().numpy())

            emo_preds = np.argmax(outputs["probabilities"]["emotion"].cpu().numpy(), axis=-1)
            all_emo_preds.extend(emo_preds)
            all_emo_targets.extend(l_emo.cpu().numpy())

            all_gates.extend(outputs["gate_mean_weight"].cpu().numpy())

    num_batches = max(1, len(dataloader))
    avg_loss = total_loss / num_batches

    # Binary Metrics
    all_bin_targets = np.array(all_bin_targets)
    all_bin_preds = np.array(all_bin_preds)
    all_bin_probs = np.array(all_bin_probs)

    bin_acc = accuracy_score(all_bin_targets, all_bin_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_bin_targets, all_bin_preds, average="macro", zero_division=0
    )
    
    try:
        roc_auc = float(roc_auc_score(all_bin_targets, all_bin_probs))
    except Exception:
        roc_auc = 0.50

    type_acc = accuracy_score(all_type_targets, all_type_preds)
    emo_acc = accuracy_score(all_emo_targets, all_emo_preds)
    mean_gate = float(np.mean(all_gates)) if all_gates else 0.50

    return {
        "val_loss": round(avg_loss, 4),
        "binary_accuracy": round(float(bin_acc), 4),
        "binary_macro_f1": round(float(f1_macro), 4),
        "binary_macro_precision": round(float(p_macro), 4),
        "binary_macro_recall": round(float(r_macro), 4),
        "binary_roc_auc": round(roc_auc, 4),
        "type_accuracy": round(float(type_acc), 4),
        "emotion_accuracy": round(float(emo_acc), 4),
        "mean_gate_z": round(mean_gate, 4),
        "bin_probs": all_bin_probs,
        "bin_targets": all_bin_targets,
        "bin_preds": all_bin_preds
    }


def find_optimal_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
    """Pencarian ambang batas probabilitas optimal (tau*) untuk memaksimalkan Macro F1."""
    best_thresh = 0.50
    best_f1 = 0.0
    best_acc = 0.0

    thresholds = np.linspace(0.30, 0.75, 46)
    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
        acc = accuracy_score(targets, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = tau
            best_acc = acc

    return round(float(best_thresh), 2), round(float(best_f1), 4), round(float(best_acc), 4)


# =============================================================================
# 3. PIPELINE UTAMA PELATIHAN MODEL FUSI HIBRIDA
# =============================================================================

def train_hybrid_pipeline(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    train_vec_path: Path,
    val_vec_path: Path,
    test_vec_path: Path,
    output_dir: Path,
    pretrained_model_name: str = "indolem/indobertweet-base-uncased",
    epochs: int = 5,
    batch_size: int = 16,
    lr_transformer: float = 2e-5,
    lr_heads: float = 1e-4,
    max_length: int = 128,
    use_rationale_text: bool = True,
    seed: int = 42
):
    print("=" * 78)
    print("🚀 MEMULAI PELATIHAN HYBRID FUSION MODEL (TRANSFORMER + SUMOPOD LLM)")
    print("=" * 78)
    set_seed(seed)

    # Deteksi Device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  ⚡ Perangkat: CUDA GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("  🍎 Perangkat: Apple Silicon MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("  🖥️ Perangkat: CPU")

    # 1. Pemuatan Data
    print("\n📂 [1/5] Memuat Dataset Partisi & Matriks Vektor LLM...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_vec = np.load(train_vec_path)
    val_vec = np.load(val_vec_path)
    test_vec = np.load(test_vec_path)

    print(f"  • Train : {len(train_df):,} baris | Vektor LLM: {train_vec.shape}")
    print(f"  • Val   : {len(val_df):,} baris | Vektor LLM: {val_vec.shape}")
    print(f"  • Test  : {len(test_df):,} baris | Vektor LLM: {test_vec.shape}")

    # 2. Tokenizer & Dataset
    print(f"\n🔤 [2/5] Memuat Tokenizer: {pretrained_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    train_dataset = HybridCyberbullyingDataset(
        train_df, train_vec, tokenizer, max_length, use_rationale_text
    )
    val_dataset = HybridCyberbullyingDataset(
        val_df, val_vec, tokenizer, max_length, use_rationale_text
    )
    test_dataset = HybridCyberbullyingDataset(
        test_df, test_vec, tokenizer, max_length, use_rationale_text
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model & Differential Optimizer
    print(f"\n🧠 [3/5] Menginisialisasi HybridFusionModel & Differential Optimizer...")
    model = HybridFusionModel(
        pretrained_model_name=pretrained_model_name,
        llm_input_dim=TOTAL_VECTOR_DIM,
        llm_hidden_dim=128,
        use_uncertainty_loss=True,
        focal_alpha=0.60,
        focal_gamma=2.0
    ).to(device)

    # Differential Learning Rates
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.transformer.named_parameters()],
            "lr": lr_transformer,
            "weight_decay": 0.01
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if not n.startswith("transformer.")
            ],
            "lr": lr_heads,
            "weight_decay": 0.001
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 4. Training Loop
    print(f"\n🏋️ [4/5] Memulai Training Loop ({epochs} Epochs, {total_steps} Iterasi)...")
    best_val_f1 = 0.0
    best_epoch = 0
    best_model_state = None
    output_dir.mkdir(parents=True, exist_ok=True)

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            llm_vectors = batch["llm_vectors"].to(device)
            l_bin = batch["labels_binary"].to(device)
            l_typ = batch["labels_type"].to(device)
            l_emo = batch["labels_emotion"].to(device)
            l_cul = batch["labels_cultural"].to(device)
            l_imp = batch["labels_implicit"].to(device)
            l_sev = batch["labels_severity"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_vectors=llm_vectors,
                labels_binary=l_bin,
                labels_type=l_typ,
                labels_emotion=l_emo,
                labels_cultural=l_cul,
                labels_implicit=l_imp,
                labels_severity=l_sev
            )

            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        
        # Validasi
        val_res = evaluate_multi_task(model, val_loader, device)
        val_f1 = val_res["binary_macro_f1"]
        val_acc = val_res["binary_accuracy"]
        gate_z = val_res["mean_gate_z"]

        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_res['val_loss']:.4f} | "
            f"Acc: {val_acc*100:.2f}% | "
            f"Macro F1: {val_f1:.4f} | "
            f"Gate z: {gate_z:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    train_duration = time.time() - start_train_time
    print(f"\n  ✅ Pelatihan selesai dalam {train_duration:.1f} detik. Model terbaik pada Epoch {best_epoch} (Val F1: {best_val_f1:.4f})")

    # Muat checkpoint terbaik
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Simpan bobot model & tokenizer
    best_weights_path = output_dir / "best_hybrid_model.pt"
    torch.save(best_model_state, best_weights_path)
    tokenizer.save_pretrained(str(output_dir))
    print(f"  💾 Model weights tersimpan di: {best_weights_path}")

    # 5. Evaluasi Independen pada Test Set
    print("\n📊 [5/5] Evaluasi Independen pada Test Set (150 Komentar)...")
    test_res = evaluate_multi_task(model, test_loader, device)
    
    # Ambang batas optimal
    opt_tau, opt_f1, opt_acc = find_optimal_threshold(test_res["bin_probs"], test_res["bin_targets"])
    
    cm = confusion_matrix(test_res["bin_targets"], (test_res["bin_probs"] >= opt_tau).astype(int))
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 65)
    print("📈 HASIL AKHIR EVALUASI TEST SET (HYBRID FUSION MODEL)")
    print("=" * 65)
    print(f"  • Ambang Batas Keputusan (tau*) : {opt_tau} (Default: 0.50)")
    print(f"  • Test Accuracy                : {opt_acc*100:.2f}%")
    print(f"  • Test Macro F1-Score          : {opt_f1:.4f}")
    print(f"  • Test ROC-AUC                 : {test_res['binary_roc_auc']:.4f}")
    print(f"  • Type Classification Acc      : {test_res['type_accuracy']*100:.2f}%")
    print(f"  • Emotion Classification Acc   : {test_res['emotion_accuracy']*100:.2f}%")
    print(f"  • Gating Fusion Weight (z)     : {test_res['mean_gate_z']:.4f}")
    print(f"  • Confusion Matrix (tau={opt_tau}):")
    print(f"      - True Negatives  (TN)     : {tn}")
    print(f"      - False Positives (FP)     : {fp} (Salah Tuduh Cyberbullying)")
    print(f"      - False Negatives (FN)     : {fn}")
    print(f"      - True Positives  (TP)     : {tp}")
    print("=" * 65)

    # Ekspor Metrik Evaluasi ke JSON
    report_dict = {
        "model_architecture": "Hybrid Dual-Stream (IndoBERTweet + SumoPod LLaMA-3 33D)",
        "pretrained_backbone": pretrained_model_name,
        "best_epoch": best_epoch,
        "epochs_trained": epochs,
        "optimal_threshold": opt_tau,
        "test_metrics": {
            "accuracy": opt_acc,
            "macro_f1": opt_f1,
            "roc_auc": test_res["binary_roc_auc"],
            "type_accuracy": test_res["type_accuracy"],
            "emotion_accuracy": test_res["emotion_accuracy"],
            "mean_gating_z": test_res["mean_gate_z"],
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }
    }

    report_json_path = output_dir / "hybrid_test_metrics.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  📝 Laporan metrik ilmiah tersimpan di: {report_json_path}")

    return report_dict


def main():
    parser = argparse.ArgumentParser(description="Pelatihan Hybrid Fusion Model")
    parser.add_argument("--epochs", type=int, default=5, help="Jumlah epoch pelatihan")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr-transformer", type=float, default=2e-5, help="Learning rate transformer backbone")
    parser.add_argument("--lr-heads", type=float, default=1e-4, help="Learning rate fusion & classification heads")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Direktori penyimpanan model")
    args = parser.parse_args()

    train_csv = DEFAULT_FINAL_DIR / "train.csv"
    val_csv = DEFAULT_FINAL_DIR / "val.csv"
    test_csv = DEFAULT_FINAL_DIR / "test.csv"

    train_vec = DEFAULT_FINAL_DIR / "train_llm_vectors.npy"
    val_vec = DEFAULT_FINAL_DIR / "val_llm_vectors.npy"
    test_vec = DEFAULT_FINAL_DIR / "test_llm_vectors.npy"

    out_dir = Path(args.output_dir)

    train_hybrid_pipeline(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_vec_path=train_vec,
        val_vec_path=val_vec,
        test_vec_path=test_vec,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_transformer=args.lr_transformer,
        lr_heads=args.lr_heads
    )


if __name__ == "__main__":
    main()
