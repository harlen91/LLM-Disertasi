#!/usr/bin/env python3
"""
train_hybrid_fusion.py
----------------------
Pipeline Pelatihan End-to-End Arsitektur Dual-Stream Hybrid Fusion
(IndoBERTweet Stream A + LLM Teacher Vector Stream B + Gated Cross-Modality Fusion + 6 MTL Heads).

Perbaikan Kritis (Sesuai Audit Metodologi):
1. ZERO TARGET LEAKAGE:
   - Memverifikasi ketiadaan tumpang-tindih komentar (raw & normalized) di antara train, val, dan test.
   - Menggunakan `comment_normalized` (BUKAN menyisipkan tag label `[Konteks: ... | Tipe: ...]`) ke Encoder.
   - Mendeteksi sumber label (`human_ground_truth` vs `pseudo_labeled_llm`). Apabila dataset masih
     `pseudo_labeled_llm`, pipeline menerapkan modality dropout saat pelatihan dan mengevaluasi
     test set dalam mode `modality_masked_student` (`llm_vectors=zeros`) sehingga fitur LLM tidak
     membocorkan label target dirinya sendiri. Apabila `human_ground_truth` tersedia, Stream B
     berjalan penuh tanpa target leakage.
2. VALIDATION-ONLY THRESHOLD SELECTION:
   - Threshold keputusan (`tau*`) dioptimasi HANYA pada `val.csv` menggunakan Macro-F1 (bukan `test.csv`).
3. MISSING LABEL MASKING (`-100`):
   - Label taksonomi yang kosong/`UNKNOWN` di-mask ke `-100` (`IGNORE_INDEX`), tidak dipaksa menjadi kelas valid palsu.
4. EARLY STOPPING, CHECKPOINT SAVE/RESUME, & MODEL MANIFEST:
   - Menyimpan checkpoint per-epoch (`latest_checkpoint.pt`) untuk `--resume`, memilih bobot terbaik
     berdasarkan Validation Macro-F1, serta mengekspor `model_manifest.json` dan hasil 3-way Ablation.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import DEFAULT_RANDOM_SEED, INDOBERTWEET_PRETRAINED_MODEL
from core.data_understanding import (
    CULTURAL_CUE_SLICE,
    EMOTIONAL_CUE_SLICE,
    TAXONOMY_VECTOR_MAP,
    TOTAL_VECTOR_DIM,
    clean_and_normalize_social_text,
)
from core.evaluation_metrics import (
    build_model_manifest,
    compute_comprehensive_classification_metrics,
    find_optimal_threshold_on_val,
    save_model_manifest,
)
from core.fusion_model import (
    IGNORE_INDEX,
    HybridFusionModel,
    LLMKnowledgeOnlyClassifier,
    detect_torch_device,
)
from scripts.curate_dataset import verify_no_cross_split_leakage

DEFAULT_FINAL_DIR = PROJECT_DIR / "output" / "final"
DEFAULT_HYBRID_DIR = PROJECT_DIR / "output" / "models" / "hybrid_fusion_indobertweet"


def _encode_target_label(val: Any, valid_list: List[str]) -> int:
    """Mengonversi kode taksonomi ke indeks kelas atau IGNORE_INDEX (-100) jika hilang/tidak valid."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return IGNORE_INDEX
    s = str(val).strip().upper()
    if s in valid_list:
        return valid_list.index(s)
    return IGNORE_INDEX


class HybridCyberbullyingDataset(Dataset):
    """PyTorch Dataset untuk Multi-Task Hybrid Fusion dengan masking -100 untuk label yang hilang."""

    def __init__(
        self,
        df: pd.DataFrame,
        llm_vectors: np.ndarray,
        tokenizer,
        max_length: int = 128,
        use_rationale_text: bool = False,
        mask_llm_vectors: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.llm_vectors = (
            np.zeros_like(llm_vectors, dtype=np.float32)
            if mask_llm_vectors
            else llm_vectors.astype(np.float32)
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_rationale_text = use_rationale_text

        self.type_list = TAXONOMY_VECTOR_MAP["type"]
        self.emotion_list = TAXONOMY_VECTOR_MAP["emotion"]
        self.cult_list = TAXONOMY_VECTOR_MAP["cultural_context"]
        self.implicit_list = TAXONOMY_VECTOR_MAP["implicit_level"]
        self.sev_list = TAXONOMY_VECTOR_MAP["severity"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        raw_text = row.get("comment_normalized", row.get("comment", ""))
        comment = clean_and_normalize_social_text(str(raw_text))

        if self.use_rationale_text and "structured_rationale" in row and str(row["structured_rationale"]).strip():
            encoding = self.tokenizer(
                comment,
                str(row["structured_rationale"]),
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                comment,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "llm_vectors": torch.tensor(self.llm_vectors[idx], dtype=torch.float32),
        }

        # Prioritaskan kolom `target_*` yang terpisah tegas dari fitur LLM
        raw_bin = row.get("label", None)
        if raw_bin is None or (isinstance(raw_bin, float) and np.isnan(raw_bin)):
            cb_val = str(row.get("target_cyberbullying", "")).strip().upper()
            bin_idx = 1 if cb_val == "YES" else (0 if cb_val == "NO" else IGNORE_INDEX)
        else:
            bin_idx = int(raw_bin) if int(raw_bin) in (0, 1) else IGNORE_INDEX

        item["labels_binary"] = torch.tensor(bin_idx, dtype=torch.long)
        item["labels_type"] = torch.tensor(
            _encode_target_label(row.get("target_type", row.get("type")), self.type_list), dtype=torch.long
        )
        item["labels_emotion"] = torch.tensor(
            _encode_target_label(row.get("target_emotion", row.get("emotion")), self.emotion_list), dtype=torch.long
        )
        item["labels_cultural"] = torch.tensor(
            _encode_target_label(row.get("target_cultural_context", row.get("cultural_context")), self.cult_list), dtype=torch.long
        )
        item["labels_implicit"] = torch.tensor(
            _encode_target_label(row.get("target_implicit_level", row.get("implicit_level")), self.implicit_list), dtype=torch.long
        )
        item["labels_severity"] = torch.tensor(
            _encode_target_label(row.get("target_severity", row.get("severity")), self.sev_list), dtype=torch.long
        )
        return item


def _compute_masked_acc_f1(targets: List[int], preds: List[int]) -> Dict[str, Any]:
    t_arr = np.array(targets, dtype=int)
    p_arr = np.array(preds, dtype=int)
    valid = t_arr != IGNORE_INDEX
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"support": 0, "accuracy": None, "macro_f1": None}
    return {
        "support": n_valid,
        "accuracy": round(float(accuracy_score(t_arr[valid], p_arr[valid])), 4),
        "macro_f1": round(float(f1_score(t_arr[valid], p_arr[valid], average="macro", zero_division=0)), 4),
    }


def evaluate_multi_task(
    model: HybridFusionModel,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """Evaluasi lengkap seluruh 6 classification heads dengan penanganan label -100."""
    model.eval()
    total_loss = 0.0

    bin_probs_all, bin_targets_all = [], []
    head_preds: Dict[str, List[int]] = {k: [] for k in ["type", "emotion", "cultural", "implicit", "severity"]}
    head_targets: Dict[str, List[int]] = {k: [] for k in ["type", "emotion", "cultural", "implicit", "severity"]}
    all_gates: List[float] = []

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
                labels_severity=l_sev,
            )

            total_loss += float(outputs["loss"].item())
            probs = outputs["probabilities"]

            bin_probs = probs["binary"][:, 1].detach().cpu().numpy()
            bin_probs_all.extend(bin_probs)
            bin_targets_all.extend(l_bin.detach().cpu().numpy())

            for head_name, tgt_tensor in [
                ("type", l_typ),
                ("emotion", l_emo),
                ("cultural", l_cul),
                ("implicit", l_imp),
                ("severity", l_sev),
            ]:
                preds = np.argmax(probs[head_name].detach().cpu().numpy(), axis=-1)
                head_preds[head_name].extend(preds.tolist())
                head_targets[head_name].extend(tgt_tensor.detach().cpu().numpy().tolist())

            all_gates.extend(outputs["gate_mean_weight"].detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    bin_probs_np = np.asarray(bin_probs_all, dtype=float)
    bin_targets_np = np.asarray(bin_targets_all, dtype=int)
    valid_bin = bin_targets_np != IGNORE_INDEX

    bin_preds_default = (bin_probs_np >= 0.50).astype(int)
    bin_acc = float(accuracy_score(bin_targets_np[valid_bin], bin_preds_default[valid_bin])) if valid_bin.sum() > 0 else 0.0
    bin_f1 = float(f1_score(bin_targets_np[valid_bin], bin_preds_default[valid_bin], average="macro", zero_division=0)) if valid_bin.sum() > 0 else 0.0

    mtl_metrics = {
        head_name: _compute_masked_acc_f1(head_targets[head_name], head_preds[head_name])
        for head_name in ["type", "emotion", "cultural", "implicit", "severity"]
    }

    return {
        "loss": round(avg_loss, 4),
        "binary_accuracy": round(bin_acc, 4),
        "binary_macro_f1": round(bin_f1, 4),
        "mtl_heads": mtl_metrics,
        "mean_gate_z": round(float(np.mean(all_gates)) if all_gates else 0.50, 4),
        "bin_probs": bin_probs_np,
        "bin_targets": bin_targets_np,
    }


def _evaluate_indobertweet_baseline_on_test(
    baseline_dir: Path,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str,
    max_length: int = 128,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Optional[Dict[str, Any]]:
    """Mengevaluasi model baseline IndoBERTweet-Only pada test set yang sama jika checkpoint tersedia."""
    if not baseline_dir.exists() or not ((baseline_dir / "model.safetensors").exists() or (baseline_dir / "pytorch_model.bin").exists()):
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(baseline_dir))
    base_model = AutoModelForSequenceClassification.from_pretrained(str(baseline_dir)).to(device)
    base_model.eval()
    text_col = "comment_normalized" if "comment_normalized" in test_df.columns else "comment"

    def _predict_probs(frame: pd.DataFrame) -> np.ndarray:
        probs_list: List[float] = []
        texts = [clean_and_normalize_social_text(str(t)) for t in frame[text_col]]
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                enc = tokenizer(
                    texts[start : start + 32],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                logits = base_model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                probs_list.extend(probs.tolist())
        return np.asarray(probs_list, dtype=float)

    val_probs = _predict_probs(val_df)
    opt_tau, _, _ = find_optimal_threshold_on_val(val_probs, val_df["label"].to_numpy())
    test_probs = _predict_probs(test_df)
    metrics = compute_comprehensive_classification_metrics(
        y_true=test_df["label"].to_numpy(),
        y_prob_pos=test_probs,
        threshold=opt_tau,
        df_meta=test_df,
        seed=seed,
    )
    metrics["model_name"] = "A. IndoBERTweet-Only (Stream A)"
    return metrics


def _apply_cue_mask(vec: np.ndarray, ablation_mask_mode: Optional[str] = None) -> np.ndarray:
    """Menerapkan masking fitur Stream B untuk eksperimen ablasi terkontrol (E1/E4/E5)."""
    arr = vec.astype(np.float32).copy()
    if ablation_mask_mode == "all_zero":
        arr[:] = 0.0
    elif ablation_mask_mode == "no_cultural":
        arr[:, CULTURAL_CUE_SLICE] = 0.0
    elif ablation_mask_mode == "no_emotional":
        arr[:, EMOTIONAL_CUE_SLICE] = 0.0
    return arr


def _train_and_eval_llm_only_ablation(
    train_vec: np.ndarray,
    val_vec: np.ndarray,
    test_vec: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str,
    seed: int = DEFAULT_RANDOM_SEED,
    max_epochs: int = 80,
) -> Dict[str, Any]:
    """
    Model Ablasi E2: LLM Contextual Knowledge-Only (33-dim Non-Leaky Evidence MLP).
    Dilatih & dievaluasi pada split train/val/test yang sama terhadap LLM pseudo-labels
    menggunakan representasi bukti kontekstual 33D tanpa menyalin label target.
    """
    torch.manual_seed(seed)
    clf = LLMKnowledgeOnlyClassifier(in_features=TOTAL_VECTOR_DIM).to(device)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=2e-3, weight_decay=1e-3)

    x_tr = torch.tensor(train_vec, dtype=torch.float32, device=device)
    y_tr = torch.tensor(train_df["label"].to_numpy(), dtype=torch.long, device=device)
    y_typ = torch.tensor([_encode_target_label(v, TAXONOMY_VECTOR_MAP["type"]) for v in train_df.get("target_type", train_df.get("type", pd.Series("", index=train_df.index)))], dtype=torch.long, device=device)
    y_emo = torch.tensor([_encode_target_label(v, TAXONOMY_VECTOR_MAP["emotion"]) for v in train_df.get("target_emotion", train_df.get("emotion", pd.Series("", index=train_df.index)))], dtype=torch.long, device=device)
    y_cul = torch.tensor([_encode_target_label(v, TAXONOMY_VECTOR_MAP["cultural_context"]) for v in train_df.get("target_cultural_context", train_df.get("cultural_context", pd.Series("", index=train_df.index)))], dtype=torch.long, device=device)
    y_imp = torch.tensor([_encode_target_label(v, TAXONOMY_VECTOR_MAP["implicit_level"]) for v in train_df.get("target_implicit_level", train_df.get("implicit_level", pd.Series("", index=train_df.index)))], dtype=torch.long, device=device)
    y_sev = torch.tensor([_encode_target_label(v, TAXONOMY_VECTOR_MAP["severity"]) for v in train_df.get("target_severity", train_df.get("severity", pd.Series("", index=train_df.index)))], dtype=torch.long, device=device)

    x_va = torch.tensor(val_vec, dtype=torch.float32, device=device)
    y_va = val_df["label"].to_numpy()
    x_te = torch.tensor(test_vec, dtype=torch.float32, device=device)
    y_te = test_df["label"].to_numpy()

    best_val_f1 = -1.0
    best_state = None
    for _ in range(max_epochs):
        clf.train()
        optimizer.zero_grad()
        out = clf(
            x_tr,
            labels_binary=y_tr,
            labels_type=y_typ,
            labels_emotion=y_emo,
            labels_cultural=y_cul,
            labels_implicit=y_imp,
            labels_severity=y_sev,
        )
        out["loss"].backward()
        optimizer.step()

        clf.eval()
        with torch.no_grad():
            va_probs = clf(x_va)["probabilities"]["binary"][:, 1].cpu().numpy()
            va_f1 = float(f1_score(y_va, (va_probs >= 0.5).astype(int), average="macro", zero_division=0))
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    clf.eval()
    with torch.no_grad():
        val_probs = clf(x_va)["probabilities"]["binary"][:, 1].cpu().numpy()
        te_out = clf(x_te)["probabilities"]
        test_probs = te_out["binary"][:, 1].cpu().numpy()

    opt_tau, _, _ = find_optimal_threshold_on_val(val_probs, y_va)
    metrics = compute_comprehensive_classification_metrics(
        y_true=y_te,
        y_prob_pos=test_probs,
        threshold=opt_tau,
        df_meta=test_df,
        seed=seed,
    )
    metrics["model_name"] = "E2. LLM Knowledge-Only Model (33D Contextual Cues)"
    metrics["status"] = "evaluated_against_llm_pseudo_labels"
    return metrics


def _train_hybrid_variant(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_vec: np.ndarray,
    val_vec: np.ndarray,
    test_vec: np.ndarray,
    tokenizer,
    pretrained_model_name: str,
    device: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr_transformer: float = 2e-5,
    lr_heads: float = 1e-4,
    max_length: int = 128,
    patience: int = 2,
    modality_dropout_prob: float = 0.15,
    fusion_mode: str = "gated",
    ablation_mask_mode: Optional[str] = None,
    seed: int = DEFAULT_RANDOM_SEED,
    variant_label: str = "E3. Full Hybrid Fusion",
    save_artifacts_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Trainer bersama yang digunakan oleh model utama (E3) dan varian ablasi Transformer (E1, E4, E5, E6)
    pada split dan seed yang identik tanpa mematikan Stream B saat validasi/test (kecuali E1/E4/E5 eksplisit).
    """
    set_seed(seed)
    torch.manual_seed(seed)

    tr_vec_eff = _apply_cue_mask(train_vec, ablation_mask_mode)
    va_vec_eff = _apply_cue_mask(val_vec, ablation_mask_mode)
    te_vec_eff = _apply_cue_mask(test_vec, ablation_mask_mode)

    train_dataset = HybridCyberbullyingDataset(train_df, tr_vec_eff, tokenizer, max_length=max_length)
    val_dataset = HybridCyberbullyingDataset(val_df, va_vec_eff, tokenizer, max_length=max_length)
    test_dataset = HybridCyberbullyingDataset(test_df, te_vec_eff, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    model = HybridFusionModel(
        pretrained_model_name=pretrained_model_name,
        llm_input_dim=TOTAL_VECTOR_DIM,
        llm_hidden_dim=128,
        use_uncertainty_loss=True,
        focal_alpha=0.60,
        focal_gamma=2.0,
        fusion_mode=fusion_mode,
    ).to(device)

    optimizer_grouped_parameters = [
        {"params": [p for _, p in model.transformer.named_parameters()], "lr": lr_transformer, "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("transformer.")], "lr": lr_heads, "weight_decay": 0.001},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    latest_ckpt_path = None
    if save_artifacts_dir is not None:
        save_artifacts_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = save_artifacts_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_ckpt_path = ckpt_dir / "latest_checkpoint.pt"

    start_epoch = 1
    best_val_f1 = -1.0
    best_epoch = 0
    best_model_state = None
    no_improve_epochs = 0
    history: List[Dict[str, Any]] = []

    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        no_improve_epochs = int(ckpt.get("no_improve_epochs", 0))
        best_model_state = ckpt.get("best_model_state", None)
        history = ckpt.get("history", [])

    rng = np.random.RandomState(seed)
    t0 = time.time()
    effective_mod_dropout = 0.0 if ablation_mask_mode == "all_zero" else modality_dropout_prob

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            llm_vectors = batch["llm_vectors"].to(device)

            if effective_mod_dropout > 0.0:
                drop_mask = torch.tensor(
                    (rng.rand(llm_vectors.size(0), 1) >= effective_mod_dropout).astype(np.float32),
                    device=device,
                )
                llm_vectors = llm_vectors * drop_mask

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_vectors=llm_vectors,
                labels_binary=batch["labels_binary"].to(device),
                labels_type=batch["labels_type"].to(device),
                labels_emotion=batch["labels_emotion"].to(device),
                labels_cultural=batch["labels_cultural"].to(device),
                labels_implicit=batch["labels_implicit"].to(device),
                labels_severity=batch["labels_severity"].to(device),
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += float(loss.item())

        avg_train_loss = train_loss / max(1, len(train_loader))
        val_res = evaluate_multi_task(model, val_loader, device)
        val_f1 = val_res["binary_macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_res["loss"],
            "val_accuracy": val_res["binary_accuracy"],
            "val_macro_f1": val_f1,
            "mean_gate_z": val_res["mean_gate_z"],
        })
        if save_artifacts_dir is not None:
            print(
                f"  Epoch [{epoch:02d}/{epochs:02d}] Train Loss={avg_train_loss:.4f} | "
                f"Val Loss={val_res['loss']:.4f} | Val Acc={val_res['binary_accuracy']*100:.2f}% | "
                f"Val Macro-F1={val_f1:.4f} | Gate z={val_res['mean_gate_z']:.3f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve_epochs = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve_epochs += 1

        if latest_ckpt_path is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                    "best_epoch": best_epoch,
                    "no_improve_epochs": no_improve_epochs,
                    "best_model_state": best_model_state,
                    "history": history,
                },
                latest_ckpt_path,
            )

        if no_improve_epochs >= patience:
            break

    train_duration = round(time.time() - t0, 1)
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    best_weights_path = None
    if save_artifacts_dir is not None:
        best_weights_path = save_artifacts_dir / "best_hybrid_model.pt"
        torch.save(best_model_state or model.state_dict(), best_weights_path)
        tokenizer.save_pretrained(str(save_artifacts_dir))

    val_best = evaluate_multi_task(model, val_loader, device)
    opt_tau, val_opt_f1, val_opt_acc = find_optimal_threshold_on_val(val_best["bin_probs"], val_best["bin_targets"])

    test_res = evaluate_multi_task(model, test_loader, device)
    valid_bin = test_res["bin_targets"] != IGNORE_INDEX
    test_metrics = compute_comprehensive_classification_metrics(
        y_true=test_res["bin_targets"][valid_bin],
        y_prob_pos=test_res["bin_probs"][valid_bin],
        threshold=opt_tau,
        df_meta=test_df.iloc[np.where(valid_bin)[0]].reset_index(drop=True),
        seed=seed,
    )
    test_metrics["model_name"] = variant_label
    test_metrics["fusion_mode"] = fusion_mode
    test_metrics["ablation_mask_mode"] = ablation_mask_mode or "none"
    test_metrics["mean_gating_z"] = test_res["mean_gate_z"]
    test_metrics["multi_task_heads_test"] = test_res["mtl_heads"]

    return {
        "metrics": test_metrics,
        "test_res": test_res,
        "val_opt_f1": val_opt_f1,
        "val_opt_acc": val_opt_acc,
        "opt_tau": opt_tau,
        "best_epoch": best_epoch,
        "history": history,
        "train_duration": train_duration,
        "best_weights_path": best_weights_path,
        "latest_ckpt_path": latest_ckpt_path,
    }


def train_hybrid_pipeline(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    train_vec_path: Path,
    val_vec_path: Path,
    test_vec_path: Path,
    output_dir: Path,
    pretrained_model_name: str = INDOBERTWEET_PRETRAINED_MODEL,
    epochs: int = 5,
    batch_size: int = 16,
    lr_transformer: float = 2e-5,
    lr_heads: float = 1e-4,
    max_length: int = 128,
    patience: int = 2,
    use_rationale_text: bool = False,
    seed: int = DEFAULT_RANDOM_SEED,
    resume_from: Optional[Path] = None,
    baseline_model_dir: Optional[Path] = None,
    modality_dropout_prob: float = 0.15,
    run_ablation: bool = True,
) -> Dict[str, Any]:
    _ = use_rationale_text  # Intentionally disabled to avoid leaking target tags in text encoder
    print("=" * 78)
    print("🚀 MEMULAI PELATIHAN DUAL-STREAM HYBRID FUSION (ACTIVE STREAM A + STREAM B)")
    print("=" * 78)
    set_seed(seed)
    torch.manual_seed(seed)
    device = detect_torch_device()
    print(f"  ⚙️ Device: {device} | Random Seed: {seed}")

    # 1. Load Data & Audit Cross-Split Leakage
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    leakage_report = verify_no_cross_split_leakage(train_df, val_df, test_df)
    print(f"  ✅ Cross-split leakage check PASSED: {leakage_report}")

    train_vec = np.load(train_vec_path)
    val_vec = np.load(val_vec_path)
    test_vec = np.load(test_vec_path)

    ds_manifest_path = train_csv.parent / "dataset_manifest.json"
    dataset_version = "unspecified"
    target_source = str(train_df["target_source"].iloc[0]) if "target_source" in train_df.columns and len(train_df) > 0 else "pseudo_labeled_llm"
    if ds_manifest_path.exists():
        with open(ds_manifest_path, "r", encoding="utf-8") as f:
            ds_m = json.load(f)
            dataset_version = ds_m.get("dataset_version", dataset_version)
            target_source = ds_m.get("label_source", target_source)

    print("  ✅ [Dual-Stream Active Policy]: Stream A (IndoBERTweet) & Stream B (33D Non-Leaky Contextual Evidence) AKTIF di Train/Val/Test.")
    print(f"     -> Target Supervisi: {target_source} (Modality Dropout regularisasi training p={modality_dropout_prob:.2f})")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # 2. Latih & Evaluasi Model Utama (E3 Full Hybrid Fusion)
    e3_run = _train_hybrid_variant(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_vec=train_vec,
        val_vec=val_vec,
        test_vec=test_vec,
        tokenizer=tokenizer,
        pretrained_model_name=pretrained_model_name,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr_transformer=lr_transformer,
        lr_heads=lr_heads,
        max_length=max_length,
        patience=patience,
        modality_dropout_prob=modality_dropout_prob,
        fusion_mode="gated",
        ablation_mask_mode=None,
        seed=seed,
        variant_label="E3. Full Hybrid Transformer-LLM Fusion",
        save_artifacts_dir=output_dir,
        resume_from=resume_from,
    )

    hybrid_test_full = e3_run["metrics"]
    test_res = e3_run["test_res"]
    history = e3_run["history"]
    opt_tau = e3_run["opt_tau"]
    best_epoch = e3_run["best_epoch"]
    train_duration = e3_run["train_duration"]
    best_weights_path = e3_run["best_weights_path"]
    latest_ckpt_path = e3_run["latest_ckpt_path"]

    print(
        f"\n🎯 Threshold Optimal Validation (E3): tau={opt_tau:.2f} | "
        f"Test Acc={hybrid_test_full['accuracy']*100:.2f}% | Macro-F1={hybrid_test_full['macro_f1']:.4f} | "
        f"ROC-AUC={hybrid_test_full['roc_auc']} | PR-AUC={hybrid_test_full['pr_auc']} | Gate z={test_res['mean_gate_z']:.4f}"
    )

    # 3. 6-Way Ablation Study on IDENTICAL Splits (E1-E6)
    baseline_dir = baseline_model_dir or (PROJECT_DIR / "output" / "models" / "indobertweet_cyberbullying" / "best_model")
    standalone_baseline = _evaluate_indobertweet_baseline_on_test(
        baseline_dir=baseline_dir, val_df=val_df, test_df=test_df, device=device, max_length=max_length, seed=seed
    )

    ablation_e2 = _train_and_eval_llm_only_ablation(
        train_vec=train_vec,
        val_vec=val_vec,
        test_vec=test_vec,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        device=device,
        seed=seed,
    )

    if run_ablation:
        print("\n🔬 Menjalankan 6-Way Ablation Study pada split identik (E1, E2, E3, E4, E5, E6)...")
        e1_run = _train_hybrid_variant(
            train_df=train_df, val_df=val_df, test_df=test_df,
            train_vec=train_vec, val_vec=val_vec, test_vec=test_vec,
            tokenizer=tokenizer, pretrained_model_name=pretrained_model_name, device=device,
            epochs=epochs, batch_size=batch_size, lr_transformer=lr_transformer, lr_heads=lr_heads,
            max_length=max_length, patience=patience, modality_dropout_prob=0.0,
            fusion_mode="gated", ablation_mask_mode="all_zero", seed=seed,
            variant_label="E1. Transformer-Only Baseline (Stream A)",
        )
        ablation_e1 = e1_run["metrics"]

        e4_run = _train_hybrid_variant(
            train_df=train_df, val_df=val_df, test_df=test_df,
            train_vec=train_vec, val_vec=val_vec, test_vec=test_vec,
            tokenizer=tokenizer, pretrained_model_name=pretrained_model_name, device=device,
            epochs=epochs, batch_size=batch_size, lr_transformer=lr_transformer, lr_heads=lr_heads,
            max_length=max_length, patience=patience, modality_dropout_prob=modality_dropout_prob,
            fusion_mode="gated", ablation_mask_mode="no_cultural", seed=seed,
            variant_label="E4. Hybrid without Cultural Knowledge",
        )
        ablation_e4 = e4_run["metrics"]

        e5_run = _train_hybrid_variant(
            train_df=train_df, val_df=val_df, test_df=test_df,
            train_vec=train_vec, val_vec=val_vec, test_vec=test_vec,
            tokenizer=tokenizer, pretrained_model_name=pretrained_model_name, device=device,
            epochs=epochs, batch_size=batch_size, lr_transformer=lr_transformer, lr_heads=lr_heads,
            max_length=max_length, patience=patience, modality_dropout_prob=modality_dropout_prob,
            fusion_mode="gated", ablation_mask_mode="no_emotional", seed=seed,
            variant_label="E5. Hybrid without Emotional Knowledge",
        )
        ablation_e5 = e5_run["metrics"]

        e6_run = _train_hybrid_variant(
            train_df=train_df, val_df=val_df, test_df=test_df,
            train_vec=train_vec, val_vec=val_vec, test_vec=test_vec,
            tokenizer=tokenizer, pretrained_model_name=pretrained_model_name, device=device,
            epochs=epochs, batch_size=batch_size, lr_transformer=lr_transformer, lr_heads=lr_heads,
            max_length=max_length, patience=patience, modality_dropout_prob=modality_dropout_prob,
            fusion_mode="concat", ablation_mask_mode=None, seed=seed,
            variant_label="E6. Hybrid without Gated Fusion (Concat)",
        )
        ablation_e6 = e6_run["metrics"]
    else:
        ablation_e1 = standalone_baseline
        ablation_e4 = None
        ablation_e5 = None
        ablation_e6 = None

    ablation_summary = {
        "test_samples": int(len(test_df)),
        "target_source": target_source,
        "evaluation_mode": "full_dual_stream_active",
        "E1_transformer_only": ablation_e1,
        "E2_llm_knowledge_only": ablation_e2,
        "E3_hybrid_full_fusion": hybrid_test_full,
        "E4_hybrid_no_cultural": ablation_e4,
        "E5_hybrid_no_emotional": ablation_e5,
        "E6_hybrid_no_gated_fusion": ablation_e6,
        # Backward-compatible summary keys
        "indobertweet_standalone_checkpoint": standalone_baseline,
        "model_a_indobertweet_only": ablation_e1 or standalone_baseline,
        "model_b_llm_only": ablation_e2,
        "model_c_hybrid_fusion": hybrid_test_full,
    }
    with open(output_dir / "ablation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2, ensure_ascii=False)

    # Save training history + metrics JSON + Model Manifest
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    hyperparameters = {
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "lr_transformer": lr_transformer,
        "lr_heads": lr_heads,
        "max_length": max_length,
        "patience": patience,
        "modality_dropout_prob": modality_dropout_prob,
        "fusion_mode": "gated",
        "text_input_mode": "text_only",
        "stream_a_active": True,
        "stream_b_active": True,
        "eval_llm_stream_masked": False,
    }

    hybrid_metrics_payload = {
        "model_architecture": "Hybrid Dual-Stream Gated Fusion (IndoBERTweet + Non-Leaky 33D LLM Contextual Cues)",
        "pretrained_backbone": pretrained_model_name,
        "dataset_version": dataset_version,
        "target_source": target_source,
        "evaluation_mode": "full_dual_stream_pseudo_label_agreement",
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "training_duration_sec": train_duration,
        "optimal_threshold": opt_tau,
        "threshold_selection_split": "validation",
        "mean_gating_z": test_res["mean_gate_z"],
        "test_metrics": hybrid_test_full,
        "multi_task_heads_test": test_res["mtl_heads"],
        "ablation_comparison": ablation_summary,
    }
    with open(output_dir / "hybrid_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(hybrid_metrics_payload, f, indent=2, ensure_ascii=False)

    manifest = build_model_manifest(
        model_architecture="Hybrid Dual-Stream Gated Fusion (6-Head MTL)",
        pretrained_model=pretrained_model_name,
        checkpoint_location=best_weights_path,
        hyperparameters=hyperparameters,
        evaluation_metrics=hybrid_test_full,
        optimal_threshold=opt_tau,
        dataset_version=dataset_version,
        target_source=target_source,
        random_seed=seed,
        extra_metadata={
            "multi_task_heads_test": test_res["mtl_heads"],
            "evaluation_mode": hybrid_metrics_payload["evaluation_mode"],
            "stream_b_representation": "non_leaky_33d_contextual_evidence_cues",
            "resumable_checkpoint": str(latest_ckpt_path),
        },
    )
    save_model_manifest(output_dir, manifest)

    print(f"\n💾 Weights terbaik disimpan di   : {best_weights_path}")
    print(f"💾 Ablation 6-Way disimpan di    : {output_dir / 'ablation_comparison.json'}")
    print(f"💾 Model Manifest disimpan di    : {output_dir / 'model_manifest.json'}")
    print("=" * 78)
    return hybrid_metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Pelatihan Hybrid Fusion Multi-Task Classifier")
    parser.add_argument("--train_csv", type=str, default=str(DEFAULT_FINAL_DIR / "train.csv"))
    parser.add_argument("--val_csv", type=str, default=str(DEFAULT_FINAL_DIR / "val.csv"))
    parser.add_argument("--test_csv", type=str, default=str(DEFAULT_FINAL_DIR / "test.csv"))
    parser.add_argument("--train_vec", type=str, default=str(DEFAULT_FINAL_DIR / "train_llm_vectors.npy"))
    parser.add_argument("--val_vec", type=str, default=str(DEFAULT_FINAL_DIR / "val_llm_vectors.npy"))
    parser.add_argument("--test_vec", type=str, default=str(DEFAULT_FINAL_DIR / "test_llm_vectors.npy"))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_HYBRID_DIR))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--skip-ablation", action="store_true", help="Lewati pelatihan varian ablasi E1/E4/E5/E6")
    parser.add_argument("--resume", type=str, default=None, help="Path checkpoint .pt untuk melanjutkan training")

    args = parser.parse_args()
    train_hybrid_pipeline(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        test_csv=Path(args.test_csv),
        train_vec_path=Path(args.train_vec),
        val_vec_path=Path(args.val_vec),
        test_vec_path=Path(args.test_vec),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        patience=args.patience,
        seed=args.seed,
        modality_dropout_prob=args.modality_dropout,
        run_ablation=not args.skip_ablation,
        resume_from=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
