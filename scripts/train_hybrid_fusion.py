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


def _train_and_eval_llm_only_ablation(
    train_vec: np.ndarray,
    val_vec: np.ndarray,
    test_vec: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    has_human_ground_truth: bool,
    device: str,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Model Ablasi B: LLM Representation Only (33-dim vector MLP).
    - Jika `has_human_ground_truth=True`, melatih dan mengevaluasi secara sah terhadap ground truth manusia.
    - Jika `has_human_ground_truth=False` (`pseudo_labeled_llm`), menolak evaluasi circular LLM->LLM.
    """
    if not has_human_ground_truth:
        return {
            "model_name": "B. LLM Representation-Only (Stream B)",
            "status": "blocked_circular_pseudo_label_leakage",
            "note": (
                "Evaluasi Stream B standalone diblokir karena dataset berstatus pseudo_labeled_llm "
                "(vektor 33D berasal dari prediksi LLM yang identik dengan target pseudo-label). "
                "Tambahkan kolom ground truth manusia tervalidasi untuk evaluasi Model B secara independen."
            ),
            "accuracy": None,
            "macro_f1": None,
        }

    torch.manual_seed(seed)
    clf = LLMKnowledgeOnlyClassifier(in_features=TOTAL_VECTOR_DIM).to(device)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-3)

    x_tr = torch.tensor(train_vec, dtype=torch.float32, device=device)
    y_tr = torch.tensor(train_df["label"].to_numpy(), dtype=torch.long, device=device)
    x_va = torch.tensor(val_vec, dtype=torch.float32, device=device)
    y_va = val_df["label"].to_numpy()
    x_te = torch.tensor(test_vec, dtype=torch.float32, device=device)
    y_te = test_df["label"].to_numpy()

    for _ in range(35):
        clf.train()
        optimizer.zero_grad()
        out = clf(x_tr, labels_binary=y_tr)
        out["loss"].backward()
        optimizer.step()

    clf.eval()
    with torch.no_grad():
        val_probs = clf(x_va)["probabilities"]["binary"][:, 1].cpu().numpy()
        test_probs = clf(x_te)["probabilities"]["binary"][:, 1].cpu().numpy()

    opt_tau, _, _ = find_optimal_threshold_on_val(val_probs, y_va)
    metrics = compute_comprehensive_classification_metrics(
        y_true=y_te,
        y_prob_pos=test_probs,
        threshold=opt_tau,
        df_meta=test_df,
        seed=seed,
    )
    metrics["model_name"] = "B. LLM Representation-Only (Stream B)"
    metrics["status"] = "evaluated_against_human_ground_truth"
    return metrics


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
) -> Dict[str, Any]:
    print("=" * 78)
    print("🚀 MEMULAI PELATIHAN DUAL-STREAM HYBRID FUSION (LEAKAGE-FREE & MTL 6-HEADS)")
    print("=" * 78)
    set_seed(seed)
    torch.manual_seed(seed)
    device = detect_torch_device()
    print(f"  ⚙️ Device: {device} | Random Seed: {seed}")

    # 1. Load Data & Audit Leakage
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
    has_human_gt = False
    target_source = str(train_df["target_source"].iloc[0]) if "target_source" in train_df.columns and len(train_df) > 0 else "pseudo_labeled_llm"
    if ds_manifest_path.exists():
        with open(ds_manifest_path, "r", encoding="utf-8") as f:
            ds_m = json.load(f)
            dataset_version = ds_m.get("dataset_version", dataset_version)
            has_human_gt = bool(ds_m.get("has_human_ground_truth", False))
            target_source = ds_m.get("label_source", target_source)
    else:
        has_human_gt = target_source == "human_ground_truth"

    # Leakage Guard Decision
    mask_eval_llm_stream = not has_human_gt
    modality_dropout_prob = 0.50 if not has_human_gt else 0.15
    if mask_eval_llm_stream:
        print("  🛡️ [Leakage Guard Aktif]: Dataset berstatus `pseudo_labeled_llm` (belum ada ground truth manusia).")
        print("     -> Stream B dimask ke vektor nol saat validasi/test evaluation (Modality-Masked Student Mode)")
        print("     -> Modality Dropout (p=0.50) aktif saat training untuk mencegah shortcut leakage & melatih 6 MTL heads.")
    else:
        print("  ✅ [Ground Truth Independen Terdeteksi]: Stream B (33D LLM Teacher Vector) aktif penuh di train/val/test.")

    if use_rationale_text and not has_human_gt:
        print("  ⚠️ `use_rationale_text` dinonaktifkan otomatis dalam mode pseudo_labeled_llm untuk mencegah kebocoran tag target di teks.")
        use_rationale_text = False

    # 2. Tokenizer, Dataset & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    train_dataset = HybridCyberbullyingDataset(
        train_df, train_vec, tokenizer, max_length=max_length, use_rationale_text=use_rationale_text, mask_llm_vectors=False
    )
    val_dataset = HybridCyberbullyingDataset(
        val_df, val_vec, tokenizer, max_length=max_length, use_rationale_text=use_rationale_text, mask_llm_vectors=mask_eval_llm_stream
    )
    test_dataset = HybridCyberbullyingDataset(
        test_df, test_vec, tokenizer, max_length=max_length, use_rationale_text=use_rationale_text, mask_llm_vectors=mask_eval_llm_stream
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    # 3. Model, Optimizer, Scheduler
    model = HybridFusionModel(
        pretrained_model_name=pretrained_model_name,
        llm_input_dim=TOTAL_VECTOR_DIM,
        llm_hidden_dim=128,
        use_uncertainty_loss=True,
        focal_alpha=0.60,
        focal_gamma=2.0,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pt"

    start_epoch = 1
    best_val_f1 = -1.0
    best_epoch = 0
    best_model_state = None
    no_improve_epochs = 0
    history: List[Dict[str, Any]] = []

    # Optional checkpoint resume
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
        print(f"  🔁 Melanjutkan pelatihan dari checkpoint epoch {start_epoch - 1} ({resume_from})")

    # 4. Training Loop dengan Early Stopping
    rng = np.random.RandomState(seed)
    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            llm_vectors = batch["llm_vectors"].to(device)

            # Modality dropout during training
            if modality_dropout_prob > 0.0:
                drop_mask = torch.tensor(
                    (rng.rand(llm_vectors.size(0), 1) >= modality_dropout_prob).astype(np.float32),
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

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_res["loss"],
            "val_accuracy": val_res["binary_accuracy"],
            "val_macro_f1": val_f1,
            "mean_gate_z": val_res["mean_gate_z"],
        }
        history.append(epoch_record)
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

        # Save resumable training checkpoint
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
            print(f"  ⏹️ Early stopping dipicu pada epoch {epoch} (tidak ada peningkatan Val Macro-F1 selama {patience} epoch).")
            break

    train_duration = round(time.time() - start_time, 1)
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    best_weights_path = output_dir / "best_hybrid_model.pt"
    torch.save(best_model_state or model.state_dict(), best_weights_path)
    tokenizer.save_pretrained(str(output_dir))

    # 5. Threshold Optimization STRICTLY on Validation Set
    val_best = evaluate_multi_task(model, val_loader, device)
    opt_tau, val_opt_f1, val_opt_acc = find_optimal_threshold_on_val(val_best["bin_probs"], val_best["bin_targets"])
    print(f"\n🎯 Threshold Optimal dari Validation Set: tau={opt_tau:.2f} (Val Macro-F1={val_opt_f1:.4f}, Val Acc={val_opt_acc*100:.2f}%)")

    # 6. Evaluasi Final pada Untouched Test Set
    test_res = evaluate_multi_task(model, test_loader, device)
    valid_bin = test_res["bin_targets"] != IGNORE_INDEX
    hybrid_test_full = compute_comprehensive_classification_metrics(
        y_true=test_res["bin_targets"][valid_bin],
        y_prob_pos=test_res["bin_probs"][valid_bin],
        threshold=opt_tau,
        df_meta=test_df.iloc[np.where(valid_bin)[0]].reset_index(drop=True),
        seed=seed,
    )

    print(
        f"📊 [Test Set Hybrid Fusion] Acc={hybrid_test_full['accuracy']*100:.2f}% | "
        f"Macro-F1={hybrid_test_full['macro_f1']:.4f} | ROC-AUC={hybrid_test_full['roc_auc']} | "
        f"PR-AUC={hybrid_test_full['pr_auc']} | ECE={hybrid_test_full['calibration']['ece']}"
    )

    # 7. 3-Way Ablation Comparison on IDENTICAL Test Set
    baseline_dir = baseline_model_dir or (PROJECT_DIR / "output" / "models" / "indobertweet_cyberbullying" / "best_model")
    ablation_a = _evaluate_indobertweet_baseline_on_test(
        baseline_dir=baseline_dir, val_df=val_df, test_df=test_df, device=device, max_length=max_length, seed=seed
    )
    ablation_b = _train_and_eval_llm_only_ablation(
        train_vec=train_vec,
        val_vec=val_vec,
        test_vec=test_vec,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        has_human_ground_truth=has_human_gt,
        device=device,
        seed=seed,
    )
    ablation_c = {**hybrid_test_full, "model_name": "C. Hybrid Dual-Stream Fusion (IndoBERTweet + LLM Knowledge)"}

    ablation_summary = {
        "test_samples": int(len(test_df)),
        "target_source": target_source,
        "evaluation_mode": "modality_masked_student" if mask_eval_llm_stream else "full_dual_stream",
        "model_a_indobertweet_only": ablation_a,
        "model_b_llm_only": ablation_b,
        "model_c_hybrid_fusion": ablation_c,
    }
    with open(output_dir / "ablation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2, ensure_ascii=False)

    # Save metrics JSON + training history + Model Manifest
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
        "text_input_mode": "text_plus_rationale" if use_rationale_text else "text_only",
        "eval_llm_stream_masked": mask_eval_llm_stream,
    }

    hybrid_metrics_payload = {
        "model_architecture": "Hybrid Dual-Stream Gated Fusion (IndoBERTweet + SumoPod LLM 33D Teacher Features)",
        "pretrained_backbone": pretrained_model_name,
        "dataset_version": dataset_version,
        "target_source": target_source,
        "evaluation_mode": "leakage_free_modality_masked_student" if mask_eval_llm_stream else "full_dual_stream_human_gt",
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
            "resumable_checkpoint": str(latest_ckpt_path),
        },
    )
    save_model_manifest(output_dir, manifest)

    print(f"\n💾 Weights terbaik disimpan di   : {best_weights_path}")
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
        resume_from=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
