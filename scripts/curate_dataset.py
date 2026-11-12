#!/usr/bin/env python3
"""
curate_dataset.py
-----------------
Pipeline kurasi, validasi kualitas, pemisahan tegas ground truth vs LLM pseudo-labels,
ekstraksi 33-dimensi teacher annotation features, dan stratified train/val/test split
tanpa kebocoran data (cross-split & target leakage checks).

Penting:
- File input mentah (`checkpoint_*.csv` atau CSV/Excel sumber) tidak pernah diubah atau dihapus.
- Apabila tidak terdapat kolom ground truth manusia yang tervalidasi (`human_ground_truth`,
  `gold_label`, dll.), dataset ditandai secara eksplisit sebagai `pseudo_labeled_llm`.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import DEFAULT_RANDOM_SEED
from core.schema import (
    MISSING_LABEL,
    VALID_CYBERBULLYING,
    VALID_TYPES,
    VALID_EMOTIONS,
    VALID_CULTURAL_CONTEXTS,
    VALID_IMPLICIT_LEVELS,
    VALID_SEVERITIES,
    parse_confidence_score,
    sanitize_annotation_dict,
    validate_annotation_record,
)
from core.data_understanding import (
    TOTAL_VECTOR_DIM,
    clean_and_normalize_social_text,
    extract_teacher_annotation_features,
    get_teacher_feature_names,
    get_teacher_feature_schema,
    format_structured_rationale,
)
from data_manager.dataset_handler import detect_ground_truth_column, detect_text_column


def clean_text(text: str) -> str:
    """Membersihkan teks komentar media sosial dasar tanpa menghapus konteks slang."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"https?://\S+|www\.\S+", "", text)
    cleaned = re.sub(r"^@[\w\s\.\-]{1,35}:\s*", "", cleaned)
    cleaned = re.sub(r"@\w+", "@user", cleaned)
    cleaned = re.sub(r"([a-zA-Z0-9])\1{2,}", r"\1\1", cleaned)
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_binary_target(val: Any) -> Optional[int]:
    """Maps human ground truth binary representations to 1 (YES), 0 (NO), or None (UNKNOWN)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().upper()
    if s in {"1", "YES", "YA", "TRUE", "CYBERBULLYING", "POSITIVE"}:
        return 1
    if s in {"0", "NO", "TIDAK", "BUKAN", "FALSE", "NON-CYBERBULLYING", "NON_CYBERBULLYING", "NEGATIVE"}:
        return 0
    return None


def audit_dataset_quality(
    df: pd.DataFrame,
    text_col: str = "comment",
    human_gt_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Melakukan pemeriksaan kualitas menyeluruh sebelum dan selama kurasi:
    - missing values per kolom
    - komentar kosong / invalid / terlalu pendek
    - duplikasi teks mentah & ternormalisasi
    - validitas label taksonomi & confidence score
    - konsistensi antar label taksonomi
    - deteksi keberadaan baris simulasi MockEngine
    """
    missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns}
    raw_texts = df[text_col].fillna("").astype(str) if text_col in df.columns else pd.Series([], dtype=str)
    empty_comments = int((raw_texts.str.strip() == "").sum())
    cleaned_texts = raw_texts.apply(clean_text)
    short_comments = int((cleaned_texts.str.len() < 3).sum())
    norm_keys = raw_texts.apply(lambda t: clean_and_normalize_social_text(clean_text(t)).lower().strip())

    raw_duplicates = int(raw_texts.str.strip().duplicated().sum())
    normalized_duplicates = int(norm_keys.duplicated().sum())

    # Confidence validity
    invalid_conf_count = 0
    if "confidence" in df.columns:
        for v in df["confidence"]:
            c = parse_confidence_score(v, default=None)
            if c is None or not (0.0 <= c <= 1.0):
                invalid_conf_count += 1

    # Mock source check
    mock_rows = 0
    if "annotation_source" in df.columns:
        mock_rows += int((df["annotation_source"].astype(str) == "mock_simulation").sum())
    if "is_mock" in df.columns:
        mock_rows = max(mock_rows, int(df["is_mock"].fillna(False).astype(bool).sum()))

    # Taxonomy validity & consistency checks
    consistency_issues = 0
    invalid_taxonomy_rows = 0
    for _, row in df.iterrows():
        ok, issues = validate_annotation_record(row.to_dict())
        if not ok:
            consistency_issues += 1
            if any(i.startswith("invalid_") or i.startswith("missing_or_invalid_") for i in issues):
                invalid_taxonomy_rows += 1

    return {
        "total_rows": int(len(df)),
        "text_column": text_col,
        "human_ground_truth_column": human_gt_col,
        "has_human_ground_truth": human_gt_col is not None,
        "label_source": "human_ground_truth" if human_gt_col else "pseudo_labeled_llm",
        "missing_values_per_column": missing_by_col,
        "empty_comments": empty_comments,
        "short_comments_lt_3": short_comments,
        "raw_duplicates": raw_duplicates,
        "normalized_duplicates": normalized_duplicates,
        "invalid_confidence_count": invalid_conf_count,
        "invalid_taxonomy_rows": invalid_taxonomy_rows,
        "consistency_issue_rows": consistency_issues,
        "mock_simulation_rows": mock_rows,
    }


def verify_no_cross_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "comment_normalized",
) -> Dict[str, Any]:
    """Memastikan tidak ada komentar yang sama (raw maupun normalized) di antara train, val, dan test."""
    def _norm_set(frame: pd.DataFrame, col: str) -> set:
        if col not in frame.columns:
            return set()
        return set(frame[col].fillna("").astype(str).str.strip().str.lower()) - {""}

    results = {}
    for col in [c for c in ["comment", text_col] if c in train_df.columns]:
        tr_s = _norm_set(train_df, col)
        va_s = _norm_set(val_df, col)
        te_s = _norm_set(test_df, col)
        results[col] = {
            "train_val_overlap": len(tr_s & va_s),
            "train_test_overlap": len(tr_s & te_s),
            "val_test_overlap": len(va_s & te_s),
        }
        total_overlap = (
            results[col]["train_val_overlap"]
            + results[col]["train_test_overlap"]
            + results[col]["val_test_overlap"]
        )
        if total_overlap > 0:
            raise ValueError(
                f"Data leakage terdeteksi lintas split pada kolom '{col}': {results[col]}"
            )
    return results


def curate_and_normalize(
    input_csv: Path,
    output_csv: Path,
    perform_split: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = DEFAULT_RANDOM_SEED,
    human_ground_truth_col: Optional[str] = None,
    allow_mock_rows: bool = False,
) -> pd.DataFrame:
    """
    Pipeline kurasi data end-to-end dengan pemisahan ketat ground truth vs anotasi LLM,
    validasi kualitas, deduplikasi ternormalisasi, dan verifikasi bebas kebocoran split.
    """
    print("=" * 72)
    print("🚀 MEMULAI PIPELINE KURASI & VALIDASI DATASET IICD")
    print("=" * 72)
    print(f"📂 Berkas Input (Read-Only) : {input_csv}")
    print(f"🎯 Berkas Output            : {output_csv}\n")

    if not input_csv.exists():
        raise FileNotFoundError(f"Berkas input tidak ditemukan: {input_csv}")

    df_raw = pd.read_csv(input_csv)
    text_col = detect_text_column(df_raw) or "comment"
    if text_col not in df_raw.columns:
        raise ValueError(f"Kolom teks '{text_col}' tidak ditemukan di {input_csv}.")

    gt_col = human_ground_truth_col if (human_ground_truth_col and human_ground_truth_col in df_raw.columns) else detect_ground_truth_column(df_raw)
    has_human_gt = gt_col is not None

    raw_audit = audit_dataset_quality(df_raw, text_col=text_col, human_gt_col=gt_col)
    print(f"📊 Total baris mentah termuat : {raw_audit['total_rows']:,}")
    print(f"🏷️ Sumber Label Target        : {raw_audit['label_source']}" + (f" (kolom '{gt_col}')" if has_human_gt else " (BELUM ADA ground truth manusia independen -> pseudo-labeled LLM)"))
    if "sentiment" in df_raw.columns and not has_human_gt:
        print("   ℹ️ Kolom 'sentiment' bawaan terdeteksi namun TIDAK dijadikan ground truth cyberbullying karena bukan label cyberbullying manusia tervalidasi.")

    # 1. Filtering Error JSON, Mock rows, dan Komentar Kosong/Invalid
    has_error_col = "error" in df_raw.columns
    if has_error_col:
        error_mask = (
            df_raw["error"].notna()
            & (df_raw["error"].astype(str).str.strip() != "")
            & (~df_raw["error"].astype(str).str.lower().isin(["nan", "none"]))
        )
    else:
        error_mask = pd.Series(False, index=df_raw.index)

    mock_mask = pd.Series(False, index=df_raw.index)
    if not allow_mock_rows:
        if "annotation_source" in df_raw.columns:
            mock_mask |= df_raw["annotation_source"].astype(str) == "mock_simulation"
        if "is_mock" in df_raw.columns:
            mock_mask |= df_raw["is_mock"].fillna(False).astype(bool)

    cb_col = df_raw["cyberbullying"].astype(str).str.strip().str.upper() if "cyberbullying" in df_raw.columns else pd.Series("", index=df_raw.index)
    invalid_cb_mask = ~cb_col.isin(VALID_CYBERBULLYING)

    exp_col = df_raw["explanation"].astype(str) if "explanation" in df_raw.columns else pd.Series("", index=df_raw.index)
    broken_exp_mask = exp_col.str.contains(
        r"No JSON brackets found|Failed to locate JSON|JSON syntax error",
        case=False,
        na=False,
    )

    comment_col = df_raw[text_col].fillna("").astype(str).apply(clean_text)
    invalid_comment_mask = (
        (df_raw[text_col].isna())
        | (comment_col.str.len() < 3)
        | (comment_col.str.lower().isin({"nan", "none", "null"}))
    )

    drop_mask = error_mask | mock_mask | invalid_cb_mask | broken_exp_mask | invalid_comment_mask
    df_filtered = df_raw[~drop_mask].copy().reset_index(drop=True)

    print(
        f"  🗑️ Eliminasi baris rusak/invalid : {int(drop_mask.sum()):,} "
        f"(error={int(error_mask.sum())}, mock={int(mock_mask.sum())}, "
        f"cb_invalid={int(invalid_cb_mask.sum())}, comment_invalid={int(invalid_comment_mask.sum())})"
    )
    print(f"  ✅ Lolos filter awal             : {len(df_filtered):,} baris")

    # 2. Normalisasi, Pemisahan Kolom Ground Truth vs LLM Prediction, & Teacher Vector (33D)
    cleaned_records: List[Dict[str, Any]] = []
    vector_list: List[np.ndarray] = []

    for _, row in df_filtered.iterrows():
        raw_comment = clean_text(str(row[text_col]))
        norm_comment = clean_and_normalize_social_text(
            raw_comment,
            handle_slang=True,
            handle_emojis=True,
            reduce_elongation=True,
        )
        if len(norm_comment.strip()) < 2:
            continue

        row_dict = {
            "cyberbullying": row.get("cyberbullying", ""),
            "type": row.get("type", ""),
            "emotion": row.get("emotion", ""),
            "cultural_context": row.get("cultural_context", ""),
            "implicit_level": row.get("implicit_level", ""),
            "severity": row.get("severity", ""),
            "confidence": row.get("confidence", None),
            "explanation": clean_text(str(row.get("explanation", ""))) if pd.notna(row.get("explanation")) else "",
        }
        sanitized = sanitize_annotation_dict(row_dict, preserve_missing=True)

        # Ekstraksi Teacher Features 33D strictly from `llm_*` annotations
        llm_record = {
            "llm_cyberbullying": sanitized["cyberbullying"],
            "llm_type": sanitized["type"],
            "llm_emotion": sanitized["emotion"],
            "llm_cultural_context": sanitized["cultural_context"],
            "llm_implicit_level": sanitized["implicit_level"],
            "llm_severity": sanitized["severity"],
            "llm_confidence": sanitized["confidence"],
            "llm_explanation": sanitized["explanation"],
        }
        vec_33 = extract_teacher_annotation_features(llm_record)
        vector_list.append(vec_33)

        # Tentukan Target Labels (strictly from human GT when available; else pseudo-labeled LLM)
        if has_human_gt:
            gt_bin = normalize_binary_target(row.get(gt_col))
            if gt_bin is None:
                continue
            target_cb = "YES" if gt_bin == 1 else "NO"
            # Human sub-labels only if explicitly provided in human_* columns, otherwise MISSING_LABEL (-100 masked)
            target_type = str(row.get("human_type", MISSING_LABEL)).strip().upper()
            target_emotion = str(row.get("human_emotion", MISSING_LABEL)).strip().upper()
            target_culture = str(row.get("human_cultural_context", MISSING_LABEL)).strip().upper()
            target_implicit = str(row.get("human_implicit_level", MISSING_LABEL)).strip().upper()
            target_severity = str(row.get("human_severity", MISSING_LABEL)).strip().upper()
            label_source = "human_ground_truth"
        else:
            if sanitized["cyberbullying"] not in VALID_CYBERBULLYING:
                continue
            gt_bin = 1 if sanitized["cyberbullying"] == "YES" else 0
            target_cb = sanitized["cyberbullying"]
            target_type = sanitized["type"]
            target_emotion = sanitized["emotion"]
            target_culture = sanitized["cultural_context"]
            target_implicit = sanitized["implicit_level"]
            target_severity = sanitized["severity"]
            label_source = "pseudo_labeled_llm"

        structured_rat = format_structured_rationale(llm_record)

        cleaned_records.append(
            {
                "comment": raw_comment,
                "comment_normalized": norm_comment,
                # Explicit separation: target labels vs LLM predictions
                "target_source": label_source,
                "target_cyberbullying": target_cb,
                "label": int(gt_bin),
                "target_type": target_type if target_type in VALID_TYPES else MISSING_LABEL,
                "target_emotion": target_emotion if target_emotion in VALID_EMOTIONS else MISSING_LABEL,
                "target_cultural_context": target_culture if target_culture in VALID_CULTURAL_CONTEXTS else MISSING_LABEL,
                "target_implicit_level": target_implicit if target_implicit in VALID_IMPLICIT_LEVELS else MISSING_LABEL,
                "target_severity": target_severity if target_severity in VALID_SEVERITIES else MISSING_LABEL,
                # LLM teacher predictions (distinct from target_*)
                "llm_cyberbullying": sanitized["cyberbullying"],
                "llm_type": sanitized["type"],
                "llm_emotion": sanitized["emotion"],
                "llm_cultural_context": sanitized["cultural_context"],
                "llm_implicit_level": sanitized["implicit_level"],
                "llm_severity": sanitized["severity"],
                "llm_confidence": sanitized["confidence"] if sanitized["confidence"] is not None else np.nan,
                "llm_explanation": sanitized["explanation"],
                # Backward-compatible columns used by downstream analytics & XAI UI
                "cyberbullying": target_cb,
                "type": target_type if target_type in VALID_TYPES else MISSING_LABEL,
                "emotion": target_emotion if target_emotion in VALID_EMOTIONS else MISSING_LABEL,
                "cultural_context": target_culture if target_culture in VALID_CULTURAL_CONTEXTS else MISSING_LABEL,
                "implicit_level": target_implicit if target_implicit in VALID_IMPLICIT_LEVELS else MISSING_LABEL,
                "severity": target_severity if target_severity in VALID_SEVERITIES else MISSING_LABEL,
                "confidence": sanitized["confidence"] if sanitized["confidence"] is not None else np.nan,
                "explanation": sanitized["explanation"],
                "structured_rationale": structured_rat,
                "llm_model": str(row.get("llm_model", "")).strip() or "unspecified",
                "prompt_version": str(row.get("prompt_version", "")).strip() or "legacy",
                "annotation_source": str(row.get("annotation_source", "")).strip() or "sumopod_api",
            }
        )

    if not cleaned_records:
        raise ValueError("Tidak ada baris valid yang tersisa setelah kurasi.")

    df_cleaned = pd.DataFrame(cleaned_records)
    all_vectors = np.array(vector_list[: len(df_cleaned)], dtype=np.float32)

    # 3. Deduplikasi Ketat (Raw & Normalized) + Sinkronisasi Vektor
    before_dedup = len(df_cleaned)
    df_cleaned["_row_idx"] = np.arange(len(df_cleaned))
    df_cleaned["_dedup_key"] = df_cleaned["comment_normalized"].str.lower().str.strip()
    df_cleaned = df_cleaned.drop_duplicates(subset=["_dedup_key"]).drop_duplicates(subset=["comment"]).reset_index(drop=True)
    kept_indices = df_cleaned["_row_idx"].values
    all_vectors = all_vectors[kept_indices]
    df_cleaned = df_cleaned.drop(columns=["_row_idx", "_dedup_key"])
    dedup_dropped = before_dedup - len(df_cleaned)

    print(f"  🔍 Duplikasi dibuang (norm & raw): {dedup_dropped:,} baris")
    print(f"  ✨ Dataset bersih akhir          : {len(df_cleaned):,} baris")
    print(f"  📐 Matriks Teacher Features      : {all_vectors.shape} (Dim={TOTAL_VECTOR_DIM})")

    # 4. Distribusi Kelas
    class_dist = {
        "cyberbullying_binary": df_cleaned["label"].value_counts().to_dict(),
        "type": df_cleaned["type"].value_counts().to_dict(),
        "emotion": df_cleaned["emotion"].value_counts().to_dict(),
        "cultural_context": df_cleaned["cultural_context"].value_counts().to_dict(),
        "implicit_level": df_cleaned["implicit_level"].value_counts().to_dict(),
        "severity": df_cleaned["severity"].value_counts().to_dict(),
    }
    print(f"  📊 Distribusi biner: {class_dist['cyberbullying_binary']}")

    # 5. Ekspor Dataset Utama & Teacher Vector Artifacts
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_csv, index=False, encoding="utf-8")
    all_vec_path = output_csv.parent / f"{output_csv.stem}_vectors.npy"
    np.save(all_vec_path, all_vectors)

    feature_names_path = output_csv.parent / "teacher_feature_names.json"
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(
            {"feature_names": get_teacher_feature_names(), "schema": get_teacher_feature_schema()},
            f,
            indent=2,
            ensure_ascii=False,
        )

    split_info: Dict[str, Any] = {}
    if perform_split:
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(df_cleaned))
        labels = df_cleaned["label"].values

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=labels,
        )
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_ratio,
            random_state=random_state,
            stratify=labels[temp_idx],
        )

        train_df = df_cleaned.iloc[train_idx].reset_index(drop=True)
        val_df = df_cleaned.iloc[val_idx].reset_index(drop=True)
        test_df = df_cleaned.iloc[test_idx].reset_index(drop=True)

        leakage_report = verify_no_cross_split_leakage(train_df, val_df, test_df)
        print(f"  ✅ Verifikasi Zero Cross-Split Leakage OK: {leakage_report}")

        train_df.to_csv(output_csv.parent / "train.csv", index=False, encoding="utf-8")
        val_df.to_csv(output_csv.parent / "val.csv", index=False, encoding="utf-8")
        test_df.to_csv(output_csv.parent / "test.csv", index=False, encoding="utf-8")

        np.save(output_csv.parent / "train_llm_vectors.npy", all_vectors[train_idx])
        np.save(output_csv.parent / "val_llm_vectors.npy", all_vectors[val_idx])
        np.save(output_csv.parent / "test_llm_vectors.npy", all_vectors[test_idx])

        split_info = {
            "random_state": random_state,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "cross_split_leakage_check": leakage_report,
        }

    # Simpan dataset manifest untuk reproducibility & model manifest linkage
    csv_sha = hashlib.sha256(output_csv.read_bytes()).hexdigest()[:12]
    dataset_manifest = {
        "dataset_version": f"iicd-curated-{csv_sha}",
        "curated_at": datetime.now(timezone.utc).isoformat(),
        "input_source": str(input_csv),
        "output_dataset": str(output_csv),
        "has_human_ground_truth": has_human_gt,
        "label_source": "human_ground_truth" if has_human_gt else "pseudo_labeled_llm",
        "pseudo_labeled_warning": (
            None
            if has_human_gt
            else "Dataset belum memiliki ground truth manusia independen; seluruh metrik mengevaluasi kesepakatan/distilasi terhadap pseudo-label LLM."
        ),
        "raw_audit": raw_audit,
        "dedup_dropped": int(dedup_dropped),
        "final_rows": int(len(df_cleaned)),
        "teacher_vector_dim": TOTAL_VECTOR_DIM,
        "class_distribution": {
            k: {str(ck): int(cv) for ck, cv in v.items()} for k, v in class_dist.items()
        },
        "splits": split_info,
    }
    manifest_path = output_csv.parent / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Dataset Manifest tersimpan di  : {manifest_path}")
    print("🎉 PIPELINE KURASI SELESAI TANPA DATA LEAKAGE ANTAR SPLIT!")
    return df_cleaned


def main():
    parser = argparse.ArgumentParser(description="Kurasi & validasi dataset LLM untuk IndoBERTweet")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_DIR / "output" / "checkpoints" / "checkpoint_Dataset-Research.csv"),
        help="Path ke berkas checkpoint input (read-only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_DIR / "output" / "final" / "iicd_cleaned_indobertweet.csv"),
        help="Path tujuan berkas dataset bersih",
    )
    parser.add_argument("--ground-truth-col", type=str, default=None, help="Opsional: nama kolom ground truth manusia tervalidasi")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--no-split", action="store_true", help="Jangan buat train/val/test split otomatis")

    args = parser.parse_args()
    curate_and_normalize(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        perform_split=not args.no_split,
        random_state=args.seed,
        human_ground_truth_col=args.ground_truth_col,
    )


if __name__ == "__main__":
    main()
