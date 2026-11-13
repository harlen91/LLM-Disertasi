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
from core.prompt_templates import PROMPT_HASH
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
    CONTEXTUAL_EVIDENCE_SCHEMA,
    TOTAL_VECTOR_DIM,
    clean_and_normalize_social_text,
    extract_contextual_evidence_vector,
    get_teacher_feature_names,
    get_teacher_feature_schema,
    format_structured_rationale,
)
from data_manager.dataset_handler import detect_ground_truth_column, detect_text_column

CANDIDATE_GROUP_COLS = [
    "group_id", "conversation_id", "thread_id", "post_id", "user_id", "author_id", "username"
]


def _detect_group_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    """Detects optional user/thread/conversation identifier column for group-aware splitting."""
    if preferred and preferred in df.columns:
        non_empty = df[preferred].dropna().astype(str).str.strip()
        if (non_empty != "").sum() > 0:
            return preferred
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in CANDIDATE_GROUP_COLS:
        if cand in lower_map:
            col = lower_map[cand]
            if df[col].dropna().astype(str).str.strip().ne("").sum() > 0:
                return col
    return None


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
    group_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Memastikan tidak ada komentar yang sama (raw/normalized) atau group ID di antara train, val, dan test."""
    def _norm_set(frame: pd.DataFrame, col: str) -> set:
        if col not in frame.columns:
            return set()
        return set(frame[col].fillna("").astype(str).str.strip().str.lower()) - {""}

    results = {}
    check_cols = [c for c in ["comment", text_col] if c in train_df.columns]
    if group_col and group_col in train_df.columns:
        check_cols.append(group_col)

    for col in check_cols:
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


def _build_composite_stratify_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Membangun label stratifikasi komposit berbasis pseudo-label (`label` + `target_implicit_level`)
    bila setiap strata memiliki minimal >= 4 sampel; jika tidak, fallback ke `label` biner.
    """
    if "label" in df.columns:
        base_series = df["label"]
    elif "cyberbullying" in df.columns:
        base_series = (df["cyberbullying"].astype(str).str.upper() == "YES").astype(int)
    else:
        return np.zeros(len(df), dtype=int)

    imp_col = "target_implicit_level" if "target_implicit_level" in df.columns else ("implicit_level" if "implicit_level" in df.columns else None)
    if imp_col:
        comp = base_series.astype(str) + "__" + df[imp_col].astype(str)
        counts = comp.value_counts()
        if (counts >= 4).all() and len(counts) > 1:
            return comp.values
    return base_series.values


def _split_train_val_test_indices(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
    group_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Membagi dataset secara konsisten:
    - Group-aware stratified split apabila identitas pengguna/percakapan (`group_col`) tersedia.
    - Stratified split berbasis pseudo-label jika tidak ada `group_col`.
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    strat_labels = _build_composite_stratify_labels(df)

    if group_col and group_col in df.columns:
        groups = df[group_col].fillna("").astype(str).str.strip()
        if (groups != "").all() and groups.nunique() >= 5:
            rng = np.random.RandomState(random_state)
            group_stats = (
                pd.DataFrame({"group": groups, "label": df["label"].values, "idx": indices})
                .groupby("group", sort=False)
                .agg(size=("idx", "count"), pos_rate=("label", "mean"))
                .reset_index()
            )
            group_stats["_rand"] = rng.rand(len(group_stats))
            group_stats = group_stats.sort_values(by=["pos_rate", "size", "_rand"]).reset_index(drop=True)

            target_train = (1.0 - val_ratio - test_ratio) * len(df)
            target_val = val_ratio * len(df)
            target_test = test_ratio * len(df)

            buckets: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
            bucket_sizes = {"train": 0, "val": 0, "test": 0}
            targets = {"train": target_train, "val": target_val, "test": target_test}

            for _, grow in group_stats.iterrows():
                gname = str(grow["group"])
                gsize = int(grow["size"])
                deficits = {b: (targets[b] - bucket_sizes[b]) / max(1.0, targets[b]) for b in ("train", "val", "test")}
                chosen = max(deficits.keys(), key=lambda b: (deficits[b], b == "train"))
                buckets[chosen].append(gname)
                bucket_sizes[chosen] += gsize

            g2b = {g: b for b, gs in buckets.items() for g in gs}
            split_assign = groups.map(g2b)
            train_idx = indices[(split_assign == "train").values]
            val_idx = indices[(split_assign == "val").values]
            test_idx = indices[(split_assign == "test").values]
            if len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0:
                return train_idx, val_idx, test_idx, f"group_aware_stratified({group_col})"

    # Safely check whether initial stratify labels have >= 2 samples per class
    init_strat = strat_labels
    if init_strat is not None:
        init_counts = pd.Series(init_strat).value_counts()
        if (init_counts < 2).any() or len(init_counts) < 2:
            # Fall back to binary label if possible
            if "label" in df.columns:
                bin_counts = df["label"].value_counts()
                init_strat = df["label"].values if (bin_counts >= 2).all() and len(bin_counts) > 1 else None
            else:
                init_strat = None

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=init_strat,
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    temp_strat = strat_labels[temp_idx]
    val_counts = pd.Series(temp_strat).value_counts()
    if (val_counts < 2).any() and "label" in df.columns:
        temp_strat = df["label"].values[temp_idx]

    strat_arg = temp_strat
    if strat_arg is not None:
        c_counts = pd.Series(strat_arg).value_counts()
        if (c_counts < 2).any() or len(c_counts) < 2:
            strat_arg = None

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=strat_arg,
    )
    strat_kind = "stratified_composite_pseudo_label" if (init_strat is not None and len(np.unique(init_strat)) > 2) else ("stratified_binary_pseudo_label" if init_strat is not None else "random_split_small_dataset")
    return train_idx, val_idx, test_idx, strat_kind


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
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pipeline kurasi data berbasis kebijakan LLM Pseudo-Labels:
    - Berkas mentah tidak diubah.
    - Pseudo-label LLM (`target_*`) dipisahkan tegas dari representasi bukti kontekstual 33D (Stream B).
    - Seluruh baris yang gagal/invalid dicatat lengkap alasannya di `failed_annotations_log.json`.
    - Pembagian train/val/test menggunakan stratified / group-aware split dengan verifikasi 0 overlap.
    """
    print("=" * 72)
    print("🚀 MEMULAI PIPELINE KURASI & VALIDASI DATASET IICD (PSEUDO-LABEL POLICY)")
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
    detected_group_col = _detect_group_column(df_raw, preferred=group_col)

    raw_audit = audit_dataset_quality(df_raw, text_col=text_col, human_gt_col=gt_col)
    print(f"📊 Total baris mentah termuat : {raw_audit['total_rows']:,}")
    print(
        f"🏷️ Kebijakan Supervisi       : {raw_audit['label_source']}"
        + (f" (kolom referensi opsional '{gt_col}')" if has_human_gt else " (Supervisi utama menggunakan LLM-generated pseudo-labels)")
    )
    if detected_group_col:
        print(f"👥 Group-Aware Split Aktif    : kolom `{detected_group_col}` terdeteksi")
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

    # Catat seluruh baris gagal anotasi/validasi beserta alasannya
    failed_records: List[Dict[str, Any]] = []
    for raw_idx, is_dropped in drop_mask.items():
        if not bool(is_dropped):
            continue
        reasons = []
        if bool(error_mask.loc[raw_idx]):
            reasons.append(f"error_col:{str(df_raw.loc[raw_idx, 'error'])[:120]}")
        if bool(mock_mask.loc[raw_idx]):
            reasons.append("mock_simulation")
        if bool(invalid_cb_mask.loc[raw_idx]):
            reasons.append(f"invalid_cyberbullying:{cb_col.loc[raw_idx]}")
        if bool(broken_exp_mask.loc[raw_idx]):
            reasons.append("json_parse_error_in_explanation")
        if bool(invalid_comment_mask.loc[raw_idx]):
            reasons.append("empty_or_short_comment")
        failed_records.append(
            {
                "raw_row_index": int(raw_idx),
                "comment_preview": str(df_raw.loc[raw_idx, text_col])[:100],
                "reasons": reasons,
                "llm_model": str(df_raw.loc[raw_idx].get("llm_model", "")).strip(),
                "prompt_version": str(df_raw.loc[raw_idx].get("prompt_version", "")).strip(),
            }
        )

    df_filtered = df_raw[~drop_mask].copy().reset_index(drop=True)

    print(
        f"  🗑️ Eliminasi baris rusak/invalid : {int(drop_mask.sum()):,} "
        f"(error={int(error_mask.sum())}, mock={int(mock_mask.sum())}, "
        f"cb_invalid={int(invalid_cb_mask.sum())}, comment_invalid={int(invalid_comment_mask.sum())})"
    )
    print(f"  ✅ Lolos filter awal             : {len(df_filtered):,} baris")

    # 2. Normalisasi, Pemisahan Pseudo-Labels vs Bukti Kontekstual 33D Non-Target
    cleaned_records: List[Dict[str, Any]] = []
    vector_list: List[np.ndarray] = []

    for f_idx, row in df_filtered.iterrows():
        raw_comment = clean_text(str(row[text_col]))
        norm_comment = clean_and_normalize_social_text(
            raw_comment,
            handle_slang=True,
            handle_emojis=True,
            reduce_elongation=True,
        )
        if len(norm_comment.strip()) < 2:
            failed_records.append(
                {
                    "raw_row_index": int(f_idx),
                    "comment_preview": raw_comment[:100],
                    "reasons": ["empty_after_normalization"],
                    "llm_model": str(row.get("llm_model", "")).strip(),
                    "prompt_version": str(row.get("prompt_version", "")).strip(),
                }
            )
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
            "contextual_cues": row.get("contextual_cues", None),
        }
        sanitized = sanitize_annotation_dict(row_dict, preserve_missing=True)

        # ANTI-LEAKAGE: Stream B HANYA menerima bukti kontekstual/rasional & confidence (tanpa label target)
        evidence_input = {
            "comment": raw_comment,
            "comment_normalized": norm_comment,
            "explanation": sanitized["explanation"],
            "confidence": sanitized["confidence"],
            "contextual_cues": sanitized.get("contextual_cues"),
        }
        vec_33 = extract_contextual_evidence_vector(evidence_input, text=raw_comment)
        vector_list.append(vec_33)

        # Tentukan Target Labels (LLM pseudo-labels secara default; atau kolom referensi opsional jika eksplisit diberikan)
        if has_human_gt:
            gt_bin = normalize_binary_target(row.get(gt_col))
            if gt_bin is None:
                continue
            target_cb = "YES" if gt_bin == 1 else "NO"
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
        structured_rat = format_structured_rationale(llm_record)

        record_out = {
            "comment": raw_comment,
            "comment_normalized": norm_comment,
            # Explicit separation: pseudo-label targets vs LLM knowledge evidence
            "target_source": label_source,
            "target_cyberbullying": target_cb,
            "label": int(gt_bin),
            "target_type": target_type if target_type in VALID_TYPES else MISSING_LABEL,
            "target_emotion": target_emotion if target_emotion in VALID_EMOTIONS else MISSING_LABEL,
            "target_cultural_context": target_culture if target_culture in VALID_CULTURAL_CONTEXTS else MISSING_LABEL,
            "target_implicit_level": target_implicit if target_implicit in VALID_IMPLICIT_LEVELS else MISSING_LABEL,
            "target_severity": target_severity if target_severity in VALID_SEVERITIES else MISSING_LABEL,
            # LLM teacher outputs & contextual metadata
            "llm_cyberbullying": sanitized["cyberbullying"],
            "llm_type": sanitized["type"],
            "llm_emotion": sanitized["emotion"],
            "llm_cultural_context": sanitized["cultural_context"],
            "llm_implicit_level": sanitized["implicit_level"],
            "llm_severity": sanitized["severity"],
            "llm_confidence": sanitized["confidence"] if sanitized["confidence"] is not None else np.nan,
            "llm_explanation": sanitized["explanation"],
            # Backward-compatible columns used by downstream analytics & UI
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
            "annotated_at": str(row.get("annotated_at", "")).strip() or "",
            "annotation_source": str(row.get("annotation_source", "")).strip() or "sumopod_api",
        }
        if detected_group_col:
            record_out[detected_group_col] = row.get(detected_group_col, "")
        cleaned_records.append(record_out)

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

    failed_log_path = output_csv.parent / "failed_annotations_log.json"
    with open(failed_log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_failed_rows": len(failed_records),
                "input_source": str(input_csv),
                "records": failed_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    split_info: Dict[str, Any] = {}
    if perform_split:
        train_idx, val_idx, test_idx, split_strategy = _split_train_val_test_indices(
            df_cleaned,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            group_col=detected_group_col,
        )

        train_df = df_cleaned.iloc[train_idx].reset_index(drop=True)
        val_df = df_cleaned.iloc[val_idx].reset_index(drop=True)
        test_df = df_cleaned.iloc[test_idx].reset_index(drop=True)

        leakage_report = verify_no_cross_split_leakage(
            train_df,
            val_df,
            test_df,
            group_col=detected_group_col,
        )
        print(f"  ✅ Strategi Split ({split_strategy}) & Zero Leakage OK: {leakage_report}")

        train_df.to_csv(output_csv.parent / "train.csv", index=False, encoding="utf-8")
        val_df.to_csv(output_csv.parent / "val.csv", index=False, encoding="utf-8")
        test_df.to_csv(output_csv.parent / "test.csv", index=False, encoding="utf-8")

        np.save(output_csv.parent / "train_llm_vectors.npy", all_vectors[train_idx])
        np.save(output_csv.parent / "val_llm_vectors.npy", all_vectors[val_idx])
        np.save(output_csv.parent / "test_llm_vectors.npy", all_vectors[test_idx])

        split_info = {
            "random_state": random_state,
            "strategy": split_strategy,
            "group_col": detected_group_col,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "cross_split_leakage_check": leakage_report,
        }

    # Simpan dataset manifest untuk reproducibility & model manifest linkage
    csv_sha = hashlib.sha256(output_csv.read_bytes()).hexdigest()[:12]
    llm_models_used = sorted(df_cleaned["llm_model"].dropna().astype(str).unique().tolist())
    prompt_versions_used = sorted(df_cleaned["prompt_version"].dropna().astype(str).unique().tolist())
    prompt_hashes_used = (
        sorted(df_cleaned["prompt_hash"].dropna().astype(str).unique().tolist())
        if "prompt_hash" in df_cleaned.columns
        else [PROMPT_HASH]
    )
    dataset_manifest = {
        "dataset_version": f"iicd-curated-{csv_sha}",
        "curated_at": datetime.now(timezone.utc).isoformat(),
        "input_source": str(input_csv),
        "output_dataset": str(output_csv),
        "supervision_policy": "llm_generated_pseudo_labels",
        "has_human_ground_truth": has_human_gt,
        "label_source": "human_ground_truth" if has_human_gt else "pseudo_labeled_llm",
        "confidence_policy": "raw_llm_confidence_uncalibrated",
        "stream_b_representation": "non_leaky_33d_contextual_evidence_cues",
        "llm_models_used": llm_models_used,
        "prompt_versions_used": prompt_versions_used,
        "prompt_hashes_used": prompt_hashes_used,
        "evaluation_note": (
            "Evaluasi mengukur kesepakatan/konsistensi model terhadap pseudo-label yang dihasilkan LLM "
            "(tanpa ketergantungan pada human ground truth)."
        ),
        "raw_audit": raw_audit,
        "failed_annotations_count": len(failed_records),
        "failed_annotations_log": str(failed_log_path),
        "dedup_dropped": int(dedup_dropped),
        "final_rows": int(len(df_cleaned)),
        "teacher_vector_dim": TOTAL_VECTOR_DIM,
        "contextual_cue_schema": CONTEXTUAL_EVIDENCE_SCHEMA,
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
    parser.add_argument("--ground-truth-col", type=str, default=None, help="Opsional: nama kolom referensi sekunder terpisah")
    parser.add_argument("--group-col", type=str, default=None, help="Opsional: nama kolom group/user/thread untuk group-aware split")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--no-split", action="store_true", help="Jangan buat train/val/test split otomatis")

    args = parser.parse_args()
    curate_and_normalize(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        perform_split=not args.no_split,
        random_state=args.seed,
        human_ground_truth_col=args.ground_truth_col,
        group_col=args.group_col,
    )


if __name__ == "__main__":
    main()
