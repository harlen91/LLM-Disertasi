#!/usr/bin/env python3
"""
curate_dataset.py
-----------------
Skrip kurasi dan normalisasi dataset hasil anotasi LLM SumoPod (LLaMA-3)
untuk penyiapan dataset latih model Transformer Encoder IndoBERTweet.

Tugas:
1. Menyaring dan membuang baris data yang mengalami error JSON (UNKNOWN, No JSON brackets found, Malformed JSON).
2. Melakukan normalisasi string (penghapusan spasi ganda, trailing whitespace, kontrol karakter, encoding).
3. Menstandarkan seluruh kolom taksonomi kategori (cyberbullying, type, emotion, cultural_context, implicit_level, severity).
4. Menstandarkan confidence score menjadi float numerik [0.0, 1.0].
5. Menghasilkan binerisasi label (0 = Non-Cyberbullying, 1 = Cyberbullying).
6. Melakukan Stratified Train/Val/Test Split (80% / 10% / 10%).
7. Menyimpan dataset bersih ke output/final/.
"""

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Setup Project Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
from core.schema import parse_confidence_score, sanitize_annotation_dict
from core.data_understanding import (
    clean_and_normalize_social_text,
    format_structured_rationale,
    extract_taxonomy_probability_vector,
    TOTAL_VECTOR_DIM
)


def clean_text(text: str) -> str:
    """
    Membersihkan teks komentar media sosial tanpa menghilangkan bahasa gaul / slang:
    1. Menghilangkan tautan/URL web.
    2. Menghapus awalan username seperti '@human:' atau '@user:'.
    3. Menstandarkan tag mention media sosial menjadi '@user'.
    4. Mereduksi elongasi huruf berulang (> 2 kali, misal 'Narsissssss' -> 'Narsiss').
    5. Menghilangkan kontrol karakter dan whitespace berlebih.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Hapus URL
    cleaned = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 2. Hapus awalan username percakapan seperti '@human:' atau '@User Name:'
    cleaned = re.sub(r'^@[\w\s\.\-]{1,35}:\s*', '', cleaned)
    # 3. Standardisasi mention tersisa menjadi @user
    cleaned = re.sub(r'@\w+', '@user', cleaned)
    # 4. Reduksi elongasi huruf berulang (> 2 menjadi 2 huruf, misal 'bangeeeet' -> 'bangeet')
    cleaned = re.sub(r'([a-zA-Z0-9])\1{2,}', r'\1\1', cleaned)
    # 5. Ganti newline, carriage return, dan tab dengan spasi
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
    # 6. Ganti multiple whitespace berurutan menjadi satu spasi
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # 7. Hapus whitespace di awal dan akhir
    return cleaned.strip()


def curate_and_normalize(
    input_csv: Path,
    output_csv: Path,
    perform_split: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Pipeline kurasi data end-to-end dengan pelaporan statistik akademik."""
    print("=" * 70)
    print("🚀 MEMULAI PIPELINE KURASI & NORMALISASI DATASET IICD (INDOBERTWEET)")
    print("=" * 70)
    print(f"📂 Berkas Input  : {input_csv}")
    print(f"🎯 Berkas Output : {output_csv}\n")

    if not input_csv.exists():
        raise FileNotFoundError(f"Berkas input tidak ditemukan: {input_csv}")

    df_raw = pd.read_csv(input_csv)
    total_raw = len(df_raw)
    print(f"📊 Total baris mentah termuat: {total_raw:,} baris")

    # -------------------------------------------------------------
    # 1. FILTERING: HAPUS BARIS ERROR & ANOMALI JSON
    # -------------------------------------------------------------
    print("\n--- [Langkah 1: Penyaringan Anomali & Error JSON] ---")

    # Identifikasi baris dengan error kolom
    has_error_col = "error" in df_raw.columns
    if has_error_col:
        error_mask = (
            df_raw["error"].notna() &
            (df_raw["error"].astype(str).str.strip() != "") &
            (df_raw["error"].astype(str).str.lower() != "nan") &
            (df_raw["error"].astype(str).str.lower() != "none")
        )
        error_count = error_mask.sum()
    else:
        error_mask = pd.Series(False, index=df_raw.index)
        error_count = 0

    # Identifikasi baris dengan label cyberbullying yang rusak / UNKNOWN / ERROR
    cb_col = df_raw["cyberbullying"].astype(str).str.strip().str.upper()
    invalid_cb_mask = ~cb_col.isin(["YES", "NO"])
    invalid_cb_count = invalid_cb_mask.sum()

    # Identifikasi baris dengan teks penjelasan mengindikasikan kegagalan JSON
    exp_col = df_raw["explanation"].astype(str) if "explanation" in df_raw.columns else pd.Series("", index=df_raw.index)
    broken_exp_mask = exp_col.str.contains(r"No JSON brackets found|Failed to locate JSON|JSON syntax error", case=False, na=False)
    broken_exp_count = broken_exp_mask.sum()

    # Identifikasi komentar kosong atau terlalu pendek (< 3 karakter)
    comment_col = df_raw["comment"].fillna("").astype(str).apply(clean_text)
    invalid_comment_mask = comment_col.str.len() < 3
    invalid_comment_count = invalid_comment_mask.sum()

    # Gabungkan semua filter eliminasi
    drop_mask = error_mask | invalid_cb_mask | broken_exp_mask | invalid_comment_mask
    dropped_count = drop_mask.sum()

    print(f"  ❌ Baris dengan error JSON eksplisit : {error_count:,}")
    print(f"  ❌ Baris dengan label UNKNOWN/ERROR  : {invalid_cb_count:,}")
    print(f"  ❌ Baris terpotong (syntax/brackets) : {broken_exp_count:,}")
    print(f"  ❌ Baris teks kosong / < 3 karakter  : {invalid_comment_count:,}")
    print(f"  🗑️ Total baris rusak yang dieliminasi: {dropped_count:,} baris ({dropped_count/total_raw*100:.2f}%)")

    df_filtered = df_raw[~drop_mask].copy().reset_index(drop=True)
    print(f"  ✅ Baris lolos tahap eliminasi       : {len(df_filtered):,} baris")

    # -------------------------------------------------------------
    # 2. NORMALISASI STRING & STANDARISASI TAKSONOMI
    # -------------------------------------------------------------
    print("\n--- [Langkah 2: Normalisasi String, Rationale CoT, & Vektor Taksonomi (33-Dim)] ---")

    cleaned_records = []
    vector_list = []
    for idx, row in df_filtered.iterrows():
        raw_comment = clean_text(str(row["comment"]))
        norm_comment = clean_and_normalize_social_text(
            raw_comment,
            handle_slang=True,
            handle_emojis=True,
            reduce_elongation=True
        )

        # Buat dictionary untuk disanitasi oleh core/schema.py
        row_dict = {
            "cyberbullying": str(row.get("cyberbullying", "")),
            "type": str(row.get("type", "")),
            "emotion": str(row.get("emotion", "")),
            "cultural_context": str(row.get("cultural_context", "")),
            "implicit_level": str(row.get("implicit_level", "")),
            "severity": str(row.get("severity", "")),
            "confidence": row.get("confidence", 0.85),
            "explanation": clean_text(str(row.get("explanation", "")))
        }

        # Jalankan sanitasi ketat taksonomi
        sanitized = sanitize_annotation_dict(row_dict)

        # Format Structured CoT Rationale & Ekstraksi Vektor 33-Dimensi
        structured_rat = format_structured_rationale(sanitized)
        vec_33 = extract_taxonomy_probability_vector(sanitized)
        vector_list.append(vec_33)

        cleaned_records.append({
            "comment": raw_comment,
            "comment_normalized": norm_comment,
            "cyberbullying": sanitized["cyberbullying"],
            "label": 1 if sanitized["cyberbullying"] == "YES" else 0,
            "type": sanitized["type"],
            "emotion": sanitized["emotion"],
            "cultural_context": sanitized["cultural_context"],
            "implicit_level": sanitized["implicit_level"],
            "severity": sanitized["severity"],
            "confidence": sanitized["confidence"],
            "explanation": sanitized["explanation"],
            "structured_rationale": structured_rat
        })

    df_cleaned = pd.DataFrame(cleaned_records)
    all_vectors = np.array(vector_list, dtype=np.float32)

    # -------------------------------------------------------------
    # 3. DEDUPLIKASI TEKS (PREVENT DATA LEAKAGE & SINKRONISASI VEKTOR)
    # -------------------------------------------------------------
    print("\n--- [Langkah 3: Deduplikasi Komentar & Sinkronisasi Vektor] ---")
    before_dedup = len(df_cleaned)
    df_cleaned["_row_idx"] = np.arange(len(df_cleaned))
    df_cleaned = df_cleaned.drop_duplicates(subset=["comment"]).reset_index(drop=True)
    kept_indices = df_cleaned["_row_idx"].values
    all_vectors = all_vectors[kept_indices]
    df_cleaned = df_cleaned.drop(columns=["_row_idx"])
    dedup_dropped = before_dedup - len(df_cleaned)
    print(f"  🔍 Duplikasi teks komentar dibuang: {dedup_dropped:,} baris")
    print(f"  ✨ Total dataset bersih akhir       : {len(df_cleaned):,} baris")
    print(f"  📐 Bentuk matriks vektor LLM       : {all_vectors.shape} (Dimensi={TOTAL_VECTOR_DIM})")

    # -------------------------------------------------------------
    # 4. LAPORAN DISTRIBUSI DATASET BERSIH
    # -------------------------------------------------------------
    print("\n--- [Langkah 4: Statistik Distribusi Dataset Bersih] ---")
    cb_counts = df_cleaned["cyberbullying"].value_counts()
    cb_pcts = df_cleaned["cyberbullying"].value_counts(normalize=True) * 100
    for label in ["NO", "YES"]:
        cnt = cb_counts.get(label, 0)
        pct = cb_pcts.get(label, 0.0)
        tag = "Non-Cyberbullying (0)" if label == "NO" else "Cyberbullying (1)"
        print(f"  • {tag:22}: {cnt:,} sampel ({pct:.2f}%)")

    avg_conf = df_cleaned["confidence"].mean()
    min_conf = df_cleaned["confidence"].min()
    max_conf = df_cleaned["confidence"].max()
    print(f"\n  📈 Confidence Score: Rata-rata={avg_conf:.2f} | Min={min_conf:.2f} | Max={max_conf:.2f}")

    print("\n  Top 5 Cyberbullying Types (YES):")
    yes_types = df_cleaned[df_cleaned["label"] == 1]["type"].value_counts().head(5)
    for t_code, t_cnt in yes_types.items():
        print(f"    - {t_code}: {t_cnt} baris")

    # -------------------------------------------------------------
    # 5. EKSPOR DATASET BERSIH UTAMA & VEKTOR NUMPY
    # -------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_csv, index=False, encoding="utf-8")
    all_vec_path = output_csv.parent / f"{output_csv.stem}_vectors.npy"
    np.save(all_vec_path, all_vectors)
    print(f"\n💾 Dataset bersih tersimpan di        : {output_csv}")
    print(f"💾 Vektor probabilitas LLM (.npy)    : {all_vec_path}")

    # -------------------------------------------------------------
    # 6. STRATIFIED TRAIN / VAL / TEST SPLIT & VEKTOR PARTISI
    # -------------------------------------------------------------
    if perform_split:
        print("\n--- [Langkah 5: Stratified Train / Val / Test Split (80/10/10) & Partisi Vektor] ---")
        try:
            from sklearn.model_selection import train_test_split

            indices = np.arange(len(df_cleaned))
            labels = df_cleaned["label"].values

            # Split 80% train, 20% temp (val + test)
            train_idx, temp_idx = train_test_split(
                indices,
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
                stratify=labels
            )

            # Split temp into 50% val, 50% test (each 10% of total)
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.5,
                random_state=random_state,
                stratify=labels[temp_idx]
            )

            train_df = df_cleaned.iloc[train_idx].reset_index(drop=True)
            val_df = df_cleaned.iloc[val_idx].reset_index(drop=True)
            test_df = df_cleaned.iloc[test_idx].reset_index(drop=True)

            train_path = output_csv.parent / "train.csv"
            val_path = output_csv.parent / "val.csv"
            test_path = output_csv.parent / "test.csv"

            train_df.to_csv(train_path, index=False, encoding="utf-8")
            val_df.to_csv(val_path, index=False, encoding="utf-8")
            test_df.to_csv(test_path, index=False, encoding="utf-8")

            # Simpan partisi vektor LLM
            np.save(output_csv.parent / "train_llm_vectors.npy", all_vectors[train_idx])
            np.save(output_csv.parent / "val_llm_vectors.npy", all_vectors[val_idx])
            np.save(output_csv.parent / "test_llm_vectors.npy", all_vectors[test_idx])

            print(f"  📁 Train set ({len(train_df):,} baris, {len(train_df)/len(df_cleaned)*100:.1f}%) -> {train_path.name} & train_llm_vectors.npy")
            print(f"  📁 Val set   ({len(val_df):,} baris, {len(val_df)/len(df_cleaned)*100:.1f}%) -> {val_path.name} & val_llm_vectors.npy")
            print(f"  📁 Test set  ({len(test_df):,} baris, {len(test_df)/len(df_cleaned)*100:.1f}%) -> {test_path.name} & test_llm_vectors.npy")
            print("  ✅ Partisi stratified dan matriks vektor berhasil disinkronkan!")
        except ImportError:
            print("  ⚠️ Scikit-learn tidak terpasang, melewati pembuatan split otomatis.")

    print("\n" + "=" * 70)
    print("🎉 LANGKAH 1 SELESAI: DATASET SIAP DIGUNAKAN UNTUK TRAINING INDOBERTWEET!")
    print("=" * 70)

    return df_cleaned


def main():
    parser = argparse.ArgumentParser(description="Kurasi dataset LLM untuk IndoBERTweet")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_DIR / "output" / "checkpoints" / "checkpoint_Dataset-Research.csv"),
        help="Path ke berkas checkpoint input"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_DIR / "output" / "final" / "iicd_cleaned_indobertweet.csv"),
        help="Path tujuan berkas dataset bersih"
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Jangan buat train/val/test split otomatis"
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    curate_and_normalize(
        input_csv=input_path,
        output_csv=output_path,
        perform_split=not args.no_split
    )


if __name__ == "__main__":
    main()
