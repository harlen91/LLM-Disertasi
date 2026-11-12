#!/usr/bin/env python3
"""
scripts/evaluate_dual_xai.py
----------------------------
Skrip Evaluasi Empiris Keterjelasan Ganda (Dual Explainability Benchmark):
Menguji keselarasan antara Atribusi Fitur Token Kuantitatif (SHAP)
dengan Penalaran Kualitatif Bahasa Alami (LLaMA-3 Chain-of-Thought Rationale)
di seluruh data uji independen (test.csv, 150 sampel).

Tujuan Penelitian Disertasi (Bab 4):
1. Mengukur tingkat konsistensi empiris (Dual-XAI Alignment Rate) antara SHAP dan CoT.
2. Menganalisis perbedaan keselarasan pada berbagai tingkat keterbukaan ujaran:
   - E0 (Sangat Eksplisit)
   - E1 (Semi-Tersirat)
   - E2 (Tersirat Penuh / Kontekstual Budaya)
3. Mengidentifikasi kata kunci pemicu dengan konsensus tertinggi (High Saliency Consensus).
4. Menghasilkan tabel LaTeX resmi untuk naskah disertasi.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Setup Project Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.shap_explainer import HybridSHAPExplainer, align_shap_with_cot

DEFAULT_TEST_CSV = PROJECT_DIR / "output" / "final" / "test.csv"
DEFAULT_MODEL_PATH = PROJECT_DIR / "output" / "models" / "indobertweet_cyberbullying" / "best_model"
DEFAULT_REPORT_DIR = PROJECT_DIR / "output" / "evaluation_report"


def evaluate_dual_xai_benchmark(
    test_csv_path: Path,
    model_path: Path,
    output_dir: Path,
    max_samples: int = None
) -> Dict[str, Any]:
    print("=" * 75)
    print("🔬 MEMULAI EVALUASI EMPIRIS DUAL EXPLAINABILITY (SHAP ⟷ LLM CoT)")
    print("=" * 75)
    print(f"📂 Berkas Test Set : {test_csv_path}")
    print(f"🧠 Model Backbone  : {model_path}")
    print(f"📊 Direktori Output: {output_dir}\n")

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Berkas test set tidak ditemukan: {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    if max_samples is not None and max_samples < len(df_test):
        df_test = df_test.head(max_samples)
    total_test = len(df_test)
    print(f"📋 Total sampel uji yang dievaluasi: {total_test} baris")

    # Inisialisasi SHAP Explainer
    print("⚡ Menginisialisasi HybridSHAPExplainer...")
    explainer = HybridSHAPExplainer(model_path_or_name=str(model_path))

    results = []
    print("\n⏳ Mengekstrak atribusi SHAP & menghitung keselarasan CoT per sampel...")

    for idx, row in df_test.iterrows():
        comment = str(row.get("comment", "")).strip()
        explanation = str(row.get("explanation", "")).strip()
        cb_label = str(row.get("cyberbullying", "NO")).upper()
        impl_level = str(row.get("implicit_level", "E0")).upper()
        b_type = str(row.get("type", "NONE")).upper()

        # 1. Hitung SHAP Token Attribution
        shap_res = explainer.explain_text(comment, top_k=5)

        # 2. Hitung Penyelarasan Dual-XAI
        align_info = align_shap_with_cot(shap_res, explanation)

        results.append({
            "idx": idx,
            "comment": comment,
            "cyberbullying": cb_label,
            "implicit_level": impl_level,
            "type": b_type,
            "cyberbullying_prob": shap_res["cyberbullying_prob"],
            "top_shap_tokens": [t["token"] for t in shap_res.get("top_positive_tokens", [])],
            "aligned_tokens": align_info["aligned_tokens"],
            "alignment_score": align_info["alignment_score"],
            "is_aligned": align_info["is_aligned"]
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == total_test:
            print(f"  Processed {idx + 1}/{total_test} sampel...")

    df_res = pd.DataFrame(results)

    # =========================================================================
    # ANALISIS STATISTIK AGREGAT
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 HASIL EVALUASI STATISTIK DUAL EXPLAINABILITY")
    print("=" * 70)

    # 1. Metrik Global
    mean_align_global = float(df_res["alignment_score"].mean())
    align_rate_global = float(df_res["is_aligned"].mean() * 100.0)
    print(f"  • Rata-rata Skor Keselarasan Global (Mean Alignment) : {mean_align_global:.2f}%")
    print(f"  • Tingkat Konsistensi Global (Alignment Rate >= 50%)  : {align_rate_global:.2f}%")

    # 2. Breakdown per Tingkat Keterbukaan Ujaran (Implicitness)
    print("\n  📈 Keselarasan per Tingkat Keterbukaan (Implicitness Level):")
    impl_stats = {}
    for level in ["E0", "E1", "E2"]:
        sub = df_res[df_res["implicit_level"] == level]
        if len(sub) > 0:
            sub_mean = float(sub["alignment_score"].mean())
            sub_rate = float(sub["is_aligned"].mean() * 100.0)
            impl_stats[level] = {
                "count": len(sub),
                "mean_alignment": round(sub_mean, 2),
                "alignment_rate": round(sub_rate, 2)
            }
            desc = "E0 (Sangat Eksplisit)" if level == "E0" else ("E1 (Semi-Tersirat)" if level == "E1" else "E2 (Tersirat Kultural)")
            print(f"    - {desc:28}: {len(sub):2d} sampel | Skor: {sub_mean:5.2f}% | Konsisten: {sub_rate:5.2f}%")

    # 3. Breakdown per Cyberbullying (YES vs NO)
    cb_yes = df_res[df_res["cyberbullying"] == "YES"]
    cb_no = df_res[df_res["cyberbullying"] == "NO"]
    cb_stats = {
        "YES": {
            "count": len(cb_yes),
            "mean_alignment": round(float(cb_yes["alignment_score"].mean()), 2) if len(cb_yes) > 0 else 0.0,
            "alignment_rate": round(float(cb_yes["is_aligned"].mean() * 100.0), 2) if len(cb_yes) > 0 else 0.0
        },
        "NO": {
            "count": len(cb_no),
            "mean_alignment": round(float(cb_no["alignment_score"].mean()), 2) if len(cb_no) > 0 else 0.0,
            "alignment_rate": round(float(cb_no["is_aligned"].mean() * 100.0), 2) if len(cb_no) > 0 else 0.0
        }
    }

    # 4. Top Agreed Trigger Words (Kata Kunci Paling Sering Disepakati)
    all_agreed = []
    for al in df_res["aligned_tokens"]:
        all_agreed.extend(al)
    agreed_counts = pd.Series(all_agreed).value_counts().head(10).to_dict()

    print("\n  🔝 Top 5 Kata Pemicu dengan Konsensus Tertinggi (SHAP & CoT):")
    for tok, c in list(agreed_counts.items())[:5]:
        print(f"    - '{tok}': {c} kemunculan")

    # =========================================================================
    # EKSPOR KE JSON & LATEX TABLE UNTUK DISERTASI
    # =========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_report = {
        "evaluation_name": "Empirical Dual Explainability Benchmark (SHAP vs LLaMA-3 CoT)",
        "total_test_samples": total_test,
        "global_metrics": {
            "mean_alignment_score": round(mean_align_global, 2),
            "alignment_rate_percentage": round(align_rate_global, 2)
        },
        "implicitness_breakdown": impl_stats,
        "cyberbullying_breakdown": cb_stats,
        "top_agreed_trigger_words": agreed_counts,
        "detailed_sample_results": results
    }

    json_path = output_dir / "dual_xai_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\n💾 Laporan JSON lengkap tersimpan di: {json_path}")

    # Generate Tabel LaTeX Bab 4
    e0_m = impl_stats.get("E0", {}).get("mean_alignment", 0.0)
    e0_r = impl_stats.get("E0", {}).get("alignment_rate", 0.0)
    e1_m = impl_stats.get("E1", {}).get("mean_alignment", 0.0)
    e1_r = impl_stats.get("E1", {}).get("alignment_rate", 0.0)
    e2_m = impl_stats.get("E2", {}).get("mean_alignment", 0.0)
    e2_r = impl_stats.get("E2", {}).get("alignment_rate", 0.0)

    latex_code = f"""% Tabel Bab 4: Hasil Evaluasi Empiris Penyelarasan Dual-XAI (SHAP vs CoT Rationale)
\\begin{{table}}[htbp]
\\centering
\\caption{{Evaluasi Konsistensi Keterjelasan Ganda (Dual-XAI Alignment) antara Atribusi Token SHAP Kuantitatif dan Penalaran Kualitatif LLaMA-3 pada Berbagai Tingkat Keterbukaan Ujaran.}}
\\label{{tab:dual_xai_consistency}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Tingkat Keterbukaan Ujaran}} & \\textbf{{Jumlah Sampel}} & \\textbf{{Rata-rata Keselarasan (\\%)}} & \\textbf{{Konsistensi ($\\ge 50$\\%)}}\\\\
\\hline
E0 (Sangat Eksplisit)      & {impl_stats.get('E0', {}).get('count', 0)} & {e0_m:.2f}\\% & {e0_r:.2f}\\% \\\\
E1 (Semi-Tersirat)          & {impl_stats.get('E1', {}).get('count', 0)} & {e1_m:.2f}\\% & {e1_r:.2f}\\% \\\\
E2 (Tersirat / Kontekstual) & {impl_stats.get('E2', {}).get('count', 0)} & {e2_m:.2f}\\% & {e2_r:.2f}\\% \\\\
\\hline
\\textbf{{Total / Rata-rata Global}} & \\textbf{{{total_test}}} & \\textbf{{{mean_align_global:.2f}\\%}} & \\textbf{{{align_rate_global:.2f}\\%}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    latex_path = output_dir / "table_dual_xai_breakdown.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"📝 Tabel LaTeX Bab 4 tersimpan di  : {latex_path}")

    print("=" * 75)
    print("🎉 EVALUASI DUAL EXPLAINABILITY BERHASIL DISELESAIKAN!")
    print("=" * 75)

    return summary_report


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Dual Explainability SHAP vs CoT")
    parser.add_argument("--test-csv", type=str, default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--max-samples", type=int, default=None, help="Maksimal sampel uji yang dievaluasi")
    args = parser.parse_args()

    evaluate_dual_xai_benchmark(
        test_csv_path=Path(args.test_csv),
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
