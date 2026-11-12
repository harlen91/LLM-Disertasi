#!/usr/bin/env python3
"""
scripts/evaluate_dual_xai.py
----------------------------
Evaluasi Empiris Dual Explainability (Real SHAP Library vs SumoPod LLM CoT Rationale):
1. Menggunakan library `shap` sebenarnya pada `HybridFusionModel` (default) atau baseline IndoBERTweet.
2. Menghitung atribusi lokal, global feature importance, & faithfulness (comprehensiveness).
3. Memisahkan metrik ilmiah faithfulness dari metrik diagnostik pencocokan kata leksikal (`align_shap_with_cot`).
4. Menghormati target_source / modality masking agar tidak membocorkan label pseudo ke input XAI.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import EVAL_REPORT_DIR, FINAL_DIR, MODELS_DIR
from core.shap_explainer import (
    HybridSHAPExplainer,
    align_shap_with_cot,
    compute_faithfulness_comprehensiveness,
)

DEFAULT_TEST_CSV = FINAL_DIR / "test.csv"
DEFAULT_HYBRID_WEIGHTS = MODELS_DIR / "hybrid_fusion_indobertweet" / "best_hybrid_model.pt"
DEFAULT_BASELINE_DIR = MODELS_DIR / "indobertweet_cyberbullying" / "best_model"


def evaluate_dual_xai_benchmark(
    test_csv_path: Path,
    model_path: Optional[Path] = None,
    model_type: str = "hybrid",
    output_dir: Path = EVAL_REPORT_DIR,
    max_samples: Optional[int] = None,
    shap_max_evals: int = 100,
) -> Dict[str, Any]:
    print("=" * 76)
    print("🔬 EVALUASI EMPIRIS DUAL EXPLAINABILITY (REAL SHAP ⟷ SUMOPOD LLM CoT)")
    print("=" * 76)

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Berkas test set tidak ditemukan: {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    if max_samples is not None and max_samples < len(df_test):
        df_test = df_test.head(max_samples)
    total_test = len(df_test)

    resolved_model = model_path or (DEFAULT_HYBRID_WEIGHTS if model_type == "hybrid" else DEFAULT_BASELINE_DIR)
    explainer = HybridSHAPExplainer(model_path=resolved_model, model_type=model_type)

    # Stream B (33D Non-Leaky Contextual Evidence Cues) AKTIF konsisten untuk Hybrid model
    target_source = str(df_test["target_source"].iloc[0]) if "target_source" in df_test.columns and len(df_test) > 0 else "pseudo_labeled_llm"
    test_vec_path = test_csv_path.parent / "test_llm_vectors.npy"
    test_vecs = None
    if model_type == "hybrid" and test_vec_path.exists():
        test_vecs = np.load(test_vec_path)[:total_test]
        explainer.background_vectors = test_vecs[: min(16, len(test_vecs))]

    results: List[Dict[str, Any]] = []
    comprehensiveness_scores: List[float] = []

    for idx, row in df_test.iterrows():
        comment = str(row.get("comment_normalized", row.get("comment", ""))).strip()
        explanation = str(row.get("llm_explanation", row.get("explanation", ""))).strip()
        cb_label = str(row.get("target_cyberbullying", row.get("cyberbullying", "NO"))).upper()
        impl_level = str(row.get("target_implicit_level", row.get("implicit_level", "E0"))).upper()
        b_type = str(row.get("target_type", row.get("type", "NONE"))).upper()
        vec = test_vecs[idx] if test_vecs is not None else None

        shap_res = explainer.explain_text(comment, llm_vector=vec, max_evals=shap_max_evals, top_k=5)
        comp_score = compute_faithfulness_comprehensiveness(explainer, comment, shap_res, llm_vector=vec)
        if comp_score is not None:
            comprehensiveness_scores.append(comp_score)

        align_info = align_shap_with_cot(shap_res, explanation, comprehensiveness_score=comp_score)
        results.append(
            {
                "idx": int(idx),
                "comment": str(row.get("comment", comment)),
                "cyberbullying": cb_label,
                "implicit_level": impl_level,
                "type": b_type,
                "cyberbullying_prob": shap_res["cyberbullying_prob"],
                "top_positive_shap_tokens": [t["token"] for t in shap_res.get("top_positive_tokens", [])],
                "top_negative_shap_tokens": [t["token"] for t in shap_res.get("top_negative_tokens", [])],
                "faithfulness_comprehensiveness": comp_score,
                "diagnostic_lexical_aligned_tokens": align_info["aligned_tokens"],
                "diagnostic_lexical_alignment_score": align_info["alignment_score"],
                "is_lexically_aligned": align_info["is_aligned"],
            }
        )

        if (idx + 1) % 25 == 0 or (idx + 1) == total_test:
            print(f"  Processed {idx + 1}/{total_test} sampel...")

    df_res = pd.DataFrame(results)
    mean_lexical = float(df_res["diagnostic_lexical_alignment_score"].mean()) if len(df_res) > 0 else 0.0
    lexical_rate = float(df_res["is_lexically_aligned"].mean() * 100.0) if len(df_res) > 0 else 0.0
    mean_comp = round(float(np.mean(comprehensiveness_scores)), 4) if comprehensiveness_scores else None

    impl_stats: Dict[str, Any] = {}
    for lvl in ["E0", "E1", "E2"]:
        sub = df_res[df_res["implicit_level"] == lvl]
        if len(sub) > 0:
            impl_stats[lvl] = {
                "count": int(len(sub)),
                "mean_diagnostic_lexical_alignment": round(float(sub["diagnostic_lexical_alignment_score"].mean()), 2),
                "lexical_alignment_rate": round(float(sub["is_lexically_aligned"].mean() * 100.0), 2),
                "mean_comprehensiveness": round(float(sub["faithfulness_comprehensiveness"].dropna().mean()), 4)
                if sub["faithfulness_comprehensiveness"].notna().any()
                else None,
            }

    global_imp = explainer.compute_global_importance(
        texts=df_res["comment"].tolist(),
        llm_vectors=test_vecs,
        max_samples=min(20, len(df_res)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_report = {
        "evaluation_name": f"Dual Explainability Benchmark (SHAP Library [{model_type}] vs SumoPod LLM CoT)",
        "shap_method": "shap.Explainer(shap.maskers.Text) + shap.Explainer(Tabular Permutation on 33D Contextual Cues)",
        "model_type": model_type,
        "target_source": target_source,
        "llm_vector_conditioning": "active_33d_contextual_evidence" if test_vecs is not None else "text_only_baseline",
        "total_test_samples": total_test,
        "scientific_faithfulness_metrics": {
            "mean_comprehensiveness": mean_comp,
            "samples_measured": len(comprehensiveness_scores),
        },
        "diagnostic_lexical_heuristics": {
            "disclaimer": "Metrik lexical overlap bersifat diagnostik heuristik, bukan ukuran validitas kausal tervalidasi.",
            "mean_lexical_alignment_score": round(mean_lexical, 2),
            "lexical_alignment_rate_percentage": round(lexical_rate, 2),
        },
        "implicitness_breakdown": impl_stats,
        "global_feature_importance": global_imp,
        "detailed_sample_results": results,
    }

    json_path = output_dir / "dual_xai_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    e0 = impl_stats.get("E0", {})
    e1 = impl_stats.get("E1", {})
    e2 = impl_stats.get("E2", {})
    latex_code = f"""% Tabel Bab 4: Evaluasi Dual-XAI (SHAP Kuantitatif vs Diagnostik Leksikal SumoPod LLM CoT)
\\begin{{table}}[htbp]
\\centering
\\caption{{Evaluasi Keterjelasan SHAP dan Diagnostik Keselarasan Leksikal dengan Penalaran SumoPod LLM ({model_type})}}
\\label{{tab:dual_xai_consistency}}
\\begin{{tabular}}{{lccc}}
\\hline
\\textbf{{Tingkat Ujaran}} & \\textbf{{Sampel}} & \\textbf{{Comprehensiveness ($\\Delta p$)}} & \\textbf{{Diagnostik Overlap (\\%)}} \\\\
\\hline
E0 (Eksplisit) & {e0.get('count', 0)} & {e0.get('mean_comprehensiveness', 'N/A')} & {e0.get('mean_diagnostic_lexical_alignment', 0.0):.2f}\\% \\\\
E1 (Semi-Tersirat) & {e1.get('count', 0)} & {e1.get('mean_comprehensiveness', 'N/A')} & {e1.get('mean_diagnostic_lexical_alignment', 0.0):.2f}\\% \\\\
E2 (Tersirat) & {e2.get('count', 0)} & {e2.get('mean_comprehensiveness', 'N/A')} & {e2.get('mean_diagnostic_lexical_alignment', 0.0):.2f}\\% \\\\
\\hline
\\textbf{{Global}} & \\textbf{{{total_test}}} & \\textbf{{{mean_comp if mean_comp is not None else 'N/A'}}} & \\textbf{{{mean_lexical:.2f}\\%}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(output_dir / "table_dual_xai_breakdown.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"💾 Dual-XAI JSON & LaTeX tersimpan di: {output_dir}")
    return summary_report


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Dual Explainability (Real SHAP vs SumoPod LLM)")
    parser.add_argument("--test-csv", type=str, default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="hybrid", choices=["hybrid", "baseline"])
    parser.add_argument("--output-dir", type=str, default=str(EVAL_REPORT_DIR))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-evals", type=int, default=100)
    args = parser.parse_args()

    evaluate_dual_xai_benchmark(
        test_csv_path=Path(args.test_csv),
        model_path=Path(args.model_path) if args.model_path else None,
        model_type=args.model_type,
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        shap_max_evals=args.max_evals,
    )


if __name__ == "__main__":
    main()
