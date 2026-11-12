#!/usr/bin/env python3
"""
scripts/evaluate_consistency.py
-------------------------------
Evaluasi Reliabilitas, Konsistensi, Sensitivitas Prompt, & Error Analysis (Tanpa Human GT)
untuk Penelitian Deteksi Cyberbullying Berbasis LLM Pseudo-Labels.

Catatan Metodologis:
Seluruh metrik di modul ini mengukur **konsistensi internal LLM, kesepakatan guru-murid
(teacher-student consistency), serta analisis kesalahan terstratifikasi** terhadap
LLM-generated pseudo-labels, dan secara eksplisit TIDAK diklaim sebagai
Human Inter-Annotator Agreement (IAA).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import EVAL_REPORT_DIR, FINAL_DIR, MODELS_DIR
from core.data_understanding import CUE_FAMILY_SLICES, TOTAL_VECTOR_DIM, clean_and_normalize_social_text
from core.fusion_model import load_hybrid_classifier_with_manifest, predict_hybrid_text
from core.model_engine import run_single_annotation


def compute_krippendorff_alpha_nominal(ratings_matrix: List[List[str]]) -> Optional[float]:
    """
    Menghitung Krippendorff's Alpha (nominal) untuk N item dan R pengulangan/annotator LLM.
    `ratings_matrix`: list per item berisi list kode kelas dari R pengulangan.
    """
    if not ratings_matrix or len(ratings_matrix) < 2:
        return None
    all_vals: List[str] = []
    pair_disagreements = 0.0
    total_pairs = 0.0

    for item_ratings in ratings_matrix:
        clean_r = [str(r) for r in item_ratings if r is not None and str(r).strip() != ""]
        m_u = len(clean_r)
        if m_u < 2:
            continue
        all_vals.extend(clean_r)
        for i in range(m_u):
            for j in range(i + 1, m_u):
                total_pairs += 1.0
                if clean_r[i] != clean_r[j]:
                    pair_disagreements += 1.0

    if total_pairs == 0 or len(set(all_vals)) <= 1:
        return 1.0 if total_pairs > 0 else None

    d_o = pair_disagreements / total_pairs
    n_tot = len(all_vals)
    counts = pd.Series(all_vals).value_counts()
    d_e = 1.0 - float(sum((c * (c - 1)) for c in counts) / max(1, n_tot * (n_tot - 1)))
    if d_e <= 1e-12:
        return 1.0
    return round(float(1.0 - (d_o / d_e)), 4)


def measure_llm_annotation_consistency(
    engine: Any,
    texts: List[str],
    n_repeats: int = 3,
    temperatures: Optional[List[float]] = None,
    prompt_variants: Optional[List[Optional[str]]] = None,
) -> Dict[str, Any]:
    """
    Mengukur konsistensi anotasi ulang LLM (`repeated annotation agreement`) dan
    sensitivitas prompt (`prompt sensitivity`) pada sampel komentar.
    """
    temps = temperatures or [0.0, 0.1, 0.3][: max(2, n_repeats)]
    if len(temps) < n_repeats:
        temps = (temps * ((n_repeats // len(temps)) + 1))[:n_repeats]

    repeat_labels_bin: List[List[str]] = []
    repeat_labels_type: List[List[str]] = []
    prompt_sens_pairs: List[bool] = []

    for txt in texts:
        run_bins: List[str] = []
        run_types: List[str] = []
        for rep_idx in range(n_repeats):
            res = run_single_annotation(
                engine=engine,
                text=txt,
                temperature=float(temps[rep_idx]),
                max_new_tokens=260,
            )
            run_bins.append(str(res.get("cyberbullying", "UNKNOWN")))
            run_types.append(str(res.get("type", "UNKNOWN")))
        repeat_labels_bin.append(run_bins)
        repeat_labels_type.append(run_types)

        if prompt_variants and len(prompt_variants) >= 2:
            r0 = run_single_annotation(engine=engine, text=txt, system_prompt=prompt_variants[0], temperature=0.0)
            r1 = run_single_annotation(engine=engine, text=txt, system_prompt=prompt_variants[1], temperature=0.0)
            prompt_sens_pairs.append(str(r0.get("cyberbullying")) == str(r1.get("cyberbullying")))

    unanimous_bin = sum(1 for rs in repeat_labels_bin if len(set(rs)) == 1) / max(1, len(repeat_labels_bin))
    unanimous_type = sum(1 for rs in repeat_labels_type if len(set(rs)) == 1) / max(1, len(repeat_labels_type))

    first_col = [r[0] for r in repeat_labels_bin]
    second_col = [r[1] if len(r) > 1 else r[0] for r in repeat_labels_bin]
    kappa_pair = float(cohen_kappa_score(first_col, second_col)) if len(set(first_col + second_col)) > 1 else 1.0

    return {
        "evaluation_scope": "llm_internal_consistency_not_human_iaa",
        "sample_size": len(texts),
        "n_repeats": n_repeats,
        "temperatures": temps,
        "binary_unanimous_agreement_rate": round(unanimous_bin, 4),
        "type_unanimous_agreement_rate": round(unanimous_type, 4),
        "binary_pairwise_cohen_kappa": round(kappa_pair, 4),
        "binary_krippendorff_alpha": compute_krippendorff_alpha_nominal(repeat_labels_bin),
        "type_krippendorff_alpha": compute_krippendorff_alpha_nominal(repeat_labels_type),
        "prompt_sensitivity_agreement_rate": (
            round(float(np.mean(prompt_sens_pairs)), 4) if prompt_sens_pairs else None
        ),
    }


def evaluate_consistency_suite(
    test_csv: Path = FINAL_DIR / "test.csv",
    full_csv: Path = FINAL_DIR / "iicd_cleaned_indobertweet.csv",
    hybrid_weights: Path = MODELS_DIR / "hybrid_fusion_indobertweet" / "best_hybrid_model.pt",
    output_dir: Path = EVAL_REPORT_DIR,
    live_engine: Optional[Any] = None,
    live_sample_n: int = 10,
) -> Dict[str, Any]:
    """
    Menjalankan rangkaian evaluasi non-GT lengkap:
    1. Distribusi & rasio ketidakseimbangan kelas per split & dimensi taksonomi.
    2. Statistik aktivasi 5 famili bukti kontekstual 33D (`Linguistic`, `Cultural`, `Emotional`, `Pragmatic`, `Contextual`).
    3. Kesepakatan Teacher LLM vs Hybrid Model (Agreement %, Cohen's Kappa) pada seluruh 6 heads.
    4. Analisis kesalahan terstratifikasi (`E0/E1/E2` & kategori `type`).
    5. Opsional: konsistensi anotasi ulang & sensitivitas prompt SumoPod LLM jika `live_engine` tersedia.
    """
    if not test_csv.exists():
        raise FileNotFoundError(f"Berkas test.csv tidak ditemukan: {test_csv}")

    df_test = pd.read_csv(test_csv)
    df_full = pd.read_csv(full_csv) if full_csv.exists() else df_test.copy()

    # 1. Class distribution & imbalance audit
    class_audit: Dict[str, Any] = {}
    for col_name in [
        "target_cyberbullying",
        "target_type",
        "target_emotion",
        "target_cultural_context",
        "target_implicit_level",
        "target_severity",
    ]:
        src_col = col_name if col_name in df_full.columns else col_name.replace("target_", "")
        if src_col in df_full.columns:
            vc = df_full[src_col].fillna("UNKNOWN").astype(str).value_counts()
            max_c = int(vc.max()) if len(vc) > 0 else 0
            min_c = int(vc.min()) if len(vc) > 0 else 0
            class_audit[src_col] = {
                "counts": {str(k): int(v) for k, v in vc.items()},
                "imbalance_ratio": round(max_c / max(1, min_c), 2),
                "missing_unknown_count": int(vc.get("UNKNOWN", 0)),
            }

    # 2. Stream B Contextual Evidence Activation Summary
    test_vec_path = test_csv.parent / "test_llm_vectors.npy"
    cue_activation_stats: Dict[str, Any] = {}
    if test_vec_path.exists():
        vecs = np.load(test_vec_path)
        for fam, slc in CUE_FAMILY_SLICES.items():
            sub_v = vecs[:, slc]
            cue_activation_stats[fam] = {
                "mean_activation": round(float(np.mean(sub_v)), 4),
                "std_activation": round(float(np.std(sub_v)), 4),
                "nonzero_rate": round(float(np.mean(np.any(sub_v > 0.0, axis=1))), 4),
            }

    # 3. Teacher vs Hybrid Agreement across 6 Heads + Sliced Error Analysis
    teacher_student_agreement: Dict[str, Any] = {"available": False}
    error_breakdown: Dict[str, Any] = {}

    if hybrid_weights.exists():
        model, tokenizer, device, manifest = load_hybrid_classifier_with_manifest(hybrid_weights)
        threshold = float(manifest.get("optimal_threshold", 0.50)) if manifest else 0.50

        pred_rows: List[Dict[str, Any]] = []
        for idx, row in df_test.iterrows():
            ann_dict = {
                "comment": row.get("comment", ""),
                "explanation": row.get("llm_explanation", row.get("explanation", "")),
                "confidence": row.get("llm_confidence", row.get("confidence", None)),
            }
            res = predict_hybrid_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text=str(row.get("comment_normalized", row.get("comment", ""))),
                annotation_dict=ann_dict,
                threshold=threshold,
                manifest=manifest,
            )
            pred_rows.append(
                {
                    "idx": int(idx),
                    "comment": str(row.get("comment", "")),
                    "teacher_cyberbullying": str(row.get("target_cyberbullying", row.get("cyberbullying", "UNKNOWN"))),
                    "hybrid_cyberbullying": res["cyberbullying"],
                    "hybrid_prob": res["prob_cyberbullying"],
                    "gate_weight_z": res["gate_weight_z"],
                    "teacher_type": str(row.get("target_type", row.get("type", "UNKNOWN"))),
                    "hybrid_type": res["type"],
                    "teacher_emotion": str(row.get("target_emotion", row.get("emotion", "UNKNOWN"))),
                    "hybrid_emotion": res["emotion"],
                    "teacher_cultural": str(row.get("target_cultural_context", row.get("cultural_context", "UNKNOWN"))),
                    "hybrid_cultural": res["cultural_context"],
                    "teacher_implicit": str(row.get("target_implicit_level", row.get("implicit_level", "UNKNOWN"))),
                    "hybrid_implicit": res["implicit_level"],
                    "teacher_severity": str(row.get("target_severity", row.get("severity", "UNKNOWN"))),
                    "hybrid_severity": res["severity"],
                }
            )

        df_cmp = pd.DataFrame(pred_rows)
        head_agreements: Dict[str, Any] = {}
        for head_key in ("cyberbullying", "type", "emotion", "cultural", "implicit", "severity"):
            t_col = f"teacher_{head_key}"
            h_col = f"hybrid_{head_key}"
            valid = df_cmp[df_cmp[t_col] != "UNKNOWN"]
            if len(valid) > 0:
                yt = valid[t_col].tolist()
                yh = valid[h_col].tolist()
                acc_agree = float(np.mean([a == b for a, b in zip(yt, yh)]))
                kappa = float(cohen_kappa_score(yt, yh)) if len(set(yt + yh)) > 1 else 1.0
                head_agreements[head_key] = {
                    "support": int(len(valid)),
                    "agreement_rate": round(acc_agree, 4),
                    "cohen_kappa": round(kappa, 4),
                }

        teacher_student_agreement = {
            "available": True,
            "model_architecture": "Hybrid Dual-Stream Gated Fusion (Active Stream A + Stream B)",
            "mean_gate_weight_z": round(float(df_cmp["gate_weight_z"].mean()), 4),
            "head_agreements": head_agreements,
        }

        # Error analysis across implicit levels (FP / FN vs teacher pseudo-label)
        for lvl in ("E0", "E1", "E2"):
            sub = df_cmp[df_cmp["teacher_implicit"] == lvl]
            if len(sub) > 0:
                fp_n = int(((sub["teacher_cyberbullying"] == "NO") & (sub["hybrid_cyberbullying"] == "YES")).sum())
                fn_n = int(((sub["teacher_cyberbullying"] == "YES") & (sub["hybrid_cyberbullying"] == "NO")).sum())
                disagree = sub[sub["teacher_cyberbullying"] != sub["hybrid_cyberbullying"]]
                error_breakdown[lvl] = {
                    "count": int(len(sub)),
                    "false_positives_vs_pseudo_label": fp_n,
                    "false_negatives_vs_pseudo_label": fn_n,
                    "agreement_rate": round(float((sub["teacher_cyberbullying"] == sub["hybrid_cyberbullying"]).mean()), 4),
                    "sample_disagreements": [
                        {
                            "comment": r["comment"][:120],
                            "teacher": r["teacher_cyberbullying"],
                            "hybrid": r["hybrid_cyberbullying"],
                            "hybrid_prob": r["hybrid_prob"],
                        }
                        for _, r in disagree.head(3).iterrows()
                    ],
                }

    # 4. Optional Live LLM Repeat/Sensitivity Probe
    llm_repeat_consistency: Dict[str, Any] = {
        "status": "not_run_no_live_engine",
        "note": "Berikan --run-live-consistency dengan SUMOPOD_API_KEY aktif (atau engine instance) untuk mengukur multi-run kappa & sensitivitas prompt.",
    }
    if live_engine is not None:
        sample_texts = df_test["comment"].dropna().astype(str).head(max(2, live_sample_n)).tolist()
        llm_repeat_consistency = measure_llm_annotation_consistency(engine=live_engine, texts=sample_texts, n_repeats=3)

    report = {
        "evaluation_policy": "llm_pseudo_label_consistency_and_reliability",
        "disclaimer": (
            "Seluruh metrik konsistensi dan kesepakatan mengevaluasi stabilitas internal LLM serta "
            "konsistensi guru-murid (LLM vs Hybrid Model) terhadap pseudo-label tanpa Human Ground Truth."
        ),
        "dataset_sizes": {"full_curated": int(len(df_full)), "test_split": int(len(df_test))},
        "class_distribution_audit": class_audit,
        "stream_b_cue_family_stats": cue_activation_stats,
        "teacher_student_agreement": teacher_student_agreement,
        "error_analysis_by_implicit_level": error_breakdown,
        "llm_repeat_consistency": llm_repeat_consistency,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "consistency_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"💾 Consistency Evaluation Report tersimpan di: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Non-GT Consistency, Sensitivity & Error Analysis")
    parser.add_argument("--test-csv", type=str, default=str(FINAL_DIR / "test.csv"))
    parser.add_argument("--full-csv", type=str, default=str(FINAL_DIR / "iicd_cleaned_indobertweet.csv"))
    parser.add_argument("--hybrid-weights", type=str, default=str(MODELS_DIR / "hybrid_fusion_indobertweet" / "best_hybrid_model.pt"))
    parser.add_argument("--output-dir", type=str, default=str(EVAL_REPORT_DIR))
    args = parser.parse_args()

    evaluate_consistency_suite(
        test_csv=Path(args.test_csv),
        full_csv=Path(args.full_csv),
        hybrid_weights=Path(args.hybrid_weights),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

