#!/usr/bin/env python3
"""
scripts/generate_dissertation_report.py
---------------------------------------
Generator Laporan Evaluasi Komprehensif, Visualisasi Publikasi (300 DPI),
dan Tabel LaTeX untuk Bab 4 Naskah Disertasi.

Perbaikan Kritis:
1. Menghapus simulasi probabilitas dari `y_true` ketika checkpoint tidak tersedia;
   skrip menolak membuat metrik palsu dan meminta checkpoint model terlatih.
2. Menggunakan `find_optimal_threshold_on_val` STRICTLY dari `val.csv` atau `model_manifest.json`.
3. Memuat `ablation_comparison.json` dan `dual_xai_evaluation.json` jika tersedia untuk tabel ablasi lengkap.
4. Menghasilkan metrik lengkap: Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrix,
   ROC-AUC, PR-AUC, Class-wise, Calibration (ECE, MCE, Brier, Log-Loss), Bootstrap 95% CI,
   E0/E1/E2 breakdown, Cyberbullying Type breakdown, dan Error Analysis FP/FN.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import EVAL_REPORT_DIR, FINAL_DIR, MODELS_DIR
from core.data_understanding import clean_and_normalize_social_text
from core.evaluation_metrics import (
    compute_comprehensive_classification_metrics,
    find_optimal_threshold_on_val,
    load_model_manifest,
)

DEFAULT_MODEL_DIR = MODELS_DIR / "indobertweet_cyberbullying" / "best_model"
DEFAULT_HYBRID_DIR = MODELS_DIR / "hybrid_fusion_indobertweet"
DEFAULT_TEST_CSV = FINAL_DIR / "test.csv"
DEFAULT_REPORT_DIR = EVAL_REPORT_DIR


def evaluate_baseline_model_probs(
    model,
    tokenizer,
    device: str,
    df: pd.DataFrame,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mengembalikan `(y_true, probs_pos)` secara konsisten menggunakan `comment_normalized`."""
    text_col = "comment_normalized" if "comment_normalized" in df.columns else "comment"
    texts = [clean_and_normalize_social_text(str(t)) for t in df[text_col]]
    y_true = df["label"].to_numpy()
    probs_pos: List[float] = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_pos.extend(probs.tolist())

    return y_true, np.asarray(probs_pos, dtype=float)


def plot_publication_figures(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Menghasilkan grafik 300 DPI siap publikasi."""
    sns.set_theme(style="white")
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = np.array(metrics["confusion_matrix_2x2"], dtype=int)

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5), dpi=300)
    cm_labels = ["Non-Cyberbullying (0)", "Cyberbullying (1)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cm_labels,
        yticklabels=cm_labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Confusion Matrix pada Data Uji Independen", fontsize=12, pad=12, weight="bold")
    plt.xlabel("Prediksi Model", fontsize=10, weight="bold")
    plt.ylabel("Label Referensi", fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_confusion_matrix.png", dpi=300)
    plt.close()

    # 2. Implicitness Breakdown Plot (E0/E1/E2)
    imp = metrics.get("implicitness_breakdown", {})
    if imp:
        levels = list(imp.keys())
        accs = [imp[l]["accuracy"] * 100.0 for l in levels]
        plt.figure(figsize=(7, 4.5), dpi=300)
        colors = ["#2b6cb0", "#4299e1", "#90cdf4"]
        bars = plt.bar(levels, accs, color=colors[: len(levels)], width=0.5)
        plt.title("Akurasi Berdasarkan Tingkat Keterbukaan Ujaran (E0/E1/E2)", fontsize=12, pad=12, weight="bold")
        plt.xlabel("Dimensi Keterbukaan", fontsize=10)
        plt.ylabel("Akurasi (%)", fontsize=10)
        plt.ylim(0, 108)
        for bar in bars:
            yv = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yv + 2, f"{yv:.1f}%", ha="center", va="bottom", weight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "figure_implicitness_accuracy.png", dpi=300)
        plt.close()


def generate_latex_tables(
    metrics: Dict[str, Any],
    ablation: Optional[Dict[str, Any]] = None,
) -> str:
    cm = metrics["confusion_matrix"]
    cal = metrics.get("calibration", {})
    ci = metrics.get("confidence_intervals_95", {})
    pc0 = metrics["per_class"]["non_cyberbullying"]
    pc1 = metrics["per_class"]["cyberbullying"]

    latex_code = f"""% ===============================================================
% TABEL 1: METRIK KINERJA KESELURUHAN (BAB 4 DISERTASI)
% ===============================================================
\\begin{{table}}[htbp]
\\centering
\\caption{{Metrik Evaluasi Komprehensif pada Data Uji (Test Set, $\\tau^*={metrics['threshold']:.2f}$)}}
\\label{{tab:overall_performance}}
\\begin{{tabular}}{{lc}}
\\hline
\\textbf{{Metrik Evaluasi}} & \\textbf{{Nilai (95\\% Bootstrap CI)}} \\\\
\\hline
Accuracy & {metrics['accuracy']:.4f} ({ci.get('accuracy', ['-', '-'])}) \\\\
\\textbf{{Macro F1-Score}} & \\textbf{{{metrics['macro_f1']:.4f}}} ({ci.get('macro_f1', ['-', '-'])}) \\\\
Weighted F1-Score & {metrics['weighted_f1']:.4f} \\\\
Macro Precision & {metrics['macro_precision']:.4f} \\\\
Macro Recall & {metrics['macro_recall']:.4f} \\\\
ROC-AUC & {metrics.get('roc_auc', 'N/A')} ({ci.get('roc_auc', ['-', '-'])}) \\\\
PR-AUC & {metrics.get('pr_auc', 'N/A')} ({ci.get('pr_auc', ['-', '-'])}) \\\\
Expected Calibration Error (ECE) & {cal.get('ece', 'N/A')} \\\\
Brier Score & {cal.get('brier_score', 'N/A')} \\\\
Log-Loss & {cal.get('log_loss', 'N/A')} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}

% ===============================================================
% TABEL 2: KINERJA PER KELAS & MATRIKS KONFUSI
% ===============================================================
\\begin{{table}}[htbp]
\\centering
\\caption{{Laporan Klasifikasi per Kelas}}
\\label{{tab:class_report}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Kelas}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1-Score}} & \\textbf{{Support}} \\\\
\\hline
Non-Cyberbullying (0) & {pc0['precision']:.4f} & {pc0['recall']:.4f} & {pc0['f1']:.4f} & {pc0['support']} \\\\
Cyberbullying (1) & {pc1['precision']:.4f} & {pc1['recall']:.4f} & {pc1['f1']:.4f} & {pc1['support']} \\\\
\\hline
\\multicolumn{{5}}{{l}}{{\\footnotesize \\textit{{Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}}}}} \\\\
\\end{{tabular}}
\\end{{table}}
"""

    imp = metrics.get("implicitness_breakdown", {})
    if imp:
        latex_code += """
% ===============================================================
% TABEL 3: KETAHANAN TERHADAP TINGKAT KETERBUKAAN (E0/E1/E2)
% ===============================================================
\\begin{table}[htbp]
\\centering
\\caption{Evaluasi Kinerja pada Ujaran Eksplisit hingga Implisit (E0/E1/E2)}
\\label{tab:implicit_analysis}
\\begin{tabular}{lccc}
\\hline
\\textbf{Tingkat} & \\textbf{Support} & \\textbf{Akurasi (\\%)} & \\textbf{Macro-F1} \\\\
\\hline
"""
        for lvl, info in imp.items():
            latex_code += f"{lvl} & {info['count']} & {info['accuracy']*100:.2f}\\% & {info['macro_f1']:.4f} \\\\\n"
        latex_code += "\\hline\n\\end{tabular}\n\\end{table}\n"

    if ablation:
        latex_code += """
% ===============================================================
% TABEL 4: ABLATION STUDY (INDOBERTWEET-ONLY vs LLM-ONLY vs HYBRID)
% ===============================================================
\\begin{table}[htbp]
\\centering
\\caption{Studi Ablasi pada Test Set yang Sama}
\\label{tab:ablation_study}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Varian Model} & \\textbf{Accuracy} & \\textbf{Macro-F1} & \\textbf{ROC-AUC} & \\textbf{PR-AUC} \\\\
\\hline
"""
        for key in ["model_a_indobertweet_only", "model_b_llm_only", "model_c_hybrid_fusion"]:
            m = ablation.get(key)
            if not m:
                continue
            name = m.get("model_name", key)
            if m.get("accuracy") is None:
                latex_code += f"{name} & \\multicolumn{{4}}{{c}}{{Diblokir ({m.get('status', 'butuh human GT')})}} \\\\\n"
            else:
                latex_code += (
                    f"{name} & {m['accuracy']:.4f} & {m['macro_f1']:.4f} & "
                    f"{m.get('roc_auc', 'N/A')} & {m.get('pr_auc', 'N/A')} \\\\\n"
                )
        latex_code += "\\hline\n\\end{tabular}\n\\end{table}\n"

    return latex_code


def generate_markdown_report(
    metrics: Dict[str, Any],
    target_source: str,
    ablation: Optional[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "dissertation_evaluation_report.md"
    cm = metrics["confusion_matrix"]
    cal = metrics.get("calibration", {})
    ci = metrics.get("confidence_intervals_95", {})
    ea = metrics.get("error_analysis", {})

    warning_banner = (
        "> [!WARNING]\n"
        "> **Status Label Referensi: `pseudo_labeled_llm`**. Dataset uji belum memiliki kolom ground truth manusia yang tervalidasi secara independen. Seluruh metrik di bawah ini mengukur kesepakatan/distilasi terhadap pseudo-label LLM secara bebas kebocoran (*leakage-free*), bukan akurasi terhadap gold standard manusia.\n"
        if target_source != "human_ground_truth"
        else "> [!NOTE]\n> **Status Label Referensi: `human_ground_truth` (Tervalidasi)**. Evaluasi dijalankan secara independen terhadap ground truth manusia.\n"
    )

    content = f"""# 📑 Laporan Evaluasi Komprehensif Disertasi

{warning_banner}

## 1. Ringkasan Metrik Kinerja Utama (Test Set, $\\tau^* = {metrics['threshold']:.2f}$)

| Metrik Evaluasi | Nilai | 95% Bootstrap CI | Interpretasi |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **{metrics['accuracy']*100:.2f}%** ({metrics['accuracy']:.4f}) | {ci.get('accuracy')} | Proporsi prediksi tepat keseluruhan |
| **Macro F1-Score** | **{metrics['macro_f1']:.4f}** | {ci.get('macro_f1')} | Rata-rata harmonik F1 adil kelas (Utama) |
| **Weighted F1-Score** | {metrics['weighted_f1']:.4f} | - | Rata-rata F1 terbobot proporsi kelas |
| **Macro Precision / Recall** | {metrics['macro_precision']:.4f} / {metrics['macro_recall']:.4f} | - | Ketepatan vs sensitivitas makro |
| **ROC-AUC** | {metrics.get('roc_auc', 'N/A')} | {ci.get('roc_auc')} | Diskriminasi probabilitas ROC |
| **PR-AUC** | {metrics.get('pr_auc', 'N/A')} | {ci.get('pr_auc')} | Area kurva Precision-Recall |
| **Calibration (ECE / MCE)** | {cal.get('ece')} / {cal.get('mce')} | - | Kesesuaian kepercayaan probabilistik |
| **Brier Score / Log-Loss** | {cal.get('brier_score')} / {cal.get('log_loss')} | - | Akurasi probabilistik & ketidakpastian |

## 2. Kinerja per Kelas & Analisis Kesalahan (FP / FN)
- **Confusion Matrix**: `TN={cm['tn']}`, `FP={cm['fp']}`, `FN={cm['fn']}`, `TP={cm['tp']}`
- **False Positive Rate (FPR)**: {ea.get('false_positive_rate', 0.0)*100:.2f}% ({ea.get('false_positive_count', 0)} sampel)
- **False Negative Rate (FNR)**: {ea.get('false_negative_rate', 0.0)*100:.2f}% ({ea.get('false_negative_count', 0)} sampel)

## 3. Rincian Tingkat Keterbukaan (E0/E1/E2) & Tipe Cyberbullying

| Tingkat Keterbukaan | Jumlah | Benar | Akurasi (%) | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
"""
    for lvl, info in metrics.get("implicitness_breakdown", {}).items():
        content += f"| **{lvl}** | {info['count']} | {info['correct']} | {info['accuracy']*100:.2f}% | {info['macro_f1']:.4f} |\n"

    type_bd = metrics.get("type_breakdown", {})
    if type_bd:
        content += "\n| Tipe Taksonomi | Jumlah | Benar | Akurasi (%) |\n| :--- | :---: | :---: | :---: |\n"
        for t_code, info in type_bd.items():
            content += f"| **{t_code}** | {info['count']} | {info['correct']} | {info['accuracy']*100:.2f}% |\n"

    if ablation:
        content += "\n## 4. Studi Ablasi (Test Set Identik)\n\n| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC |\n| :--- | :---: | :---: | :---: | :---: |\n"
        for key in ["model_a_indobertweet_only", "model_b_llm_only", "model_c_hybrid_fusion"]:
            m = ablation.get(key)
            if not m:
                continue
            if m.get("accuracy") is None:
                content += f"| {m.get('model_name', key)} | N/A | N/A | N/A | {m.get('status', 'blocked')} |\n"
            else:
                content += f"| {m.get('model_name', key)} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m.get('roc_auc')} | {m.get('pr_auc')} |\n"

    content += "\n## 5. Artefak Grafik (300 DPI)\n- `figure_confusion_matrix.png`\n- `figure_implicitness_accuracy.png`\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    return md_path


def run_full_evaluation_pipeline(
    model_dir: Path = DEFAULT_MODEL_DIR,
    test_csv: Path = DEFAULT_TEST_CSV,
    output_dir: Path = DEFAULT_REPORT_DIR,
    hybrid_dir: Path = DEFAULT_HYBRID_DIR,
) -> Dict[str, Any]:
    print("=" * 75)
    print("🎓 MODUL EVALUASI & GENERATOR LAPORAN METRIK DISERTASI (BEBAS SIMULASI)")
    print("=" * 75)

    if not test_csv.exists():
        raise FileNotFoundError(f"Berkas test set tidak ditemukan di: {test_csv}")
    df_test = pd.read_csv(test_csv)
    target_source = str(df_test["target_source"].iloc[0]) if "target_source" in df_test.columns and len(df_test) > 0 else "pseudo_labeled_llm"

    has_baseline_weights = model_dir.exists() and (
        (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
    )
    if not has_baseline_weights:
        raise FileNotFoundError(
            f"Checkpoint baseline IndoBERTweet tidak ditemukan di '{model_dir}'. "
            "Laporan disertasi menolak menyimulasikan metrik dari label aktual. "
            "Jalankan 'python3 scripts/train_indobertweet.py' terlebih dahulu."
        )

    from scripts.evaluate_indobertweet import load_classifier

    model, tokenizer, device = load_classifier(model_dir)
    manifest = load_model_manifest(model_dir) or load_model_manifest(model_dir.parent)
    if manifest and manifest.get("optimal_threshold") is not None:
        opt_thresh = float(manifest["optimal_threshold"])
    else:
        val_csv = test_csv.parent / "val.csv"
        if val_csv.exists():
            _, val_probs = evaluate_baseline_model_probs(model, tokenizer, device, pd.read_csv(val_csv))
            opt_thresh, _, _ = find_optimal_threshold_on_val(val_probs, pd.read_csv(val_csv)["label"].to_numpy())
        else:
            opt_thresh = 0.50

    y_true, y_prob_pos = evaluate_baseline_model_probs(model, tokenizer, device, df_test)
    metrics = compute_comprehensive_classification_metrics(
        y_true=y_true,
        y_prob_pos=y_prob_pos,
        threshold=opt_thresh,
        df_meta=df_test,
    )
    metrics["target_source"] = target_source

    ablation_path = hybrid_dir / "ablation_comparison.json"
    ablation = None
    if ablation_path.exists():
        with open(ablation_path, "r", encoding="utf-8") as f:
            ablation = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    generate_markdown_report(metrics, target_source, ablation, output_dir)
    latex_str = generate_latex_tables(metrics, ablation)
    with open(output_dir / "dissertation_tables.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    plot_publication_figures(metrics, output_dir)
    print(f"✅ Laporan & tabel disertasi tersimpan di: {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generator Laporan Evaluasi Disertasi")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--test_csv", type=str, default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--hybrid_dir", type=str, default=str(DEFAULT_HYBRID_DIR))
    args = parser.parse_args()

    run_full_evaluation_pipeline(
        model_dir=Path(args.model_dir),
        test_csv=Path(args.test_csv),
        output_dir=Path(args.output_dir),
        hybrid_dir=Path(args.hybrid_dir),
    )


if __name__ == "__main__":
    main()
