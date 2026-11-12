import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add project root to path
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SUMOPOD_MODEL,
    EVAL_REPORT_DIR,
    FINAL_DIR,
    MODELS_DIR,
)
from core.data_understanding import extract_contextual_evidence_vector
from core.evaluation_metrics import load_model_manifest
from core.fusion_model import load_hybrid_classifier_with_manifest, predict_hybrid_text
from core.model_engine import run_single_annotation
from core.shap_explainer import (
    HybridSHAPExplainer,
    align_shap_with_cot,
    compute_faithfulness_comprehensiveness,
)
from data_manager.dataset_handler import sanitize_html_text
from scripts.curate_dataset import curate_and_normalize
from scripts.evaluate_consistency import evaluate_consistency_suite
from scripts.evaluate_dual_xai import evaluate_dual_xai_benchmark
from scripts.evaluate_indobertweet import get_saved_threshold, load_classifier, predict_text
from scripts.generate_dissertation_report import run_full_evaluation_pipeline
from scripts.train_hybrid_fusion import train_hybrid_pipeline
from scripts.train_indobertweet import train_model as train_baseline_model

CHECKPOINT_FILE = PROJECT_DIR / "output" / "checkpoints" / "checkpoint_Dataset-Research.csv"
MODEL_DIR = MODELS_DIR / "indobertweet_cyberbullying"
HYBRID_MODEL_DIR = MODELS_DIR / "hybrid_fusion_indobertweet"
REPORT_DIR = EVAL_REPORT_DIR


def render_model_training_page():
    st.markdown("## 🤖 Fine-Tuning & Evaluasi Transformer IndoBERTweet")
    st.markdown(
        "Latih model Transformer Encoder lokal (**IndoBERTweet**) menggunakan dataset hasil kurasi anotasi **SumoPod LLM API**. "
        "Menghasilkan model ringan berkecepatan tinggi (~20ms) dan metrik evaluasi lengkap untuk naskah disertasi."
    )

    tab_train, tab_eval, tab_test = st.tabs([
        "🚀 Pelatihan Model (Training)",
        "📊 Evaluasi & Tabel LaTeX Bab 4",
        "🧪 Playground Inferensi IndoBERTweet"
    ])

    # =========================================================================
    # TAB 1: PELATIHAN MODEL (TRAINING)
    # =========================================================================
    with tab_train:
        st.markdown("### 1. Status Dataset Latih & Kurasi")
        
        train_path = FINAL_DIR / "train.csv"
        val_path = FINAL_DIR / "val.csv"
        test_path = FINAL_DIR / "test.csv"
        
        has_dataset = train_path.exists() and val_path.exists() and test_path.exists()

        c1, c2, c3, c4 = st.columns(4)
        if has_dataset:
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            df_test = pd.read_csv(test_path)
            
            with c1:
                st.metric("Data Latih (Train 80%)", f"{len(df_train):,} baris")
            with c2:
                st.metric("Data Validasi (Val 10%)", f"{len(df_val):,} baris")
            with c3:
                st.metric("Data Uji (Test 10%)", f"{len(df_test):,} baris")
            with c4:
                st.metric("Status Kurasi", "✅ Siap Dilatih")
            ds_manifest_file = FINAL_DIR / "dataset_manifest.json"
            if ds_manifest_file.exists():
                try:
                    with open(ds_manifest_file, "r", encoding="utf-8") as mf:
                        dm = json.load(mf)
                    lbl_src = dm.get("label_source", "pseudo_labeled_llm")
                    leak_dict = dm.get("splits", {}).get("cross_split_leakage_check", {})
                    total_overlap = sum(
                        sum(inner.values()) for inner in leak_dict.values() if isinstance(inner, dict)
                    )
                    st.caption(
                        f"📋 Dataset Manifest: `{dm.get('dataset_version', '-')}` | "
                        f"Status Label: **`{lbl_src}`** | "
                        f"Cross-Split Overlap: `{total_overlap}`"
                    )
                except Exception:
                    pass
        else:
            with c1:
                st.metric("Data Latih", "Belum ada")
            with c2:
                st.metric("Data Validasi", "Belum ada")
            with c3:
                st.metric("Data Uji", "Belum ada")
            with c4:
                st.metric("Status Kurasi", "⚠️ Perlu Kurasi")

        # Action: Tombol Kurasi Ulang
        with st.expander("🔄 Kurasi Ulang Dataset dari Checkpoint Terbaru", expanded=not has_dataset):
            st.caption(
                "Menyaring error JSON (UNKNOWN, No JSON brackets), menormalkan string, "
                "menghapus spasi ganda, dan membagi dataset menjadi train (80%), val (10%), test (10%)."
            )
            if st.button("🧹 Jalankan Kurasi & Pembersihan Dataset Sekarang", type="secondary"):
                if not CHECKPOINT_FILE.exists():
                    st.error(f"Berkas checkpoint belum ditemukan di: `{CHECKPOINT_FILE}`")
                else:
                    with st.spinner("Sedang membersihkan dan membagi dataset..."):
                        out_csv = FINAL_DIR / "iicd_cleaned_indobertweet.csv"
                        curate_and_normalize(CHECKPOINT_FILE, out_csv, perform_split=True)
                        st.success("✅ Dataset berhasil dikurasi dan dipartisi!")
                        st.rerun()

        st.markdown("---")
        st.markdown("### 2. Konfigurasi Hyperparameter Pelatihan")

        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            base_model = st.selectbox(
                "Base Model HuggingFace:",
                [
                    "indolem/indobertweet-base-uncased",
                    "xlm-roberta-base",
                    "indobenchmark/indobert-base-p1"
                ],
                index=0,
                help="IndoBERTweet di-pretrain khusus pada 409M tweet Indonesia; XLM-RoBERTa sangat kuat untuk code-mixing dan bahasa daerah."
            )
            epochs = st.slider("Jumlah Epochs:", min_value=1, max_value=8, value=4, step=1)

        with col_h2:
            batch_size = st.selectbox("Batch Size:", [8, 16, 32], index=1)
            learning_rate = st.select_slider(
                "Learning Rate (AdamW):",
                options=[1e-5, 2e-5, 3e-5, 5e-5],
                value=2e-5
            )

        with col_h3:
            max_len = st.slider("Max Token Sequence:", min_value=64, max_value=256, value=128, step=32)
            # Hardware check
            import torch
            if torch.cuda.is_available():
                hw_info = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
            elif torch.backends.mps.is_available():
                hw_info = "Apple Silicon MPS (GPU Metal)"
            else:
                hw_info = "CPU Standar"
            st.info(f"⚡ **Akselerasi Terdeteksi**: `{hw_info}`")

        # Loss Function & Knowledge Distillation Options
        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            loss_choice = st.selectbox(
                "🎯 Fungsi Loss & Regularisasi:",
                [
                    "Focal Loss + Distilasi Confidence LLM (Rekomendasi - Reduksi False Positives)",
                    "Standard Cross-Entropy Loss"
                ],
                index=0,
                help="Focal Loss meredam loss pada sampel mudah dan memperbesar penalti pada sampel batas/sulit, dipadukan dengan soft labels berbobot confidence SumoPod LLM."
            )
        with col_opt2:
            st.caption("💡 **Optimalisasi**: Focal Loss + Normalisasi Teks terbukti menaikkan Macro-F1 dan menekan angka False Positives pada teks medsos Indonesia.")

        # Pilihan Arsitektur Model
        st.markdown("---")
        st.markdown("### 2b. Paradigma Arsitektur Model")
        arch_choice = st.radio(
            "Pilih Arsitektur yang Dilatih:",
            [
                "🛡️ Dual-Stream Hybrid Fusion (Transformer + SumoPod LLM Knowledge - Rekomendasi Disertasi)",
                "🔷 Single-Stream IndoBERTweet Baseline"
            ],
            index=0,
            help="Dual-Stream Hybrid Fusion mengintegrasikan representasi leksikal kontekstual IndoBERTweet dengan penalaran multi-dimensi kultural & emosi dari SumoPod LLM via Gated Cross-Modality Fusion."
        )

        # Tombol Mulai Training
        st.markdown("### 3. Eksekusi Pelatihan")
        btn_label = "🚀 Mulai Pelatihan Dual-Stream Hybrid Fusion" if "Dual-Stream" in arch_choice else "🚀 Mulai Pelatihan IndoBERTweet Baseline"
        start_training = st.button(btn_label, type="primary", use_container_width=True)

        if start_training:
            if not has_dataset:
                st.error("Dataset belum siap. Silakan klik tombol kurasi terlebih dahulu.")
                return

            pbar = st.progress(0.0)
            status_box = st.empty()

            if "Dual-Stream" in arch_choice:
                status_box.info("🧠 Menginisialisasi Dual-Stream Hybrid Fusion Model (Transformer + SumoPod LLM 33D)...")
                try:
                    train_vec = FINAL_DIR / "train_llm_vectors.npy"
                    val_vec = FINAL_DIR / "val_llm_vectors.npy"
                    test_vec = FINAL_DIR / "test_llm_vectors.npy"

                    if not train_vec.exists():
                        status_box.info("🔄 Menyiapkan matriks vektor probabilitas taksonomi LLM...")
                        curate_and_normalize(CHECKPOINT_FILE, FINAL_DIR / "iicd_cleaned_indobertweet.csv", perform_split=True)

                    status_box.markdown(f"**⚡ Melatih Hybrid Fusion Model pada {len(df_train)} sampel ({epochs} Epochs)...**")
                    pbar.progress(0.2)
                    
                    report_dict = train_hybrid_pipeline(
                        train_csv=train_path,
                        val_csv=val_path,
                        test_csv=test_path,
                        train_vec_path=train_vec,
                        val_vec_path=val_vec,
                        test_vec_path=test_vec,
                        output_dir=HYBRID_MODEL_DIR,
                        pretrained_model_name=base_model,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr_transformer=learning_rate,
                        lr_heads=1e-4,
                        baseline_model_dir=MODEL_DIR / "best_model",
                    )
                    evaluate_consistency_suite(
                        test_csv=test_path,
                        full_csv=FINAL_DIR / "iicd_cleaned_indobertweet.csv",
                        hybrid_weights=HYBRID_MODEL_DIR / "best_hybrid_model.pt",
                        output_dir=REPORT_DIR,
                    )
                    run_full_evaluation_pipeline(
                        model_dir=MODEL_DIR / "best_model",
                        test_csv=test_path,
                        output_dir=REPORT_DIR,
                        hybrid_dir=HYBRID_MODEL_DIR,
                    )
                    pbar.progress(1.0)
                    status_box.success("🎉 Pelatihan & Evaluasi Dual-Stream Hybrid Fusion Berhasil Selesai!")
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan selama pelatihan Hybrid Fusion: {str(e)}")
            else:
                status_box.info("🧠 Menjalankan fine-tuning IndoBERTweet Baseline (Train -> Val Threshold Tuning -> Test Eval)...")
                try:
                    pbar.progress(0.2)
                    train_baseline_model(
                        train_path=train_path,
                        val_path=val_path,
                        test_path=test_path,
                        output_dir=MODEL_DIR,
                        model_name=base_model,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        max_length=max_len,
                        seed=DEFAULT_RANDOM_SEED,
                        use_focal_loss=("Focal Loss" in loss_choice),
                    )
                    pbar.progress(0.8)
                    status_box.info("📊 Memperbarui laporan & tabel evaluasi disertasi...")
                    run_full_evaluation_pipeline(
                        model_dir=MODEL_DIR / "best_model",
                        test_csv=test_path,
                        output_dir=REPORT_DIR,
                        hybrid_dir=HYBRID_MODEL_DIR,
                    )
                    pbar.progress(1.0)
                    status_box.success("🎉 Pelatihan & Evaluasi IndoBERTweet Baseline Berhasil Selesai!")
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan selama pelatihan: {str(e)}")

    # =========================================================================
    # TAB 2: EVALUASI & TABEL LATEX BAB 4
    # =========================================================================
    with tab_eval:
        st.markdown("### 📊 Laporan Metrik Evaluasi & Kode LaTeX Disertasi")
        
        json_report = REPORT_DIR / "metrics_summary.json"
        tex_report = REPORT_DIR / "dissertation_tables.tex"
        
        if not json_report.exists():
            st.info("Laporan evaluasi belum digenerate. Silakan latih model terlebih dahulu atau klik tombol di bawah untuk membuat evaluasi dari checkpoint terlatih.")
            if st.button("📊 Generate Laporan Evaluasi Sekarang"):
                with st.spinner("Menghitung metrik evaluasi..."):
                    try:
                        run_full_evaluation_pipeline(output_dir=REPORT_DIR, hybrid_dir=HYBRID_MODEL_DIR)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal menghasilkan laporan evaluasi: {e}")
        else:
            col_t_title, col_t_btn = st.columns([3, 1])
            with col_t_btn:
                if st.button("🔄 Evaluasi Ulang Test Set", use_container_width=True):
                    with st.spinner("Mengevaluasi model pada test set..."):
                        try:
                            run_full_evaluation_pipeline(
                                model_dir=MODEL_DIR / "best_model",
                                test_csv=FINAL_DIR / "test.csv",
                                output_dir=REPORT_DIR,
                                hybrid_dir=HYBRID_MODEL_DIR,
                            )
                            st.success("Evaluasi berhasil diperbarui!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gagal memperbarui evaluasi: {e}")

            with open(json_report, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            target_src = metrics_data.get("target_source", "pseudo_labeled_llm")
            if target_src == "human_ground_truth":
                st.success("✅ **Status Label Uji: `human_ground_truth`** — Evaluasi dijalankan terhadap label referensi eksternal tervalidasi.")
            else:
                st.info(
                    "ℹ️ **Kebijakan Supervisi Riset: `pseudo_labeled_llm`** — Evaluasi mengukur kesepakatan & konsistensi model "
                    "terhadap **LLM-generated pseudo-labels** dengan **Stream A + Stream B (33D Bukti Kontekstual Non-Target) aktif penuh** "
                    "pada training, validation, testing, dan inference."
                )

            opt_th = float(metrics_data.get("threshold", metrics_data.get("optimal_threshold", 0.50)))
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Akurasi (Accuracy)", f"{metrics_data.get('accuracy', 0)*100:.2f}%")
            with m2:
                st.metric("Threshold Val (τ*)", f"{opt_th:.2f}")
            with m3:
                st.metric("Macro F1-Score", f"{metrics_data.get('macro_f1', 0):.4f}")
            with m4:
                roc_val = metrics_data.get("roc_auc")
                st.metric("ROC-AUC", f"{roc_val:.4f}" if roc_val is not None else "N/A")
            with m5:
                pr_val = metrics_data.get("pr_auc")
                st.metric("PR-AUC", f"{pr_val:.4f}" if pr_val is not None else "N/A")

            cal = metrics_data.get("calibration", {})
            ea = metrics_data.get("error_analysis", {})
            ci95 = metrics_data.get("confidence_intervals_95", {})
            c_sub1, c_sub2, c_sub3, c_sub4 = st.columns(4)
            with c_sub1:
                st.metric("Weighted F1", f"{metrics_data.get('weighted_f1', 0):.4f}")
            with c_sub2:
                ece_v = cal.get("ece")
                st.metric("Calibration ECE / MCE", f"{ece_v:.4f} / {cal.get('mce', 0):.4f}" if ece_v is not None else "N/A")
            with c_sub3:
                brier_v = cal.get("brier_score")
                st.metric("Brier / Log-Loss", f"{brier_v:.4f} / {cal.get('log_loss', 0):.4f}" if brier_v is not None else "N/A")
            with c_sub4:
                st.metric(
                    "FPR / FNR",
                    f"{ea.get('false_positive_rate', 0)*100:.1f}% / {ea.get('false_negative_rate', 0)*100:.1f}%",
                    help=f"FP={ea.get('false_positive_count', 0)}, FN={ea.get('false_negative_count', 0)}",
                )
            if ci95.get("macro_f1"):
                st.caption(f"📏 **Bootstrap 95% CI**: Accuracy={ci95.get('accuracy')}, Macro-F1={ci95.get('macro_f1')}, ROC-AUC={ci95.get('roc_auc')}")

            # 2. Confusion Matrix & Implicitness Visualizations
            st.markdown("---")
            st.markdown("### 📈 Visualisasi Publikasi Ilmiah")
            v_col1, v_col2 = st.columns(2)

            cm_dict = metrics_data.get("confusion_matrix", {})
            if "confusion_matrix_2x2" in metrics_data:
                cm_arr = np.array(metrics_data["confusion_matrix_2x2"])
            elif isinstance(cm_dict, dict):
                cm_arr = np.array([[cm_dict.get("tn", 0), cm_dict.get("fp", 0)], [cm_dict.get("fn", 0), cm_dict.get("tp", 0)]])
            elif isinstance(cm_dict, list):
                cm_arr = np.array(cm_dict)
            else:
                cm_arr = np.zeros((2, 2), dtype=int)
            base_fp = int(cm_arr[0][1])

            with v_col1:
                st.markdown("#### Matriks Konfusi (Confusion Matrix)")
                fig_cm = px.imshow(
                    cm_arr,
                    text_auto=True,
                    x=["Pred: Non-CB", "Pred: Cyberbullying"],
                    y=["Aktual: Non-CB", "Aktual: Cyberbullying"],
                    color_continuous_scale="Blues",
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                )
                fig_cm.update_layout(height=360)
                st.plotly_chart(fig_cm, use_container_width=True)

            with v_col2:
                st.markdown("#### Kinerja Empiris per Tingkat Keterbukaan (E0/E1/E2)")
                imp_bd = metrics_data.get("implicitness_breakdown", {})
                if imp_bd:
                    imp_rows = []
                    label_map = {
                        "E0": "E0 (Sangat Eksplisit)",
                        "E1": "E1 (Semi-Tersirat)",
                        "E2": "E2 (Tersirat / Kontekstual)",
                    }
                    for lvl, info in imp_bd.items():
                        imp_rows.append(
                            {
                                "Deskripsi": label_map.get(lvl, lvl),
                                "Akurasi (%)": round(float(info.get("accuracy", 0.0)) * 100, 2),
                                "Macro-F1": round(float(info.get("macro_f1", 0.0)), 4),
                                "Sampel": int(info.get("count", 0)),
                            }
                        )
                    fig_imp = px.bar(
                        pd.DataFrame(imp_rows),
                        x="Deskripsi",
                        y="Akurasi (%)",
                        text_auto=".1f",
                        hover_data=["Macro-F1", "Sampel"],
                        color="Deskripsi",
                        color_discrete_sequence=px.colors.qualitative.Prism,
                        title="Akurasi Baseline per Tingkat Keterbukaan Ujaran",
                    )
                    fig_imp.update_layout(height=360, showlegend=False)
                    st.plotly_chart(fig_imp, use_container_width=True)
                elif test_path.exists():
                    test_df = pd.read_csv(test_path)
                    imp_col = "target_implicit_level" if "target_implicit_level" in test_df.columns else "implicit_level"
                    imp_counts = test_df[imp_col].value_counts().reset_index()
                    imp_counts.columns = ["Tingkat", "Jumlah"]
                    fig_imp = px.bar(imp_counts, x="Tingkat", y="Jumlah", color="Tingkat", title="Distribusi Sampel Uji")
                    fig_imp.update_layout(height=360, showlegend=False)
                    st.plotly_chart(fig_imp, use_container_width=True)

            # 2b. Benchmark Komparatif Disertasi & Studi Ablasi 3-Arah
            hybrid_json = HYBRID_MODEL_DIR / "hybrid_test_metrics.json"
            if hybrid_json.exists():
                st.markdown("---")
                st.markdown("### 🏛️ Komparasi Empiris & Studi Ablasi Bab 4 (Test Set Identik)")
                st.caption("Perbandingan kuantitatif Single-Stream IndoBERTweet vs Dual-Stream Hybrid Fusion serta Model Manifest.")

                try:
                    with open(hybrid_json, "r", encoding="utf-8") as f:
                        hyb_data = json.load(f)
                    hyb_test = hyb_data.get("test_metrics", {})
                    hyb_cm = hyb_test.get("confusion_matrix", {})
                    hyb_mtl = hyb_data.get("multi_task_heads_test", {})
                    mean_z = float(hyb_data.get("mean_gating_z", 0.0))
                    type_acc = float(hyb_mtl.get("type", {}).get("accuracy", 0.0)) * 100
                    emo_acc = float(hyb_mtl.get("emotion", {}).get("accuracy", 0.0)) * 100
                    hyb_fp = int(hyb_cm.get("fp", 0))

                    bench_records = [
                        {
                            "Arsitektur Model": "🔷 IndoBERTweet Baseline (Single-Stream)",
                            "Akurasi": f"{metrics_data.get('accuracy', 0)*100:.2f}%",
                            "Macro F1": f"{metrics_data.get('macro_f1', 0):.4f}",
                            "ROC-AUC": f"{metrics_data.get('roc_auc', 0):.4f}" if metrics_data.get("roc_auc") is not None else "N/A",
                            "PR-AUC": f"{metrics_data.get('pr_auc', 0):.4f}" if metrics_data.get("pr_auc") is not None else "N/A",
                            "False Positives (FP)": f"{base_fp} sampel",
                            "Multi-Task Output": "Hanya Binary Cyberbullying",
                            "Gating Weight (z)": "1.000 (100% Leksikal)",
                        },
                        {
                            "Arsitektur Model": "🛡️ Dual-Stream Hybrid Fusion (6-Head MTL)",
                            "Akurasi": f"{hyb_test.get('accuracy', 0)*100:.2f}%",
                            "Macro F1": f"{hyb_test.get('macro_f1', 0):.4f}",
                            "ROC-AUC": f"{hyb_test.get('roc_auc', 0):.4f}" if hyb_test.get("roc_auc") is not None else "N/A",
                            "PR-AUC": f"{hyb_test.get('pr_auc', 0):.4f}" if hyb_test.get("pr_auc") is not None else "N/A",
                            "False Positives (FP)": f"{hyb_fp} sampel",
                            "Multi-Task Output": f"6 Dimensi (Tipe: {type_acc:.1f}%, Emosi: {emo_acc:.1f}%)",
                            "Gating Weight (z)": f"{mean_z:.4f} ({hyb_data.get('evaluation_mode', 'hybrid')})",
                        },
                    ]
                    st.dataframe(pd.DataFrame(bench_records), use_container_width=True, hide_index=True)

                    ablation_data = hyb_data.get("ablation_comparison")
                    if not ablation_data and (HYBRID_MODEL_DIR / "ablation_comparison.json").exists():
                        with open(HYBRID_MODEL_DIR / "ablation_comparison.json", "r", encoding="utf-8") as af:
                            ablation_data = json.load(af)
                    if ablation_data:
                        ab_keys = [
                            "E1_transformer_only",
                            "E2_llm_knowledge_only",
                            "E3_hybrid_full_fusion",
                            "E4_hybrid_no_cultural",
                            "E5_hybrid_no_emotional",
                            "E6_hybrid_no_gated_fusion",
                        ]
                        if not any(ablation_data.get(k) for k in ab_keys):
                            ab_keys = ["model_a_indobertweet_only", "model_b_llm_only", "model_c_hybrid_fusion"]

                        ab_rows = []
                        for k in ab_keys:
                            m = ablation_data.get(k)
                            if not m:
                                continue
                            gz = m.get("mean_gating_z")
                            ab_rows.append(
                                {
                                    "Eksperimen Ablasi": m.get("model_name", k),
                                    "Status": m.get("status", "evaluated"),
                                    "Accuracy": f"{m['accuracy']*100:.2f}%" if m.get("accuracy") is not None else "N/A",
                                    "Macro-F1": f"{m['macro_f1']:.4f}" if m.get("macro_f1") is not None else "N/A",
                                    "ROC-AUC": f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "N/A",
                                    "PR-AUC": f"{m['pr_auc']:.4f}" if m.get("pr_auc") is not None else "N/A",
                                    "Gate z": f"{gz:.4f}" if isinstance(gz, (int, float)) else "-",
                                }
                            )
                        st.markdown("#### 🧪 Tabel Studi Ablasi 6-Konfigurasi E1–E6 (`ablation_comparison.json`)")
                        st.dataframe(pd.DataFrame(ab_rows), use_container_width=True, hide_index=True)

                    cons_json = REPORT_DIR / "consistency_evaluation.json"
                    if cons_json.exists():
                        with open(cons_json, "r", encoding="utf-8") as cf:
                            cons_data = json.load(cf)
                        ts_ag = cons_data.get("teacher_student_agreement", {})
                        if ts_ag.get("available"):
                            with st.expander("🤝 Reliabilitas Non-GT & Kesepakatan Guru-Murid (LLM vs Hybrid 6-Head)"):
                                st.caption(cons_data.get("disclaimer", ""))
                                head_rows = [
                                    {"Classification Head": hk, **hv}
                                    for hk, hv in ts_ag.get("head_agreements", {}).items()
                                ]
                                st.dataframe(pd.DataFrame(head_rows), use_container_width=True, hide_index=True)
                                cue_stats = cons_data.get("stream_b_cue_family_stats", {})
                                if cue_stats:
                                    st.markdown("**Aktivasi 5 Famili Bukti Kontekstual Stream B (Test Split):**")
                                    st.dataframe(
                                        pd.DataFrame([{"Famili Bukti": fk, **fv} for fk, fv in cue_stats.items()]),
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                    hyb_manifest = load_model_manifest(HYBRID_MODEL_DIR)
                    if hyb_manifest:
                        with st.expander("📋 Model Manifest & Reproduksibilitas Hybrid Fusion"):
                            st.json(hyb_manifest)
                except Exception as e:
                    st.caption(f"Info metrik hybrid: {e}")

            # 2c. Evaluasi Empiris Dual Explainability (Real SHAP Library vs Diagnostik Leksikal CoT)
            dual_xai_json = REPORT_DIR / "dual_xai_evaluation.json"
            dual_xai_tex = REPORT_DIR / "table_dual_xai_breakdown.tex"

            if dual_xai_json.exists():
                st.markdown("---")
                col_dx_t, col_dx_b = st.columns([3, 1])
                with col_dx_t:
                    st.markdown("### 🔬 Evaluasi Empiris Dual-XAI (SHAP Library ⟷ Diagnostik CoT LLM)")
                    st.caption(
                        "Mengukur kesetiaan kausal SHAP (Comprehensiveness) pada model target dan diagnostik tumpang tindih leksikal dengan CoT LLM."
                    )
                with col_dx_b:
                    if st.button("🔄 Evaluasi Ulang Dual-XAI (30 Sampel)", use_container_width=True):
                        with st.spinner("Menghitung atribusi SHAP & faithfulness pada test set..."):
                            try:
                                hyb_w = HYBRID_MODEL_DIR / "best_hybrid_model.pt"
                                chosen_type = "hybrid" if hyb_w.exists() else "baseline"
                                chosen_path = hyb_w if hyb_w.exists() else (MODEL_DIR / "best_model")
                                evaluate_dual_xai_benchmark(
                                    test_csv_path=FINAL_DIR / "test.csv",
                                    model_path=chosen_path,
                                    model_type=chosen_type,
                                    output_dir=REPORT_DIR,
                                    max_samples=30,
                                )
                                st.success("Evaluasi Dual-XAI berhasil diperbarui!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Gagal menjalankan Dual-XAI: {e}")

                try:
                    with open(dual_xai_json, "r", encoding="utf-8") as f:
                        dx_data = json.load(f)

                    faith = dx_data.get("scientific_faithfulness_metrics", {})
                    diag = dx_data.get("diagnostic_lexical_heuristics", {})
                    dx_impl = dx_data.get("implicitness_breakdown", {})

                    dx1, dx2, dx3, dx4 = st.columns(4)
                    with dx1:
                        comp_m = faith.get("mean_comprehensiveness")
                        st.metric("Faithfulness Comprehensiveness (Δp)", f"{comp_m:.4f}" if comp_m is not None else "N/A")
                    with dx2:
                        st.metric("Diagnostik Lexical Overlap", f"{diag.get('mean_lexical_alignment_score', 0):.2f}%")
                    with dx3:
                        st.metric("Konsistensi Leksikal (≥ 50%)", f"{diag.get('lexical_alignment_rate_percentage', 0):.2f}%")
                    with dx4:
                        st.metric("Model & Kondisi Vektor", f"{dx_data.get('model_type', 'hybrid')} | {dx_data.get('llm_vector_conditioning', 'n/a')}")

                    impl_rows = []
                    lvl_labels = {"E0": "E0 (Sangat Eksplisit)", "E1": "E1 (Semi-Tersirat)", "E2": "E2 (Tersirat)"}
                    for lvl in ["E0", "E1", "E2"]:
                        item = dx_impl.get(lvl, {})
                        impl_rows.append(
                            {
                                "Tingkat": lvl_labels[lvl],
                                "Comprehensiveness (Δp)": item.get("mean_comprehensiveness") or 0.0,
                                "Diagnostik Lexical Overlap (%)": item.get("mean_diagnostic_lexical_alignment", 0.0),
                                "Sampel": item.get("count", 0),
                            }
                        )
                    fig_dx = px.bar(
                        pd.DataFrame(impl_rows),
                        x="Tingkat",
                        y="Diagnostik Lexical Overlap (%)",
                        color="Tingkat",
                        text_auto=".1f",
                        hover_data=["Comprehensiveness (Δp)", "Sampel"],
                        color_discrete_sequence=["#3b82f6", "#8b5cf6", "#ec4899"],
                        title="Diagnostik Lexical Overlap & Faithfulness per Tingkat Keterbukaan",
                    )
                    fig_dx.update_layout(height=340, showlegend=False)
                    st.plotly_chart(fig_dx, use_container_width=True)

                    global_imp = dx_data.get("global_feature_importance", {})
                    top_tokens = global_imp.get("top_global_tokens", global_imp.get("top_tokens", []))[:15]
                    if top_tokens:
                        st.markdown("#### 🌐 Global SHAP Feature Importance (Top Token Leksikal)")
                        fig_gi = px.bar(
                            pd.DataFrame(top_tokens),
                            x="mean_abs_shap",
                            y="token",
                            orientation="h",
                            hover_data=["occurrences"] if "occurrences" in top_tokens[0] else None,
                            title="Top Token berdasarkan Mean |SHAP Value|",
                        )
                        fig_gi.update_layout(height=360, yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig_gi, use_container_width=True)

                    st.info(
                        "ℹ️ **Catatan Metodologis Explainability**: *Comprehensiveness (Δp)* mengukur kesetiaan kausal SHAP "
                        "(penurunan probabilitas prediksi ketika token ber-SHAP tertinggi dihilangkan), sedangkan *Lexical Overlap* "
                        "merupakan metrik diagnostik heuristik dan tidak diklaim sebagai bukti validitas kausal penuh."
                    )
                except Exception as e:
                    st.caption(f"Info visualisasi Dual-XAI: {e}")

            # 3. LaTeX Tables Generator
            st.markdown("---")
            st.markdown("### 📝 Kode LaTeX Bab 4 (Hasil & Pembahasan)")
            st.caption("Salin kode LaTeX berikut langsung ke dalam dokumen Overleaf/naskah disertasi Anda:")

            tab_tex1, tab_tex2 = st.tabs(["📄 Tabel Metrik Klasifikasi", "🔬 Tabel Konsistensi Dual-XAI"])
            with tab_tex1:
                if tex_report.exists():
                    with open(tex_report, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="latex")
                else:
                    st.info("Kode LaTeX klasifikasi sedang dipersiapkan...")

            with tab_tex2:
                if dual_xai_tex.exists():
                    with open(dual_xai_tex, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="latex")
                else:
                    st.info("Kode LaTeX Dual-XAI sedang dipersiapkan...")

    # =========================================================================
    # TAB 3: PLAYGROUND INFERENSI LIVE (HYBRID & SINGLE TRANSFORMER)
    # =========================================================================
    with tab_test:
        st.markdown("### 🧪 Uji Coba Cepat (Live Inference Playground)")
        st.markdown("Uji inferensi kalimat secara instan menggunakan arsitektur pilihan Anda.")

        best_single_path = MODEL_DIR / "best_model"
        best_hybrid_path = HYBRID_MODEL_DIR / "best_hybrid_model.pt"

        single_ready = best_single_path.exists() and (
            (best_single_path / "model.safetensors").exists() or (best_single_path / "pytorch_model.bin").exists()
        )
        hybrid_ready = best_hybrid_path.exists()

        if not single_ready and not hybrid_ready:
            st.warning("⚠️ Model belum selesai dilatih. Silakan latih model pada Tab 1 terlebih dahulu.")
            st.text_input("Contoh Komentar:", value="makannya segentong buset mirip ursula")
            st.button("Prediksi", disabled=True)
        else:
            # Pilihan Model Inferensi
            model_play_choice = st.radio(
                "Pilih Model untuk Inferensi:",
                [
                    "🛡️ Dual-Stream Hybrid Fusion Model (Transformer + SumoPod LLM - 6 Multi-Task Heads)",
                    "🔷 Single-Stream IndoBERTweet Baseline",
                ],
                index=0 if hybrid_ready else 1,
                horizontal=True,
            )

            # Status Backend LLM & muat threshold optimal dari Model Manifest
            if "Dual-Stream" in model_play_choice:
                active_eng = st.session_state.get("active_engine")
                is_mock_eng = getattr(active_eng, "is_mock", True) if active_eng else True
                if is_mock_eng:
                    st.warning("ℹ️ **Backend LLM Stream: Mode Simulasi (Mock Fallback)**. Hasil Stream B berasal dari heuristik simulasi, bukan API SumoPod LLM.")
                else:
                    st.caption(f"🌐 **Backend LLM Stream: SumoPod AI Gateway** (`{getattr(active_eng, 'model_name', DEFAULT_SUMOPOD_MODEL)}` aktif).")

                hyb_manifest = load_model_manifest(HYBRID_MODEL_DIR)
                if hyb_manifest and hyb_manifest.get("optimal_threshold") is not None:
                    opt_th_default = float(hyb_manifest["optimal_threshold"])
                elif (HYBRID_MODEL_DIR / "hybrid_test_metrics.json").exists():
                    try:
                        with open(HYBRID_MODEL_DIR / "hybrid_test_metrics.json", "r", encoding="utf-8") as hf:
                            opt_th_default = float(json.load(hf).get("optimal_threshold", 0.50))
                    except Exception:
                        opt_th_default = 0.50
                else:
                    opt_th_default = 0.50
            else:
                opt_th_default = get_saved_threshold(best_single_path, default=0.50)

            c_inp1, c_inp2 = st.columns([3, 1])
            with c_inp1:
                test_input = st.text_input(
                    "Masukkan komentar media sosial untuk diuji:",
                    value="makannya segentong buset mirip ursula di little mermaid",
                    placeholder="Ketik komentar di sini...",
                )
            with c_inp2:
                th_choice = st.slider(
                    "Ambang Batas (Threshold):",
                    min_value=0.20,
                    max_value=0.80,
                    value=float(min(0.80, max(0.20, opt_th_default))),
                    step=0.02,
                    help=f"Threshold optimal Validation Set dari model_manifest.json: {opt_th_default:.2f}",
                )

            btn_pred_text = (
                "🔍 Jalankan Inferensi Hybrid Multi-Task"
                if "Dual-Stream" in model_play_choice
                else "🔍 Klasifikasikan dengan IndoBERTweet Baseline"
            )

            if st.button(btn_pred_text, type="primary"):
                if not test_input.strip():
                    st.warning("Silakan masukkan teks terlebih dahulu.")
                elif "Dual-Stream" in model_play_choice:
                    if not hybrid_ready:
                        st.error("Model Hybrid Fusion belum dilatih. Silakan pilih 'Dual-Stream Hybrid Fusion' di Tab 1 dan jalankan pelatihan.")
                    else:
                        with st.spinner("Menjalankan inferensi Dual-Stream Hybrid Fusion & Dual-Modality SHAP..."):
                            try:
                                engine = st.session_state.get("active_engine")
                                temp = st.session_state.get("temperature", 0.1)
                                max_tok = st.session_state.get("max_tokens", 160)
                                sys_prompt = st.session_state.get("custom_system_prompt", None)

                                llm_anno = run_single_annotation(
                                    engine=engine,
                                    text=test_input.strip(),
                                    system_prompt=sys_prompt,
                                    max_new_tokens=max_tok,
                                    temperature=temp,
                                )

                                hyb_model, hyb_tok, hyb_dev, hyb_manifest = load_hybrid_classifier_with_manifest(best_hybrid_path)
                                h_res = predict_hybrid_text(
                                    model=hyb_model,
                                    tokenizer=hyb_tok,
                                    device=hyb_dev,
                                    text=test_input.strip(),
                                    annotation_dict=llm_anno,
                                    threshold=th_choice,
                                    manifest=hyb_manifest,
                                )

                                is_cb = h_res["cyberbullying"] == "YES"
                                badge_color = "#e53e3e" if is_cb else "#38a169"
                                verdict_title = "🚨 TERDETEKSI CYBERBULLYING (HYBRID FUSION)" if is_cb else "✅ NON-CYBERBULLYING (NETRAL)"
                                safe_text_html = sanitize_html_text(h_res["text"])

                                st.markdown(
                                    f"""
                                    <div style="background-color: {'#fff5f5' if is_cb else '#f0fff4'}; border: 2px solid {badge_color}; border-radius: 12px; padding: 20px; margin: 15px 0;">
                                        <h3 style="color: {badge_color}; margin: 0 0 8px 0;">{verdict_title}</h3>
                                        <p style="font-size: 16px; margin: 0; color: #2d3748;"><strong>Teks:</strong> "{safe_text_html}"</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                c1, c2, c3, c4 = st.columns(4)
                                with c1:
                                    st.metric("Probabilitas Cyberbullying", f"{h_res['prob_cyberbullying']*100:.2f}%")
                                with c2:
                                    z_val = h_res["gate_weight_z"]
                                    st.metric(
                                        "Bobot Gerbang Fusi (z)",
                                        f"{z_val:.3f}",
                                        help=f"{z_val*100:.1f}% Atensi Leksikal IndoBERTweet, {(1-z_val)*100:.1f}% Penalaran LLM",
                                    )
                                with c3:
                                    st.metric("Threshold Digunakan", f"{h_res.get('threshold', th_choice):.2f}")
                                with c4:
                                    st.metric("Latency", f"{h_res['latency_ms']} ms")

                                st.markdown("#### 🎯 Hasil Prediksi 6-Head Multi-Task Taksonomi Disertasi")
                                m1, m2, m3, m4, m5 = st.columns(5)
                                with m1:
                                    st.metric("Tipe (Type)", h_res["type"])
                                with m2:
                                    st.metric("Emosi (Emotion)", h_res["emotion"])
                                with m3:
                                    st.metric("Konteks Budaya", h_res["cultural_context"])
                                with m4:
                                    st.metric("Keterbukaan", h_res["implicit_level"])
                                with m5:
                                    st.metric("Tingkat Keparahan", h_res["severity"])

                                st.markdown("---")
                                st.markdown("#### 🔬 Dual-Modality Explainability Inspector (Real `shap` Library)")
                                explainer = HybridSHAPExplainer(
                                    model_type="hybrid",
                                    loaded_model=hyb_model,
                                    loaded_tokenizer=hyb_tok,
                                    device=hyb_dev,
                                    manifest=hyb_manifest,
                                )
                                dual_shap = explainer.explain_dual_modalities(
                                    text=test_input.strip(),
                                    annotation_dict=llm_anno,
                                    max_evals_text=90,
                                    max_evals_vec=120,
                                )
                                shap_res = dual_shap["textual_shap"]
                                vec_shap = dual_shap["llm_vector_shap"]
                                llm_vec_used = extract_contextual_evidence_vector(llm_anno, text=test_input.strip()) if llm_anno else None
                                comp_score = compute_faithfulness_comprehensiveness(
                                    explainer, test_input.strip(), shap_res, llm_vector=llm_vec_used
                                )

                                col_sh, col_cot = st.columns(2)
                                with col_sh:
                                    st.markdown("##### 📊 Stream A: Atribusi Token Leksikal (SHAP)")
                                    st.markdown(shap_res["html_heatmap"], unsafe_allow_html=True)
                                    pos_t = [f"`{t['token']}` (+{t['shap_value']:.3f})" for t in shap_res.get("top_positive_tokens", [])]
                                    neg_t = [f"`{t['token']}` ({t['shap_value']:.3f})" for t in shap_res.get("top_negative_tokens", [])]
                                    if pos_t:
                                        st.markdown(f"🔴 **Kontribusi Positif (+)**: {', '.join(pos_t)}")
                                    if neg_t:
                                        st.markdown(f"🔵 **Kontribusi Negatif (-)**: {', '.join(neg_t)}")
                                    if comp_score is not None:
                                        st.caption(f"🎯 Faithfulness Comprehensiveness (Δp setelah ablasi top token): **{comp_score:.4f}**")

                                with col_cot:
                                    st.markdown("##### 💬 Stream B & CoT Rationale (SumoPod LLM)")
                                    safe_expl_html = sanitize_html_text(llm_anno.get("explanation", ""))
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 14px 18px; border-radius: 6px; min-height: 110px;">
                                            <p style="font-size: 15px; line-height: 1.6; color: #1e293b; margin: 0;">
                                                {safe_expl_html}
                                            </p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    top_feats = (vec_shap.get("top_positive_features", []) + vec_shap.get("top_negative_features", []))[:8]
                                    if top_feats:
                                        st.markdown("**Kontribusi Fitur Pengetahuan LLM 33D (SHAP Numerik Terpisah):**")
                                        st.dataframe(pd.DataFrame(top_feats), use_container_width=True, hide_index=True)

                                align_info = align_shap_with_cot(shap_res, llm_anno.get("explanation", ""), comprehensiveness_score=comp_score)
                                score = align_info["alignment_score"]
                                safe_align_sum = sanitize_html_text(align_info["alignment_summary"])
                                safe_disc = sanitize_html_text(
                                    "Metrik lexical overlap bersifat diagnostik heuristik, bukan ukuran validitas kausal tervalidasi."
                                )
                                st.markdown(
                                    f"""
                                    <div style="margin-top: 14px; background-color: #f0fdf4; border: 1px solid #22c55e; border-radius: 8px; padding: 12px 18px;">
                                        <span style="font-weight: 600; color: #15803d;">🛡️ Diagnostik Tumpang Tindih Leksikal Dual-XAI: {score}%</span>
                                        <p style="margin: 4px 0 0 0; font-size: 13px; color: #15803d;">
                                            {safe_align_sum}
                                        </p>
                                        <p style="margin: 4px 0 0 0; font-size: 12px; color: #475569;">
                                            <em>Catatan: {safe_disc}</em>
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            except Exception as e:
                                st.error(f"Gagal inferensi Hybrid Fusion: {str(e)}")
                else:
                    if not single_ready:
                        st.error("Model IndoBERTweet Baseline belum ditemukan. Silakan latih model baseline di Tab 1.")
                    else:
                        with st.spinner("Melakukan inferensi Transformer Baseline..."):
                            try:
                                model, tokenizer, dev = load_classifier(best_single_path)
                                res = predict_text(model, tokenizer, dev, test_input, threshold=th_choice)

                                is_cb = res["label"] == 1
                                badge_color = "#e53e3e" if is_cb else "#38a169"
                                verdict_title = "🚨 TERDETEKSI CYBERBULLYING" if is_cb else "✅ NON-CYBERBULLYING (NETRAL)"
                                safe_base_html = sanitize_html_text(res["text"])

                                st.markdown(
                                    f"""
                                    <div style="background-color: {'#fff5f5' if is_cb else '#f0fff4'}; border: 2px solid {badge_color}; border-radius: 12px; padding: 20px; margin: 15px 0;">
                                        <h3 style="color: {badge_color}; margin: 0 0 8px 0;">{verdict_title}</h3>
                                        <p style="font-size: 16px; margin: 0; color: #2d3748;"><strong>Teks:</strong> "{safe_base_html}"</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                                with col_res1:
                                    st.metric("Tingkat Keyakinan (Confidence)", f"{res['confidence']*100:.2f}%")
                                with col_res2:
                                    st.metric("Probabilitas Cyberbullying", f"{res['prob_cb']*100:.2f}%")
                                with col_res3:
                                    st.metric("Threshold Digunakan", f"{res.get('threshold', th_choice):.2f}")
                                with col_res4:
                                    st.metric("Kecepatan Inferensi", f"{res['latency_ms']} ms")

                            except Exception as e:
                                st.error(f"Gagal memuat model: {str(e)}")
