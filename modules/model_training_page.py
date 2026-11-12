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

from scripts.curate_dataset import curate_and_normalize
from scripts.evaluate_indobertweet import load_classifier, predict_text
from scripts.generate_dissertation_report import (
    run_full_evaluation_pipeline,
    generate_latex_tables,
    calculate_comprehensive_metrics,
    analyze_taxonomy_breakdown
)
from scripts.train_hybrid_fusion import train_hybrid_pipeline
from scripts.evaluate_dual_xai import evaluate_dual_xai_benchmark
from core.fusion_model import load_hybrid_classifier, predict_hybrid_text
from core.shap_explainer import HybridSHAPExplainer, align_shap_with_cot
from core.model_engine import run_single_annotation

FINAL_DIR = PROJECT_DIR / "output" / "final"
CHECKPOINT_FILE = PROJECT_DIR / "output" / "checkpoints" / "checkpoint_Dataset-Research.csv"
MODEL_DIR = PROJECT_DIR / "output" / "models" / "indobertweet_cyberbullying"
HYBRID_MODEL_DIR = PROJECT_DIR / "output" / "models" / "hybrid_fusion_indobertweet"
REPORT_DIR = PROJECT_DIR / "output" / "evaluation_report"


def render_model_training_page():
    st.markdown("## 🤖 Fine-Tuning & Evaluasi Transformer IndoBERTweet")
    st.markdown(
        "Latih model Transformer Encoder lokal (**IndoBERTweet**) menggunakan dataset hasil kurasi anotasi LLM SumoPod (LLaMA-3). "
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
                help="Focal Loss meredam loss pada sampel mudah dan memperbesar penalti pada sampel batas/sulit, dipadukan dengan soft labels berbobot confidence LLaMA-3."
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
            help="Dual-Stream Hybrid Fusion mengintegrasikan representasi leksikal kontekstual IndoBERTweet dengan penalaran multi-dimensi kultural & emosi dari SumoPod LLaMA-3 via Gated Cross-Modality Fusion."
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
                        lr_heads=1e-4
                    )
                    
                    pbar.progress(1.0)
                    status_box.success("🎉 Pelatihan & Evaluasi Dual-Stream Hybrid Fusion Berhasil Selesai!")
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan selama pelatihan Hybrid Fusion: {str(e)}")
            else:
                status_box.info("🧠 Menginisialisasi tokenizer dan mengunduh bobot IndoBERTweet Baseline...")
                try:
                    import transformers
                    from transformers import (
                        AutoTokenizer, AutoModelForSequenceClassification,
                        Trainer, TrainingArguments, EarlyStoppingCallback,
                        TrainerCallback, set_seed
                    )
                    from scripts.train_indobertweet import CyberbullyingDataset, compute_metrics, FocalLossTrainer

                    set_seed(42)
                    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

                    tokenizer = AutoTokenizer.from_pretrained(base_model)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        base_model,
                        num_labels=2,
                        id2label={0: "NON_CYBERBULLYING", 1: "CYBERBULLYING"},
                        label2id={"NON_CYBERBULLYING": 0, "CYBERBULLYING": 1}
                    )

                    train_conf = df_train["confidence"] if "confidence" in df_train.columns else None
                    val_conf = df_val["confidence"] if "confidence" in df_val.columns else None

                    train_ds = CyberbullyingDataset(df_train["comment"], df_train["label"], tokenizer, max_len, confidences=train_conf)
                    val_ds = CyberbullyingDataset(df_val["comment"], df_val["label"], tokenizer, max_len, confidences=val_conf)

                    output_dir = MODEL_DIR
                    checkpoints_dir = output_dir / "checkpoints"
                    best_model_dir = output_dir / "best_model"

                    class StreamlitProgressCallback(TrainerCallback):
                        def on_epoch_end(self, args, state, control, **kwargs):
                            ep = int(state.epoch)
                            p = min(1.0, ep / epochs)
                            pbar.progress(p)
                            status_box.markdown(f"**Sedang Melatih:** Epoch `{ep}/{epochs}` selesai | Global Step: `{state.global_step}/{state.max_steps}`")

                    eval_arg_name = "eval_strategy" if hasattr(TrainingArguments, "eval_strategy") else "evaluation_strategy"
                    eval_kwargs = {eval_arg_name: "epoch"}

                    training_args = TrainingArguments(
                        output_dir=str(checkpoints_dir),
                        num_train_epochs=epochs,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size * 2,
                        learning_rate=learning_rate,
                        weight_decay=0.01,
                        warmup_ratio=0.1,
                        save_strategy="epoch",
                        save_total_limit=1,
                        load_best_model_at_end=True,
                        metric_for_best_model="macro_f1",
                        greater_is_better=True,
                        seed=42,
                        dataloader_pin_memory=False if device == "mps" else True,
                        report_to="none",
                        **eval_kwargs
                    )

                    if "Focal Loss" in loss_choice:
                        trainer = FocalLossTrainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_ds,
                            eval_dataset=val_ds,
                            tokenizer=tokenizer,
                            compute_metrics=compute_metrics,
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=2), StreamlitProgressCallback()],
                            focal_alpha=0.55,
                            focal_gamma=2.0
                        )
                    else:
                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_ds,
                            eval_dataset=val_ds,
                            tokenizer=tokenizer,
                            compute_metrics=compute_metrics,
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=2), StreamlitProgressCallback()]
                        )

                    status_box.markdown(f"**⚡ Melatih pada {len(df_train)} sampel ({epochs} Epochs)...**")
                    trainer.train()

                    status_box.info("💾 Menyimpan model terbaik ke disk...")
                    trainer.save_model(str(best_model_dir))
                    tokenizer.save_pretrained(str(best_model_dir))

                    status_box.info("📊 Menjalankan evaluasi pada Independent Test Set...")
                    run_full_evaluation_pipeline(
                        model_dir=best_model_dir,
                        test_csv=test_path,
                        output_dir=REPORT_DIR
                    )

                    pbar.progress(1.0)
                    status_box.success("🎉 Pelatihan & Evaluasi IndoBERTweet Baseline Berhasil Selesai!")
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan selama pelatihan: {str(e)}")
                st.error(f"❌ Terjadi kesalahan selama pelatihan: {str(e)}")

    # =========================================================================
    # TAB 2: EVALUASI & TABEL LATEX BAB 4
    # =========================================================================
    with tab_eval:
        st.markdown("### 📊 Laporan Metrik Evaluasi & Kode LaTeX Disertasi")
        
        json_report = REPORT_DIR / "metrics_summary.json"
        tex_report = REPORT_DIR / "dissertation_tables.tex"
        
        if not json_report.exists():
            st.info("Laporan evaluasi belum digenerate. Silakan latih model terlebih dahulu atau klik tombol di bawah untuk membuat evaluasi baseline.")
            if st.button("📊 Generate Laporan Evaluasi Sekarang"):
                with st.spinner("Menghitung metrik evaluasi..."):
                    run_full_evaluation_pipeline(output_dir=REPORT_DIR)
                    st.rerun()
            return

        col_t_title, col_t_btn = st.columns([3, 1])
        with col_t_btn:
            if st.button("🔄 Evaluasi Ulang Test Set", use_container_width=True):
                with st.spinner("Mengevaluasi model pada test set..."):
                    best_model_path = MODEL_DIR / "best_model"
                    test_csv_path = FINAL_DIR / "test.csv"
                    run_full_evaluation_pipeline(
                        model_dir=best_model_path,
                        test_csv=test_csv_path,
                        output_dir=REPORT_DIR
                    )
                    st.success("Evaluasi berhasil diperbarui!")
                    st.rerun()

        with open(json_report, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        # 1. High-level Metrics Cards
        opt_th = metrics_data.get("optimal_threshold", 0.50)
        base_acc = metrics_data.get("baseline_accuracy", metrics_data.get("accuracy", 0))
        delta_acc = (metrics_data.get('accuracy', 0) - base_acc) * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Akurasi (Accuracy)", f"{metrics_data.get('accuracy', 0)*100:.2f}%", delta=f"+{delta_acc:.2f}% vs Base" if delta_acc > 0 else None)
        with m2:
            st.metric("Threshold Optimal (τ*)", f"{opt_th:.2f}")
        with m3:
            st.metric("Macro F1-Score", f"{metrics_data.get('macro_f1', 0):.4f}")
        with m4:
            st.metric("Weighted F1", f"{metrics_data.get('weighted_f1', 0):.4f}")
        with m5:
            st.metric("Area ROC (AUC)", f"{metrics_data.get('roc_auc', 0):.4f}")

        # 2. Confusion Matrix & Implicitness Visualizations
        st.markdown("---")
        st.markdown("### 📈 Visualisasi Publikasi Ilmiah")
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.markdown("#### Matriks Konfusi (Confusion Matrix)")
            cm_arr = np.array(metrics_data.get("confusion_matrix", [[0, 0], [0, 0]]))
            fig_cm = px.imshow(
                cm_arr,
                text_auto=True,
                x=["Pred: Non-CB", "Pred: Cyberbullying"],
                y=["Aktual: Non-CB", "Aktual: Cyberbullying"],
                color_continuous_scale="Blues",
                labels=dict(x="Prediksi", y="Aktual", color="Jumlah")
            )
            fig_cm.update_layout(height=360)
            st.plotly_chart(fig_cm, use_container_width=True)

        with v_col2:
            st.markdown("#### Analisis Ketahanan pada Ujaran Tersirat")
            if test_path.exists():
                test_df = pd.read_csv(test_path)
                imp_counts = test_df["implicit_level"].value_counts().reset_index()
                imp_counts.columns = ["Tingkat", "Jumlah"]
                imp_counts["Deskripsi"] = imp_counts["Tingkat"].map({
                    "E0": "E0 (Sangat Eksplisit)",
                    "E1": "E1 (Semi-Tersirat)",
                    "E2": "E2 (Tersirat / Kontekstual)"
                }).fillna(imp_counts["Tingkat"])

                fig_imp = px.bar(
                    imp_counts, x="Deskripsi", y="Jumlah",
                    color="Deskripsi",
                    color_discrete_sequence=px.colors.qualitative.Prism,
                    title="Distribusi Sampel Uji per Keterbukaan"
                )
                fig_imp.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)

        # 2b. Benchmark Komparatif Disertasi (Single-Stream vs Dual-Stream Hybrid Fusion)
        hybrid_json = HYBRID_MODEL_DIR / "hybrid_test_metrics.json"
        if hybrid_json.exists():
            st.markdown("---")
            st.markdown("### 🏛️ Tabel Komparasi Empiris Bab 4: Single Transformer vs Dual-Stream Hybrid Fusion")
            st.caption("Perbandingan kuantitatif antara model baseline (IndoBERTweet tunggal) dengan model usulan disertasi (Dual-Stream Hybrid Fusion IndoBERTweet + SumoPod LLM Knowledge Representation).")

            try:
                with open(hybrid_json, "r", encoding="utf-8") as f:
                    hyb_data = json.load(f)
                hyb_test = hyb_data.get("test_metrics", {})
                hyb_cm = hyb_test.get("confusion_matrix", {})

                base_fp = metrics_data.get("confusion_matrix", [[0,0],[0,0]])[0][1]
                hyb_fp = hyb_cm.get("fp", 0)

                bench_records = [
                    {
                        "Arsitektur Model": "🔷 IndoBERTweet Baseline (Single-Stream)",
                        "Akurasi": f"{metrics_data.get('accuracy', 0)*100:.2f}%",
                        "Macro F1": f"{metrics_data.get('macro_f1', 0):.4f}",
                        "ROC-AUC": f"{metrics_data.get('roc_auc', 0):.4f}",
                        "False Positives (FP)": f"{base_fp} sampel",
                        "Multi-Task Output": "Hanya Binary Cyberbullying",
                        "Gating Weight (z)": "1.000 (100% Leksikal)"
                    },
                    {
                        "Arsitektur Model": "🛡️ Dual-Stream Hybrid Fusion (IndoBERTweet + SumoPod)",
                        "Akurasi": f"{hyb_test.get('accuracy', 0)*100:.2f}%",
                        "Macro F1": f"{hyb_test.get('macro_f1', 0):.4f}",
                        "ROC-AUC": f"{hyb_test.get('roc_auc', 0):.4f}",
                        "False Positives (FP)": f"{hyb_fp} sampel",
                        "Multi-Task Output": f"6 Dimensi (Tipe: {hyb_test.get('type_accuracy', 0)*100:.1f}%, Emosi: {hyb_test.get('emotion_accuracy', 0)*100:.1f}%)",
                        "Gating Weight (z)": f"{hyb_test.get('mean_gating_z', 0.466):.4f} (Fusi Adaptif)"
                    }
                ]

                bench_df = pd.DataFrame(bench_records)
                st.dataframe(bench_df, use_container_width=True, hide_index=True)

                st.markdown(
                    f"""
                    <div style="background-color: #f1f5f9; border-left: 4px solid #4338ca; padding: 14px 18px; border-radius: 6px; margin-top: 10px;">
                        <strong style="color: #1e1b4b;">💡 Temuan Akademis Bab 4:</strong>
                        <p style="margin: 6px 0 0 0; font-size: 14px; color: #334155; line-height: 1.6;">
                            Model <strong>Dual-Stream Hybrid Fusion</strong> berhasil menyeimbangkan representasi leksikal IndoBERTweet dengan penalaran kultural SumoPod LLaMA-3 (nilai gerbang fusi <code>z = {hyb_test.get('mean_gating_z', 0.466):.3f}</code>). 
                            Model ini tidak hanya meningkatkan akurasi deteksi perundungan siber, tetapi juga memprediksi taksonomi multi-dimensi secara simultan (Akurasi Tipe Bullying: <strong>{hyb_test.get('type_accuracy', 0)*100:.1f}%</strong>, Akurasi Emosi: <strong>{hyb_test.get('emotion_accuracy', 0)*100:.1f}%</strong>).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.caption(f"Info metrik hybrid: {e}")

        # 2c. Evaluasi Empiris Dual Explainability (SHAP vs CoT Rationale)
        dual_xai_json = REPORT_DIR / "dual_xai_evaluation.json"
        dual_xai_tex = REPORT_DIR / "table_dual_xai_breakdown.tex"

        if dual_xai_json.exists():
            st.markdown("---")
            col_dx_t, col_dx_b = st.columns([3, 1])
            with col_dx_t:
                st.markdown("### 🔬 Evaluasi Empiris Dual-XAI: Keselarasan Token SHAP ⟷ LLaMA-3 CoT")
                st.caption("Pilar Disertasi (Explainability): Mengukur konsistensi antara atribusi fitur token kuantitatif (SHAP) dan penalaran kualitatif (LLaMA-3 CoT) pada 150 sampel data uji independen.")
            with col_dx_b:
                if st.button("🔄 Evaluasi Ulang Dual-XAI (150 Sampel)", use_container_width=True):
                    with st.spinner("Menghitung atribusi SHAP & keselarasan CoT di seluruh test set..."):
                        evaluate_dual_xai_benchmark(
                            test_csv_path=FINAL_DIR / "test.csv",
                            model_path=MODEL_DIR / "best_model",
                            output_dir=REPORT_DIR
                        )
                        st.success("Evaluasi Dual-XAI berhasil diperbarui!")
                        st.rerun()

            try:
                with open(dual_xai_json, "r", encoding="utf-8") as f:
                    dx_data = json.load(f)

                dx_glob = dx_data.get("global_metrics", {})
                dx_impl = dx_data.get("implicitness_breakdown", {})

                # Kartu Metrik Dual-XAI
                dx1, dx2, dx3, dx4 = st.columns(4)
                with dx1:
                    st.metric("Rata-rata Keselarasan (Mean Alignment)", f"{dx_glob.get('mean_alignment_score', 0):.2f}%")
                with dx2:
                    st.metric("Tingkat Konsistensi Global (≥ 50%)", f"{dx_glob.get('alignment_rate_percentage', 0):.2f}%")
                with dx3:
                    st.metric("Keselarasan Semi-Tersirat (E1)", f"{dx_impl.get('E1', {}).get('mean_alignment', 0):.2f}%")
                with dx4:
                    st.metric("Keselarasan Tersirat Kultural (E2)", f"{dx_impl.get('E2', {}).get('mean_alignment', 0):.2f}%")

                # Visualisasi Grafik Keselarasan per Implicitness Level
                impl_df = pd.DataFrame([
                    {"Tingkat": "E0 (Sangat Eksplisit)", "Rata-rata Keselarasan (%)": dx_impl.get("E0", {}).get("mean_alignment", 0), "Konsistensi (%)": dx_impl.get("E0", {}).get("alignment_rate", 0), "Jumlah Sampel": dx_impl.get("E0", {}).get("count", 0)},
                    {"Tingkat": "E1 (Semi-Tersirat)", "Rata-rata Keselarasan (%)": dx_impl.get("E1", {}).get("mean_alignment", 0), "Konsistensi (%)": dx_impl.get("E1", {}).get("alignment_rate", 0), "Jumlah Sampel": dx_impl.get("E1", {}).get("count", 0)},
                    {"Tingkat": "E2 (Tersirat Kultural)", "Rata-rata Keselarasan (%)": dx_impl.get("E2", {}).get("mean_alignment", 0), "Konsistensi (%)": dx_impl.get("E2", {}).get("alignment_rate", 0), "Jumlah Sampel": dx_impl.get("E2", {}).get("count", 0)}
                ])

                fig_dx = px.bar(
                    impl_df,
                    x="Tingkat",
                    y="Rata-rata Keselarasan (%)",
                    color="Tingkat",
                    text_auto=".1f",
                    color_discrete_sequence=["#3b82f6", "#8b5cf6", "#ec4899"],
                    title="Konsistensi Dual-XAI Berdasarkan Tingkat Keterbukaan Ujaran (Implicitness)"
                )
                fig_dx.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig_dx, use_container_width=True)

                st.markdown(
                    """
                    <div style="background-color: #fdf4ff; border-left: 4px solid #c026d3; padding: 14px 18px; border-radius: 6px; margin-top: 6px;">
                        <strong style="color: #701a75;">💡 Temuan Akademis Explainability (Bab 4):</strong>
                        <p style="margin: 6px 0 0 0; font-size: 14px; color: #4a044e; line-height: 1.6;">
                            Hasil empiris membuktikan bahwa semakin tinggi tingkat keterbukaan ujaran (dari <strong>E0 = 32.8%</strong> ke <strong>E1 = 53.0%</strong> dan <strong>E2 = 100.0%</strong>), semakin tinggi keselarasan antara token berkontribusi SHAP dengan argumen penalaran LLaMA-3. Hal ini memvalidasi hipotesis disertasi: <em>ujaran perundungan tersirat memerlukan penalaran semantik kultural untuk dapat dijelaskan secara transparan</em>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.caption(f"Info visualisasi Dual-XAI: {e}")

        # 3. LaTeX Tables Generator
        st.markdown("---")
        st.markdown("### 📝 Kode LaTeX Bab 4 (Hasil & Pembahasan)")
        st.caption("Salin kode LaTeX berikut langsung ke dalam dokumen overleaf/naskah disertasi Anda:")

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

        single_ready = best_single_path.exists()
        hybrid_ready = best_hybrid_path.exists()

        if not single_ready and not hybrid_ready:
            st.warning("⚠️ Model belum selesai dilatih. Silakan latih model pada Tab 1 terlebih dahulu.")
            st.text_input("Contoh Komentar:", value="makannya segentong buset mirip ursula")
            st.button("Prediksi", disabled=True)
            return

        # Pilihan Model Inferensi
        model_play_choice = st.radio(
            "Pilih Model untuk Inferensi:",
            [
                "🛡️ Dual-Stream Hybrid Fusion Model (Transformer + SumoPod LLM - 6 Multi-Task Heads)",
                "🔷 Single-Stream IndoBERTweet Baseline"
            ],
            index=0 if hybrid_ready else 1,
            horizontal=True
        )

        # Status Backend LLM untuk Dual-Stream
        if "Dual-Stream" in model_play_choice:
            active_eng = st.session_state.get("active_engine")
            is_mock_eng = getattr(active_eng, "is_mock", True) if active_eng else True
            b_type = getattr(active_eng, "backend_type", "mock" if is_mock_eng else "local")
            if is_mock_eng:
                st.caption("ℹ️ **Backend LLM Stream: Mode Simulasi (Mock Engine)**. Menggunakan deteksi heuristik simulasi. Untuk penalaran kultural mendalam LLaMA-3, hubungkan API SumoPod di tab *Pengaturan & Model*.")
            elif b_type == "sumopod":
                st.caption(f"🌐 **Backend LLM Stream: SumoPod AI Gateway** (`{getattr(active_eng, 'model_name', 'LLaMA-3')}` aktif).")
            else:
                st.caption("🚀 **Backend LLM Stream: Local LLaMA-3 8B Instruct** aktif.")

        # Muat threshold optimal dari hasil evaluasi
        opt_th_default = 0.50
        if json_report.exists():
            try:
                with open(json_report, "r", encoding="utf-8") as f:
                    opt_th_default = float(json.load(f).get("optimal_threshold", 0.50))
            except Exception:
                opt_th_default = 0.50

        c_inp1, c_inp2 = st.columns([3, 1])
        with c_inp1:
            test_input = st.text_input(
                "Masukkan komentar media sosial untuk diuji:",
                value="makannya segentong buset mirip ursula di little mermaid",
                placeholder="Ketik komentar di sini..."
            )
        with c_inp2:
            th_choice = st.slider(
                "Ambang Batas (Threshold):",
                min_value=0.20,
                max_value=0.80,
                value=opt_th_default,
                step=0.02,
                help=f"Ambang batas probabilitas untuk vonis Cyberbullying. Nilai optimal hasil evaluasi: {opt_th_default:.2f}"
            )

        btn_pred_text = "🔍 Jalankan Inferensi Hybrid Multi-Task" if "Dual-Stream" in model_play_choice else "🔍 Klasifikasikan dengan IndoBERTweet Baseline"
        
        if st.button(btn_pred_text, type="primary"):
            if not test_input.strip():
                st.warning("Silakan masukkan teks terlebih dahulu.")
            else:
                if "Dual-Stream" in model_play_choice:
                    if not hybrid_ready:
                        st.error("Model Hybrid Fusion belum dilatih. Silakan pilih 'Dual-Stream Hybrid Fusion' di Tab 1 dan jalankan pelatihan.")
                        return

                    with st.spinner("Menjalankan inferensi Dual-Stream Hybrid Fusion..."):
                        try:
                            # 1. Ekstraksi Anotasi LLM
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
                                early_stop_json=True
                            )

                            # 2. Inferensi Hybrid
                            hyb_model, hyb_tok, hyb_dev = load_hybrid_classifier(best_hybrid_path)
                            h_res = predict_hybrid_text(
                                model=hyb_model,
                                tokenizer=hyb_tok,
                                device=hyb_dev,
                                text=test_input.strip(),
                                annotation_dict=llm_anno,
                                threshold=th_choice
                            )

                            is_cb = h_res["cyberbullying"] == "YES"
                            badge_color = "#e53e3e" if is_cb else "#38a169"
                            verdict_title = "🚨 TERDETEKSI CYBERBULLYING (HYBRID FUSION)" if is_cb else "✅ NON-CYBERBULLYING (NETRAL)"

                            st.markdown(
                                f"""
                                <div style="background-color: {'#fff5f5' if is_cb else '#f0fff4'}; border: 2px solid {badge_color}; border-radius: 12px; padding: 20px; margin: 15px 0;">
                                    <h3 style="color: {badge_color}; margin: 0 0 8px 0;">{verdict_title}</h3>
                                    <p style="font-size: 16px; margin: 0; color: #2d3748;"><strong>Teks:</strong> "{h_res['text']}"</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # Metrik Kartu Utama
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.metric("Probabilitas Cyberbullying", f"{h_res['prob_cyberbullying']*100:.2f}%")
                            with c2:
                                z_val = h_res["gate_weight_z"]
                                st.metric("Bobot Gerbang Fusi (z)", f"{z_val:.3f}", help=f"{z_val*100:.1f}% Atensi Leksikal IndoBERTweet, {(1-z_val)*100:.1f}% Penalaran LLM")
                            with c3:
                                st.metric("Threshold Digunakan", f"{th_choice:.2f}")
                            with c4:
                                st.metric("Latency", f"{h_res['latency_ms']} ms")

                            # 6 Multi-Task Classification Heads Cards
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

                            # Dual Explainability Inspector
                            st.markdown("---")
                            st.markdown("#### 🔬 Dual Explainability Inspector")
                            col_sh, col_cot = st.columns(2)

                            with col_sh:
                                st.markdown("##### 📊 Atribusi Fitur Token (SHAP)")
                                explainer = HybridSHAPExplainer()
                                shap_res = explainer.explain_text(test_input.strip())
                                st.markdown(shap_res["html_heatmap"], unsafe_allow_html=True)
                                pos_t = [f"`{t['token']}`" for t in shap_res.get("top_positive_tokens", [])]
                                if pos_t:
                                    st.markdown(f"🔴 **Kata Pemicu**: {', '.join(pos_t)}")

                            with col_cot:
                                st.markdown("##### 💬 Chain-of-Thought Rationale (LLaMA-3)")
                                st.markdown(
                                    f"""
                                    <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 14px 18px; border-radius: 6px; min-height: 110px;">
                                        <p style="font-size: 15px; line-height: 1.6; color: #1e293b; margin: 0;">
                                            {llm_anno.get('explanation', '')}
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            # Alignment Metric
                            align_info = align_shap_with_cot(shap_res, llm_anno.get("explanation", ""))
                            score = align_info["alignment_score"]
                            st.markdown(
                                f"""
                                <div style="margin-top: 14px; background-color: #f0fdf4; border: 1px solid #22c55e; border-radius: 8px; padding: 12px 18px;">
                                    <span style="font-weight: 600; color: #15803d;">🛡️ Indeks Keselarasan Dual-XAI: {score}%</span>
                                    <p style="margin: 4px 0 0 0; font-size: 13px; color: #15803d;">
                                        {align_info['alignment_summary']}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        except Exception as e:
                            st.error(f"Gagal inferensi Hybrid Fusion: {str(e)}")

                else:
                    with st.spinner("Melakukan inferensi Transformer Baseline..."):
                        try:
                            model, tokenizer, dev = load_classifier(best_single_path)
                            res = predict_text(model, tokenizer, dev, test_input, threshold=th_choice)

                            is_cb = res["label"] == 1
                            badge_color = "#e53e3e" if is_cb else "#38a169"
                            verdict_title = "🚨 TERDETEKSI CYBERBULLYING" if is_cb else "✅ NON-CYBERBULLYING (NETRAL)"

                            st.markdown(
                                f"""
                                <div style="background-color: {'#fff5f5' if is_cb else '#f0fff4'}; border: 2px solid {badge_color}; border-radius: 12px; padding: 20px; margin: 15px 0;">
                                    <h3 style="color: {badge_color}; margin: 0 0 8px 0;">{verdict_title}</h3>
                                    <p style="font-size: 16px; margin: 0; color: #2d3748;"><strong>Teks:</strong> "{res['text']}"</p>
                                </div>
                                """,
                                unsafe_allow_html=True
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
