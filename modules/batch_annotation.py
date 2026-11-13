import streamlit as st
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime

from config import DEFAULT_DATASET_PATH, CHECKPOINTS_DIR, DEFAULT_SUMOPOD_MODEL
from data_manager.dataset_handler import load_dataset_file, detect_text_column, validate_dataset_dataframe
from data_manager.checkpoint_manager import save_checkpoint, load_checkpoint, get_checkpoint_filepath
from core.model_engine import run_single_annotation, run_batch_annotation_step

def render_batch_annotation_page():
    st.markdown("## ⚡ Batch Annotation Pipeline")
    st.markdown("Anotasi dataset riset dalam jumlah besar melalui **SumoPod AI Gateway** dengan pengaturan **Konkurensi API**, **Auto-Checkpointing**, dan **Cooldown Throttling**.")

    # Check active engine
    engine = st.session_state.get("active_engine")
    is_mock = getattr(engine, "is_mock", True) if engine else True

    if is_mock:
        st.error(
            "⚠️ **PERINGATAN PENELITIAN (Mode Simulasi / Mock Fallback Aktif)**: "
            "Hasil anotasi MockEngine HANYA untuk pengujian antarmuka, DILARANG dijadikan data penelitian final "
            "dan akan ditolak secara otomatis oleh pipeline kurasi. Isi **SumoPod API Key** di tab *Pengaturan & Model*."
        )
    else:
        st.success(f"🌐 **SumoPod AI Gateway Aktif**: Menggunakan model `{getattr(engine, 'model_name', DEFAULT_SUMOPOD_MODEL)}` via Cloud API. Siap untuk anotasi batch!")

    # 1. Dataset Selection
    st.markdown("### 1. Sumber Dataset")
    ds_source = st.radio(
        "Pilih Sumber Data:",
        ["Gunakan Dataset Riset Default (Dataset-Research.csv)", "Unggah File CSV / Excel Kustom"],
        horizontal=True
    )

    df_raw = None
    dataset_name = "Dataset-Research"

    if "Default" in ds_source:
        if DEFAULT_DATASET_PATH.exists():
            df_raw, err = load_dataset_file(DEFAULT_DATASET_PATH)
            if err:
                st.error(f"Gagal membaca dataset default: {err}")
            else:
                st.success(f"Berhasil memuat `{DEFAULT_DATASET_PATH.name}` ({len(df_raw)} baris data).")
        else:
            st.error(f"File `{DEFAULT_DATASET_PATH}` tidak ditemukan di root direktori.")
    else:
        uploaded_file = st.file_uploader("Unggah Dataset (CSV atau Excel):", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            dataset_name = Path(uploaded_file.name).stem
            df_raw, err = load_dataset_file(uploaded_file)
            if err:
                st.error(f"Gagal membaca file unggahan: {err}")
            else:
                st.success(f"Berhasil memuat `{uploaded_file.name}` ({len(df_raw)} baris).")

    if df_raw is None:
        st.info("Silakan pilih atau unggah dataset terlebih dahulu.")
        return

    # 2. Text Column Selection & Validation
    detected_col = detect_text_column(df_raw)
    col_idx = list(df_raw.columns).index(detected_col) if detected_col in df_raw.columns else 0
    text_column = st.selectbox(
        "Pilih kolom teks komentar yang akan dianotasi:",
        options=list(df_raw.columns),
        index=col_idx,
        help="Kolom ini akan diekstrak dan dikirim ke model untuk dianotasi."
    )
    is_valid_df, val_warnings, val_errors = validate_dataset_dataframe(df_raw, text_col=text_column)
    for w in val_warnings:
        st.warning(f"⚠️ Validasi Dataset: {w}")
    if not is_valid_df:
        for err_msg in val_errors:
            st.error(f"❌ Validasi Dataset Gagal: {err_msg}")
        return

    # 3. Check for Existing Checkpoint
    checkpoint_file = get_checkpoint_filepath(dataset_name)
    existing_ckpt_df, _ = load_checkpoint(dataset_name)
    
    start_index = 0
    annotated_df = None

    if existing_ckpt_df is not None and len(existing_ckpt_df) > 0:
        if "cyberbullying" in existing_ckpt_df.columns:
            completed_count = existing_ckpt_df["cyberbullying"].dropna().count()
            st.info(f"💾 **Checkpoint Terdeteksi**: Ditemukan file checkpoint dengan **{completed_count}** baris yang sudah dianotasi dari {len(existing_ckpt_df)} baris.")
            
            ckpt_action = st.radio(
                "Aksi Checkpoint:",
                ["Lanjutkan dari Checkpoint (Resume)", "Mulai Ulang dari Awal (Reset)"],
                horizontal=True
            )
            
            if "Resume" in ckpt_action:
                annotated_df = existing_ckpt_df.copy()
                start_index = int(completed_count)
            else:
                annotated_df = df_raw.copy()
                start_index = 0
        else:
            annotated_df = df_raw.copy()
    else:
        annotated_df = df_raw.copy()

    # 4. Speed & Execution Controls
    st.markdown("### 2. Pengaturan Kecepatan & Eksekusi Batch")
    
    with st.expander("🚀 Panel Konfigurasi Kecepatan SumoPod API (Tuning Throughput)", expanded=True):
        col_speed1, col_speed2, col_speed3 = st.columns(3)
        
        with col_speed1:
            batch_size = st.selectbox(
                "⚡ Ukuran Batch / Concurrent Workers:",
                [1, 2, 4, 8],
                index=1,
                help="Mengirim beberapa request SumoPod API secara paralel dalam 1 chunk untuk mempercepat proses anotasi."
            )
        
        with col_speed2:
            max_tok = st.number_input(
                "✂️ Max Output Tokens:",
                min_value=80,
                max_value=400,
                value=160,
                step=20,
                help="Batas panjang token respons dari API. Skema JSON riset umumnya membutuhkan ~110-160 token."
            )
            
        with col_speed3:
            cooldown_delay = st.slider(
                "❄️ Cooldown Delay (detik):",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help="Jeda istirahat antar batch request untuk menghindari rate limit pada SumoPod API."
            )

    c1, c2, c3 = st.columns(3)
    with c1:
        max_rows_to_process = st.number_input(
            "Batas Maksimum Baris yang Diproses:",
            min_value=1,
            max_value=len(df_raw),
            value=min(50, len(df_raw)),
            help="Atur batas sampel yang ingin diproses (misal 50 sampel untuk tes atau semua baris)."
        )
    with c2:
        ckpt_interval = st.number_input(
            "Interval Simpan Checkpoint (baris):",
            min_value=1,
            max_value=100,
            value=10,
            help="Menyimpan otomatis progres ke disk setiap N baris agar data aman."
        )
    with c3:
        st.metric("Total Baris Dataset", len(df_raw))

    # Calculate actual range
    target_end_index = min(start_index + max_rows_to_process, len(df_raw))
    st.caption(f"Rentang pemrosesan: Baris indeks ke-**{start_index}** hingga **{target_end_index - 1}** (Total target: {target_end_index - start_index} baris | Batch: {batch_size}).")

    # 5. Start Execution Button
    col_btn, _ = st.columns([2, 3])
    with col_btn:
        start_btn = st.button("🚀 Mulai Eksekusi Batch Anotasi", type="primary", use_container_width=True)

    if start_btn:
        annotation_cols = [
            "cyberbullying", "type", "emotion", "cultural_context",
            "implicit_level", "severity", "confidence", "explanation", "contextual_cues", "error",
            "llm_model", "prompt_version", "annotated_at", "annotation_source", "is_mock"
        ]
        for col in annotation_cols:
            if col not in annotated_df.columns:
                annotated_df[col] = None

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        preview_container = st.container()

        total_to_run = target_end_index - start_index
        if total_to_run <= 0:
            st.warning("Semua baris dalam batas ini sudah selesai dianotasi.")
            return

        start_time = time.time()
        success_count = 0
        error_count = 0

        temp = st.session_state.get("temperature", 0.1)
        sys_prompt = st.session_state.get("custom_system_prompt", None)

        recent_results = []
        processed_in_run = 0

        # Chunk indices according to batch_size
        indices = list(range(start_index, target_end_index))
        chunks = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        for chunk_idx, chunk in enumerate(chunks):
            chunk_texts = [str(annotated_df.iloc[i][text_column]) for i in chunk]
            
            # Run batch or single inference
            try:
                if len(chunk) == 1:
                    res_list = [run_single_annotation(
                        engine=engine,
                        text=chunk_texts[0],
                        system_prompt=sys_prompt,
                        max_new_tokens=max_tok,
                        temperature=temp
                    )]
                else:
                    res_list = run_batch_annotation_step(
                        engine=engine,
                        texts=chunk_texts,
                        system_prompt=sys_prompt,
                        max_new_tokens=max_tok,
                        temperature=temp
                    )

                # Assign results back to dataframe
                for idx, res in zip(chunk, res_list):
                    for k in annotation_cols:
                        if k in res:
                            val = res[k]
                            if k == "contextual_cues" and isinstance(val, (dict, list)):
                                val = json.dumps(val, ensure_ascii=False)
                            annotated_df.at[idx, k] = val

                    success_count += 1
                    curr_text = str(annotated_df.iloc[idx][text_column])
                    recent_results.insert(0, {
                        "Baris": idx + 1,
                        "Komentar": curr_text[:60] + "..." if len(curr_text) > 60 else curr_text,
                        "Cyberbullying": res.get("cyberbullying"),
                        "Type": res.get("type"),
                        "Severity": res.get("severity"),
                        "Latency (s)": res.get("latency_sec")
                    })

                recent_results = recent_results[:5]  # Keep 5 latest

            except Exception as e:
                for idx in chunk:
                    error_count += 1
                    annotated_df.at[idx, "error"] = str(e)

            processed_in_run += len(chunk)
            elapsed = time.time() - start_time
            rate = processed_in_run / elapsed if elapsed > 0 else 0
            remaining = total_to_run - processed_in_run
            eta = remaining / rate if rate > 0 else 0

            # Update status message
            status_text.markdown(
                f"**Memproses:** `{processed_in_run}/{total_to_run}` baris "
                f"| ⚡ Kecepatan: **`{rate:.2f} sampel/detik`** "
                f"| ⏳ ETA: **`{eta:.1f} detik`**"
            )
            progress_bar.progress(processed_in_run / total_to_run)

            # Live Preview Update
            with preview_container:
                st.markdown("##### 🔴 Live Stream (5 Sampel Terakhir):")
                st.dataframe(pd.DataFrame(recent_results), use_container_width=True)

            # Auto-save Checkpoint
            if (processed_in_run % ckpt_interval == 0) or (chunk_idx == len(chunks) - 1):
                save_checkpoint(annotated_df, dataset_name=dataset_name)

            # Cooldown Throttling Delay if requested
            if cooldown_delay > 0 and chunk_idx < len(chunks) - 1:
                time.sleep(cooldown_delay)

        # Finished
        total_time = max(time.time() - start_time, 1e-4)
        save_checkpoint(annotated_df, dataset_name=dataset_name)
        st.session_state["latest_batch_df"] = annotated_df

        if is_mock:
            st.warning(
                f"🧪 Simulasi Batch Selesai ({total_to_run} baris dalam {total_time:.1f}s). "
                "Data bertanda `annotation_source='mock_simulation'` dan TIDAK digunakan untuk training riset."
            )
        else:
            st.success(
                f"🎉 Selesai! Berhasil memproses **{total_to_run}** baris dalam **{total_time:.1f}** detik "
                f"(Rata-rata: {total_to_run / total_time:.2f} sampel/detik | Sukses: {success_count}, Error: {error_count}). "
                f"Checkpoint tersimpan di `{checkpoint_file.name}`."
            )

    # 6. Preview & Download Checkpoint
    if "latest_batch_df" in st.session_state or existing_ckpt_df is not None:
        view_df = st.session_state.get("latest_batch_df", existing_ckpt_df)
        st.markdown("---")
        st.markdown("### 3. Data Teranotasi Saat Ini")
        st.dataframe(view_df.head(100), use_container_width=True, height=260)

        csv_dl = view_df.to_csv(index=False).encode('utf-8')
        c_dl1, c_dl2 = st.columns([1, 1])
        with c_dl1:
            st.download_button(
                "📥 Unduh Hasil Batch Terkini (CSV)",
                data=csv_dl,
                file_name=f"IICD_annotated_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_dl2:
            st.success("💾 Progres anotasi tersimpan otomatis ke checkpoint dan siap digunakan pada tab 'Training & Evaluasi IndoBERTweet'.")
