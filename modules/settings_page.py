import os
import streamlit as st
import torch
from config import (
    TAXONOMY,
    DEFAULT_DATASET_PATH,
    DEFAULT_SUMOPOD_BASE_URL,
    DEFAULT_SUMOPOD_MODEL,
    SUMOPOD_POPULAR_MODELS,
)
from core.model_engine import get_cached_sumopod_engine, get_cached_mock_engine, mask_secret
from core.prompt_templates import DEFAULT_SYSTEM_PROMPT, LABEL_SCHEMA, PROMPT_VERSION


def render_settings_page():
    st.markdown("## ⚙️ Konfigurasi Sistem & SumoPod API")
    st.markdown(
        "Kelola koneksi **SumoPod AI Gateway** sebagai mesin inferensi LLM utama serta referensi taksonomi riset."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🌐 SumoPod AI Gateway (Cloud LLM)")
        st.info(
            "🌐 **SumoPod AI Gateway** adalah satu-satunya backend LLM aktif (OpenAI-compatible Cloud API). "
            "Jika API Key belum diisi, sistem otomatis menggunakan **MockEngine** sebagai fallback aman."
        )

        secret_key = ""
        try:
            secret_key = str(st.secrets.get("SUMOPOD_API_KEY", "") or "").strip()
        except Exception:
            secret_key = ""
        if not secret_key:
            secret_key = os.environ.get("SUMOPOD_API_KEY", "").strip()

        sumo_key = st.text_input(
            "SumoPod API Key:",
            value=st.session_state.get("sumopod_api_key", secret_key),
            type="password",
            help="Dapatkan API Key di sumopod.com pada menu AI -> API Keys."
        ).strip()
        st.session_state["sumopod_api_key"] = sumo_key

        sumo_url = st.text_input(
            "SumoPod Base URL:",
            value=st.session_state.get("sumopod_base_url", DEFAULT_SUMOPOD_BASE_URL),
            help="Default: https://api.sumopod.com/v1"
        ).strip() or DEFAULT_SUMOPOD_BASE_URL
        st.session_state["sumopod_base_url"] = sumo_url

        custom_label = "Custom (ketik manual)"
        model_options = SUMOPOD_POPULAR_MODELS + [custom_label]
        saved_model = st.session_state.get("sumopod_model", DEFAULT_SUMOPOD_MODEL)
        default_idx = (
            SUMOPOD_POPULAR_MODELS.index(saved_model)
            if saved_model in SUMOPOD_POPULAR_MODELS
            else len(SUMOPOD_POPULAR_MODELS)
        )

        selected_option = st.selectbox(
            "Model SumoPod API:",
            options=model_options,
            index=default_idx,
            help="Pilih model LLM di SumoPod AI Gateway atau pilih Custom untuk memasukkan identifier model lainnya."
        )
        if selected_option == custom_label:
            sumo_model = st.text_input(
                "Identifier Model Custom:",
                value="" if saved_model in SUMOPOD_POPULAR_MODELS else saved_model,
                placeholder=f"contoh: {DEFAULT_SUMOPOD_MODEL}"
            ).strip() or DEFAULT_SUMOPOD_MODEL
        else:
            sumo_model = selected_option
        st.session_state["sumopod_model"] = sumo_model

        demo_fallback = st.checkbox(
            "Gunakan Mode Simulasi (Mock Fallback) sementara untuk demo UI",
            value=st.session_state.get("engine_mode", "sumopod") == "mock",
            help="Centang opsi ini jika ingin menguji alur UI tanpa memanggil kuota SumoPod API."
        )
        st.session_state["engine_mode"] = "mock" if demo_fallback else "sumopod"

        active_engine = st.session_state.get("active_engine")
        is_sumo_active = (
            not demo_fallback
            and bool(sumo_key)
            and getattr(active_engine, "backend_type", "") == "sumopod"
        )

        if demo_fallback:
            st.session_state["active_engine"] = get_cached_mock_engine()
            st.session_state["sumopod_connected"] = False
            st.warning("⚡ **Mode Simulasi (Mock Fallback) Aktif**: Anotasi dijalankan secara heuristik lokal untuk demo UI.")
        elif not sumo_key:
            st.session_state["active_engine"] = get_cached_mock_engine()
            st.session_state["sumopod_connected"] = False
            st.warning(
                "⚠️ **SumoPod API Key belum diisi**: Sistem berjalan pada **MockEngine Fallback**. "
                "Masukkan SumoPod API Key di atas untuk mengaktifkan inferensi LLM riil."
            )
        elif is_sumo_active:
            st.success(f"✅ **SumoPod Gateway AKTIF**: Menggunakan model `{sumo_model}`.")

        sumo_btn = st.button("🔌 Uji Koneksi & Aktifkan SumoPod API", type="primary", use_container_width=True)
        if sumo_btn:
            if not sumo_key:
                st.error("❌ Silakan masukkan SumoPod API Key terlebih dahulu.")
            else:
                with st.spinner("Menguji koneksi ke SumoPod AI Gateway..."):
                    test_engine = get_cached_sumopod_engine(
                        api_key=sumo_key,
                        base_url=sumo_url,
                        model_name=sumo_model
                    )
                    ok, msg = test_engine.test_connection()
                    if ok:
                        st.session_state["active_engine"] = test_engine
                        st.session_state["sumopod_connected"] = True
                        st.session_state["engine_mode"] = "sumopod"
                        st.session_state["sumopod_api_key"] = sumo_key
                        st.session_state["sumopod_base_url"] = sumo_url
                        st.session_state["sumopod_model"] = sumo_model
                        st.success(f"🎉 {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ Koneksi gagal: {msg}")

        # Hyperparameter & Speed Settings
        st.markdown("---")
        st.markdown("### 🎛️ Hyperparameter & Optimasi Kecepatan")

        speed_profile = st.selectbox(
            "Profil Optimasi Performa:",
            [
                "⚖️ Mode Standar Riset (Max 220 Token, Lengkap & Aman)",
                "🚀 Mode Kecepatan Tinggi (Max 180 Token, Ringkas)",
                "❄️ Mode Ekstra / Safe (Max 260 Token, Penjelasan Panjang)"
            ],
            index=0
        )

        if "Standar Riset" in speed_profile:
            rec_tokens = 220
        elif "Kecepatan Tinggi" in speed_profile:
            rec_tokens = 180
        else:
            rec_tokens = 260

        col_t, col_m = st.columns(2)
        with col_t:
            temp = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Nilai rendah (0.1) direkomendasikan untuk anotasi deterministik dan konsisten."
            )
            st.session_state["temperature"] = temp
        with col_m:
            max_tok = st.slider(
                "Max New Tokens:",
                min_value=80,
                max_value=400,
                value=rec_tokens,
                step=20,
                help="Membatasi panjang token output membantu mempercepat respons API dan menghemat kuota."
            )
            st.session_state["max_tokens"] = max_tok

    with col2:
        st.markdown("### 📚 Taksonomi Anotasi & Skema Riset")
        st.markdown("Taksonomi multi-dimensi cyberbullying Bahasa Indonesia:")

        tab1, tab2, tab3 = st.tabs(["🏷️ Taksonomi Label", "📝 System Prompt", "🖥️ Status Sistem"])

        with tab1:
            for cat, items in TAXONOMY.items():
                with st.expander(f"Dimensi: **{cat.upper()}**", expanded=False):
                    if isinstance(items, list):
                        st.write(", ".join([f"`{x}`" for x in items]))
                    else:
                        for code, desc in items.items():
                            st.markdown(f"- **`{code}`**: {desc}")

        with tab2:
            st.markdown("**System Prompt Default:**")
            sys_prompt = st.text_area(
                "Edit System Prompt:",
                value=st.session_state.get("custom_system_prompt", DEFAULT_SYSTEM_PROMPT),
                height=180
            )
            st.session_state["custom_system_prompt"] = sys_prompt

            with st.expander("Lihat Format Skema Prompt SumoPod API"):
                st.code(LABEL_SCHEMA, language="text")

        with tab3:
            st.markdown("**Informasi Environment Host & API:**")
            st.markdown(f"- **Platform OS**: `{os.uname().sysname} {os.uname().machine}`")
            st.markdown(f"- **Backend LLM Utama**: `SumoPod AI Gateway ({sumo_model})`")
            api_status = f"Siap ({mask_secret(sumo_key)})" if sumo_key else "API Key belum diisi (Mock Fallback aktif)"
            st.markdown(f"- **Status Kredensial SumoPod**: `{api_status}`")
            st.markdown(f"- **Prompt Version Aktif**: `{PROMPT_VERSION}`")
            st.markdown(f"- **PyTorch Version (IndoBERTweet)**: `{torch.__version__}`")
            st.markdown(f"- **MPS (Apple Silicon)**: `{'Tersedia' if torch.backends.mps.is_available() else 'Tidak'}`")
            st.markdown(f"- **Dataset Default**: `{DEFAULT_DATASET_PATH.name}` (`{'Ditemukan' if DEFAULT_DATASET_PATH.exists() else 'Tidak Ditemukan'}`)")
