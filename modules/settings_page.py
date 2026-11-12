import streamlit as st
import os
import torch
from pathlib import Path
from config import (
    DEFAULT_CACHE_DIR, DEFAULT_MODEL_NAME, TAXONOMY, DEFAULT_DATASET_PATH,
    DEFAULT_SUMOPOD_BASE_URL, DEFAULT_SUMOPOD_MODEL, SUMOPOD_POPULAR_MODELS
)
from core.prompt_templates import DEFAULT_SYSTEM_PROMPT, LABEL_SCHEMA

def render_settings_page():
    st.markdown("## ⚙️ Konfigurasi Sistem & Model Backend")
    st.markdown("Kelola preferensi komputasi, sumber model LLaMA-3 dari cache lokal HuggingFace, dan referensi taksonomi riset.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🧠 Backend Mesin Inferensi")
        
        # Engine Mode Selector with persistent key
        mode_saved = st.session_state.get("engine_mode", "mock")
        if mode_saved == "sumopod":
            current_idx = 1
        elif mode_saved == "local_llama":
            current_idx = 2
        else:
            current_idx = 0

        engine_mode = st.radio(
            "Pilih Mode Backend Model:",
            [
                "Simulation / Mock Engine (Instan - Evaluasi UI)",
                "SumoPod AI Gateway (Cloud API - Rekomendasi Deploy & Cepat)",
                "Local LLaMA-3 8B Instruct (HuggingFace Cache Lokal)"
            ],
            index=current_idx,
            key="engine_mode_radio_selector",
            help="Gunakan SumoPod API untuk eksekusi cloud super cepat & deploy Streamlit Cloud. Gunakan Local LLaMA-3 untuk pemrosesan offline di Mac."
        )
        
        if "SumoPod" in engine_mode:
            selected_mode = "sumopod"
        elif "Local" in engine_mode:
            selected_mode = "local_llama"
        else:
            selected_mode = "mock"

        st.session_state["engine_mode"] = selected_mode

        # ----------------- OPTION 1: SUMOPOD AI GATEWAY -----------------
        if selected_mode == "sumopod":
            st.info("🌐 **SumoPod AI Gateway**: Menggunakan API cloud OpenAI-compatible. Sangat ringan (RAM < 100MB), cepat, dan siap dideploy ke Streamlit Community Cloud.")
            
            secret_key = ""
            try:
                secret_key = st.secrets.get("SUMOPOD_API_KEY", "")
            except Exception:
                secret_key = ""

            sumo_key = st.text_input(
                "SumoPod API Key:",
                value=st.session_state.get("sumopod_api_key", secret_key),
                type="password",
                help="Dapatkan API Key di sumopod.com pada menu AI -> API Keys."
            )
            st.session_state["sumopod_api_key"] = sumo_key

            sumo_url = st.text_input(
                "SumoPod Base URL:",
                value=st.session_state.get("sumopod_base_url", DEFAULT_SUMOPOD_BASE_URL),
                help="Default: https://api.sumopod.com/v1"
            )
            st.session_state["sumopod_base_url"] = sumo_url

            sumo_model = st.text_input(
                "Model Identifier di SumoPod:",
                value=st.session_state.get("sumopod_model", DEFAULT_SUMOPOD_MODEL),
                help="Ketik identifier model sesuai yang terdaftar di SumoPod (contoh: meta-llama/Meta-Llama-3-8B-Instruct atau llama-3-8b-instruct)."
            )
            st.session_state["sumopod_model"] = sumo_model
            st.caption("💡 Contoh: `meta-llama/Meta-Llama-3-8B-Instruct`, `llama-3-8b-instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `deepseek-ai/DeepSeek-V3`")

            is_sumo_active = st.session_state.get("sumopod_connected", False) and getattr(st.session_state.get("active_engine"), "backend_type", "") == "sumopod"
            if is_sumo_active:
                st.success(f"✅ **SumoPod Gateway AKTIF**: Terhubung ke `{sumo_model}`. Semua tab akan menggunakan backend ini.")

            sumo_btn = st.button("🔌 Uji Koneksi & Aktifkan SumoPod API", type="primary", use_container_width=True)
            if sumo_btn:
                if not sumo_key.strip():
                    st.error("❌ Silakan masukkan SumoPod API Key terlebih dahulu.")
                else:
                    with st.spinner("Menguji koneksi ke SumoPod AI Gateway..."):
                        from core.model_engine import get_cached_sumopod_engine
                        test_engine = get_cached_sumopod_engine(
                            api_key=sumo_key,
                            base_url=sumo_url,
                            model_name=sumo_model
                        )
                        ok, msg = test_engine.test_connection()
                        if ok:
                            st.session_state["active_engine"] = test_engine
                            st.session_state["sumopod_connected"] = True
                            st.session_state["llama_loaded"] = False
                            st.session_state["engine_mode"] = "sumopod"
                            st.session_state["sumopod_api_key"] = sumo_key
                            st.session_state["sumopod_base_url"] = sumo_url
                            st.session_state["sumopod_model"] = sumo_model
                            st.success(f"🎉 {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ Koneksi gagal: {msg}")

        # ----------------- OPTION 2: LOCAL LLAMA-3 8B -----------------
        elif selected_mode == "local_llama":
            st.info("💡 Model akan dimuat dari cache lokal HuggingFace yang telah terunduh di sistem Anda.")
            
            hf_cache = st.text_input(
                "Direktori HuggingFace Hub Cache:",
                value=st.session_state.get("hf_cache_dir", DEFAULT_CACHE_DIR)
            )
            st.session_state["hf_cache_dir"] = hf_cache
            
            model_id = st.text_input(
                "Model Identifier:",
                value=st.session_state.get("model_name", DEFAULT_MODEL_NAME)
            )
            st.session_state["model_name"] = model_id
            
            # Hardware & Quantization
            st.markdown("#### Akselerasi Perangkat Keras")
            mps_available = torch.backends.mps.is_available()
            cuda_available = torch.cuda.is_available()
            
            dev_options = ["Auto"]
            if mps_available:
                dev_options.append("Apple Silicon GPU (MPS)")
            if cuda_available:
                dev_options.append("NVIDIA GPU (CUDA)")
            dev_options.append("CPU")
            
            device_choice = st.selectbox("Device Akselerasi:", dev_options)
            st.session_state["device_choice"] = device_choice

            quant_choice = st.selectbox(
                "Kuantisasi Model:",
                ["4-bit (NF4 BitsAndBytes) - Rekomendasi VRAM Hemat", "None (float16)"]
            )
            st.session_state["quant_choice"] = "4-bit" if "4-bit" in quant_choice else "none"
            
            # Status Banner
            is_active_now = st.session_state.get("llama_loaded", False) and not getattr(st.session_state.get("active_engine"), "is_mock", True) and getattr(st.session_state.get("active_engine"), "backend_type", "") != "sumopod"
            if is_active_now:
                st.success("🧠 **Model LLaMA-3 8B saat ini sedang AKTIF di memori!** Semua tab akan menggunakan model riil ini.")

            # Button to trigger model load
            load_label = "🔄 Muat Ulang Model ke Memori" if is_active_now else "🚀 Muat Model ke Memori Sekarang"
            load_btn = st.button(load_label, use_container_width=True, type="primary")
            if load_btn:
                with st.spinner("Memuat LLaMA-3 8B Instruct dari cache... Proses ini memakan waktu ~20-40 detik."):
                    try:
                        from core.model_engine import get_cached_llama_engine
                        q_val = "4-bit" if "4-bit" in quant_choice else "none"
                        d_mode = "mps" if "MPS" in device_choice else ("cuda" if "CUDA" in device_choice else "auto")
                        
                        engine = get_cached_llama_engine(
                            model_name=model_id,
                            cache_dir=hf_cache,
                            device_mode=d_mode,
                            quantization=q_val
                        )
                        st.session_state["active_engine"] = engine
                        st.session_state["llama_loaded"] = True
                        st.session_state["sumopod_connected"] = False
                        st.session_state["engine_mode"] = "local_llama"
                        st.session_state["model_name"] = model_id
                        st.session_state["hf_cache_dir"] = hf_cache
                        st.session_state["device_choice"] = device_choice
                        st.session_state["quant_choice"] = q_val
                        st.success("✅ Model LLaMA-3 8B berhasil dimuat ke memori!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Gagal memuat model: {str(e)}")

        # ----------------- OPTION 3: SIMULATION / MOCK -----------------
        else:
            # Mock mode setup
            from core.model_engine import get_cached_mock_engine
            st.session_state["active_engine"] = get_cached_mock_engine()
            st.session_state["llama_loaded"] = False
            st.session_state["sumopod_connected"] = False
            st.session_state["engine_mode"] = "mock"
            st.info("⚡ Mock Engine aktif! Respons inferensi akan dihasilkan secara cepat untuk prototyping.")

        # Hyperparameter & Speed Settings
        st.markdown("---")
        st.markdown("### 🎛️ Hyperparameter & Optimasi Kecepatan")
        
        speed_profile = st.selectbox(
            "Profil Optimasi Performa:",
            [
                "⚖️ Mode Standar Riset (Max 220 Token, Lengkap & Aman)",
                "🚀 Mode Kecepatan Tinggi (Max 180 Token, Stop pada '}')",
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
            temp = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.1, step=0.05, 
                             help="Nilai rendah (0.1) direkomendasikan untuk anotasi deterministik dan konsisten.")
            st.session_state["temperature"] = temp
        with col_m:
            max_tok = st.slider("Max New Tokens:", min_value=80, max_value=400, value=rec_tokens, step=20,
                                help="Memotong panjang token maksimum secara signifikan mempercepat waktu generasi (skema JSON riset hanya butuh ~110 token).")
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
            
            with st.expander("Lihat Format Skema Prompt LLaMA-3"):
                st.code(LABEL_SCHEMA, language="text")

        with tab3:
            st.markdown("**Informasi Environment Host:**")
            st.markdown(f"- **Platform OS**: `{os.uname().sysname} {os.uname().machine}`")
            st.markdown(f"- **PyTorch Version**: `{torch.__version__}`")
            st.markdown(f"- **MPS (Apple Silicon)**: `{'Tersedia' if torch.backends.mps.is_available() else 'Tidak'}`")
            st.markdown(f"- **Dataset Default**: `{DEFAULT_DATASET_PATH.name}` (`{'Ditemukan' if DEFAULT_DATASET_PATH.exists() else 'Tidak Ditemukan'}`)")
            
            cache_p = Path(DEFAULT_CACHE_DIR)
            model_cached = (cache_p / "models--meta-llama--Meta-Llama-3-8B-Instruct").exists()
            st.markdown(f"- **Cache LLaMA-3 8B**: `{'Tersedia di ' + str(cache_p) if model_cached else 'Tidak Ditemukan'}`")
