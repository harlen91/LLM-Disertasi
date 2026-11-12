import streamlit as st
import sys
from pathlib import Path

# Add app directory to sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import DEFAULT_DATASET_PATH, CHECKPOINTS_DIR
from modules.settings_page import render_settings_page
from modules.single_inference import render_single_inference_page
from modules.batch_annotation import render_batch_annotation_page
from modules.analytics_dashboard import render_analytics_dashboard_page
from modules.model_training_page import render_model_training_page
from core.model_engine import MockEngine

# Streamlit Page Config
st.set_page_config(
    page_title="IICD Annotation Studio | LLaMA-3 Explainable",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS Injection)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    .main-header h1 {
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        color: #f8fafc;
    }
    
    .main-header p {
        font-size: 14px;
        color: #94a3b8;
        margin: 6px 0 0 0;
    }
    
    /* Metric Card Enhancements */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Custom Badge Styles */
    .badge-cyberbullying-yes {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    
    .badge-cyberbullying-no {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

from config import (
    DEFAULT_DATASET_PATH, CHECKPOINTS_DIR, DEFAULT_MODEL_NAME, 
    DEFAULT_CACHE_DIR, DEFAULT_SUMOPOD_BASE_URL, DEFAULT_SUMOPOD_MODEL
)
from core.model_engine import (
    get_cached_llama_engine, get_cached_mock_engine, get_cached_sumopod_engine
)

# Session State & Global Engine Initialization
if "engine_mode" not in st.session_state:
    st.session_state["engine_mode"] = "mock"
if "llama_loaded" not in st.session_state:
    st.session_state["llama_loaded"] = False
if "sumopod_connected" not in st.session_state:
    st.session_state["sumopod_connected"] = False

# Resolve active engine globally before any tabs run
engine_mode = st.session_state.get("engine_mode", "mock")

if engine_mode == "sumopod" and st.session_state.get("sumopod_connected", False):
    try:
        api_key = st.session_state.get("sumopod_api_key", "")
        if not api_key:
            try:
                api_key = st.secrets.get("SUMOPOD_API_KEY", "")
            except Exception:
                api_key = ""
                
        st.session_state["active_engine"] = get_cached_sumopod_engine(
            api_key=api_key,
            base_url=st.session_state.get("sumopod_base_url", DEFAULT_SUMOPOD_BASE_URL),
            model_name=st.session_state.get("sumopod_model", DEFAULT_SUMOPOD_MODEL)
        )
    except Exception as e:
        st.session_state["active_engine"] = get_cached_mock_engine()
        st.session_state["engine_mode"] = "mock"
        st.session_state["sumopod_connected"] = False

elif engine_mode == "local_llama" and st.session_state.get("llama_loaded", False):
    try:
        st.session_state["active_engine"] = get_cached_llama_engine(
            model_name=st.session_state.get("model_name", DEFAULT_MODEL_NAME),
            cache_dir=st.session_state.get("hf_cache_dir", DEFAULT_CACHE_DIR),
            device_mode=st.session_state.get("device_choice", "auto"),
            quantization=st.session_state.get("quant_choice", "4-bit")
        )
    except Exception as e:
        st.session_state["active_engine"] = get_cached_mock_engine()
        st.session_state["engine_mode"] = "mock"
        st.session_state["llama_loaded"] = False
else:
    st.session_state["active_engine"] = get_cached_mock_engine()

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ IICD - LLM-Augmented Explainable Annotation Studio</h1>
    <p>Aplikasi Penelitian Disertasi: Pipeline Anotasi & Distilasi Multi-Dimensi Cyberbullying Bahasa Indonesia (Meta LLaMA-3 8B ke Transformer IndoBERTweet)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.image("https://img.shields.io/badge/Model-Meta%20LLaMA--3%208B%20Instruct-blue?style=for-the-badge&logo=meta", use_container_width=True)
    st.markdown("### 📌 Status Eksekusi")
    
    engine = st.session_state.get("active_engine")
    is_mock = getattr(engine, "is_mock", True)
    backend_type = getattr(engine, "backend_type", "mock" if is_mock else "local")
    
    if is_mock:
        st.warning("⚡ **Backend**: Simulation / Mock Mode")
    elif backend_type == "sumopod":
        st.success(f"🌐 **Backend**: SumoPod Gateway\n(`{getattr(engine, 'model_name', 'LLaMA-3')}`)")
    else:
        st.success("🧠 **Backend**: Local LLaMA-3 8B (HF Cache)")
        
    st.markdown(f"- **Dataset Default**: `{DEFAULT_DATASET_PATH.name}`")
    
    # Checkpoints counter
    ckpts = list(CHECKPOINTS_DIR.glob("*.csv"))
    st.markdown(f"- **Berkas Checkpoint**: `{len(ckpts)} ditemukan`")

    st.markdown("---")
    st.markdown("### 🎓 Konteks Penelitian")
    st.caption("Penelitian Disertasi mengenai *Explainable Cyberbullying Detection* dengan mengombinasikan taksonomi budaya (cultural context), tingkat implisit (implicitness), dan penalaran rantai berpikir (chain-of-thought rationale).")
    
    st.markdown("---")
    st.caption("Dikembangkan untuk macOS / Apple Silicon & Server Komputasi.")

# Main Navigation Tabs
tab_nav1, tab_nav2, tab_nav3, tab_nav4, tab_nav5 = st.tabs([
    "🧪 Playground & Explainability",
    "⚡ Batch Annotation Pipeline",
    "📊 Dashboard & Statistik",
    "🤖 Training & Evaluasi IndoBERTweet",
    "⚙️ Pengaturan & Model"
])

with tab_nav1:
    render_single_inference_page()

with tab_nav2:
    render_batch_annotation_page()

with tab_nav3:
    render_analytics_dashboard_page()

with tab_nav4:
    render_model_training_page()

with tab_nav5:
    render_settings_page()
