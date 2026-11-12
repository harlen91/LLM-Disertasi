import streamlit as st
import pandas as pd
from datetime import datetime
from config import SAMPLE_COMMENTS, TAXONOMY, DEFAULT_SUMOPOD_MODEL
from core.model_engine import run_single_annotation
from core.shap_explainer import HybridSHAPExplainer, align_shap_with_cot

@st.cache_resource
def get_cached_shap_explainer():
    try:
        return HybridSHAPExplainer()
    except Exception:
        return None

def get_full_label(category: str, code: str) -> str:
    """Helper to convert code to human-friendly description."""
    cat_dict = TAXONOMY.get(category, {})
    if isinstance(cat_dict, dict):
        return cat_dict.get(code, code)
    return code

def render_single_inference_page():
    st.markdown("## 🧪 Playground Anotasi & Explainability Inspector")
    st.markdown("Uji komentar individual secara instan untuk memeriksa klasifikasi multi-dimensi dan penalaran (*chain-of-thought explanation*) dari model LLM via **SumoPod AI Gateway**.")

    # Check active engine
    engine = st.session_state.get("active_engine")
    is_mock = getattr(engine, "is_mock", True) if engine else True

    if is_mock:
        st.caption("ℹ️ Mode saat ini: **Mode Simulasi / Mock Fallback** (Masukkan SumoPod API Key di tab *Pengaturan & Model* untuk inferensi LLM riil).")
    else:
        st.caption(f"🌐 Mode saat ini: **SumoPod AI Gateway** aktif (`{getattr(engine, 'model_name', DEFAULT_SUMOPOD_MODEL)}`).")

    # Session history list for single runs
    if "single_history" not in st.session_state:
        st.session_state["single_history"] = []

    # Quick Sample Selector
    col_sel, col_clear = st.columns([4, 1])
    with col_sel:
        selected_sample = st.selectbox(
            "Pilih contoh komentar dari dataset riset:",
            ["-- Ketik komentar kustom di bawah --"] + SAMPLE_COMMENTS
        )
    with col_clear:
        if st.button("🗑️ Bersihkan Riwayat", use_container_width=True):
            st.session_state["single_history"] = []
            st.rerun()

    # Input Box
    default_text = "" if selected_sample.startswith("--") else selected_sample
    user_text = st.text_area(
        "Komentar Bahasa Indonesia yang diuji:",
        value=default_text,
        placeholder="Contoh: 'makannya segentong buset' atau ketik komentar apapun...",
        height=100
    )

    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        analyze_btn = st.button("🔍 Jalankan Anotasi", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_text.strip():
            st.warning("Silakan masukkan teks komentar terlebih dahulu.")
            return

        with st.spinner("Menganalisis teks menggunakan taksonomi riset..."):
            temp = st.session_state.get("temperature", 0.1)
            max_tok = st.session_state.get("max_tokens", 160)
            sys_prompt = st.session_state.get("custom_system_prompt", None)
            
            result = run_single_annotation(
                engine=engine,
                text=user_text.strip(),
                system_prompt=sys_prompt,
                max_new_tokens=max_tok,
                temperature=temp
            )

            # Record to history
            record = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "comment": user_text.strip(),
                **result
            }
            st.session_state["single_history"].insert(0, record)
            st.session_state["last_result"] = record

    # Display Last Result
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        st.markdown("---")
        st.markdown("### 📋 Hasil Anotasi & Penjelasan Model")

        # Top banner: Cyberbullying verdict
        is_cb = res.get("cyberbullying", "").upper() == "YES"
        verdict_color = "#e53e3e" if is_cb else "#38a169"
        verdict_bg = "#fff5f5" if is_cb else "#f0fff4"
        verdict_border = "#feb2b2" if is_cb else "#9ae6b4"
        verdict_icon = "⚠️ TERDETEKSI CYBERBULLYING" if is_cb else "✅ BUKAN CYBERBULLYING"

        st.markdown(
            f"""
            <div style="background-color: {verdict_bg}; border: 2px solid {verdict_border}; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
                <h3 style="color: {verdict_color}; margin: 0 0 6px 0; font-size: 22px;">{verdict_icon}</h3>
                <p style="margin: 0; color: #4a5568; font-size: 15px;"><strong>Teks Diuji:</strong> "{res.get('comment', '')}"</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Multi-dimensional Badge Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            code = res.get("type", "-")
            st.metric("Tipe (Type)", code, help=get_full_label("type", code))
            st.caption(f"_{get_full_label('type', code)}_")
        with c2:
            code = res.get("emotion", "-")
            st.metric("Emosi (Emotion)", code, help=get_full_label("emotion", code))
            st.caption(f"_{get_full_label('emotion', code)}_")
        with c3:
            code = res.get("cultural_context", "-")
            st.metric("Konteks Kultural", code, help=get_full_label("cultural_context", code))
            st.caption(f"_{get_full_label('cultural_context', code)}_")
        with c4:
            code = res.get("implicit_level", "-")
            st.metric("Keterbukaan", code, help=get_full_label("implicit_level", code))
            st.caption(f"_{get_full_label('implicit_level', code)}_")
        with c5:
            code = res.get("severity", "-")
            st.metric("Tingkat Keparahan", code, help=get_full_label("severity", code))
            st.caption(f"_{get_full_label('severity', code)}_")

        # Dual Explainability Framework (SHAP + CoT Reasoning)
        st.markdown("#### 🔬 Dual Explainability Framework (SHAP Token Attribution + CoT Rationale)")
        st.caption("Pilar Disertasi: Mengintegrasikan atribusi fitur token kuantitatif (SHAP) dengan penalaran bahasa alami kualitatif (SumoPod LLM Chain-of-Thought).")

        col_xai_left, col_xai_right = st.columns([1, 1])

        # KIRI: Kuantitatif (SHAP Token Saliency Heatmap)
        with col_xai_left:
            st.markdown("##### 📊 1. Kuantitatif: Token Feature Attribution (SHAP)")
            explainer = get_cached_shap_explainer()
            shap_res = None
            if explainer is not None:
                with st.spinner("Menghitung atribusi SHAP token..."):
                    shap_res = explainer.explain_text(res.get("comment", ""))
                    st.markdown(shap_res["html_heatmap"], unsafe_allow_html=True)
                    
                    pos_tokens = shap_res.get("top_positive_tokens", [])
                    neg_tokens = shap_res.get("top_negative_tokens", [])
                    
                    if pos_tokens:
                        pos_str = ", ".join([f"`{t['token']}` (+{t['shap_value']:.3f})" for t in pos_tokens])
                        st.markdown(f"🔴 **Kata Pemicu Cyberbullying**: {pos_str}")
                    if neg_tokens:
                        neg_str = ", ".join([f"`{t['token']}` ({t['shap_value']:.3f})" for t in neg_tokens])
                        st.markdown(f"🟢 **Kata Peredam / Netral**: {neg_str}")
            else:
                st.info("Modul SHAP sedang diinisialisasi atau belum tersedia.")

        # KANAN: Kualitatif (SumoPod LLM CoT Rationale)
        with col_xai_right:
            st.markdown("##### 💬 2. Kualitatif: Chain-of-Thought Rationale (SumoPod LLM)")
            exp_text = res.get("explanation", "Tidak ada penjelasan yang dihasilkan.")
            conf_raw = res.get("confidence", "N/A")
            conf_val = f"{conf_raw:.2f}" if isinstance(conf_raw, (int, float)) else str(conf_raw)
            latency = res.get("latency_sec", 0)

            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 14px 18px; border-radius: 6px; min-height: 120px;">
                    <p style="font-size: 15px; line-height: 1.6; color: #1e293b; margin: 0;">
                        {exp_text}
                    </p>
                    <div style="margin-top: 10px; font-size: 13px; color: #64748b;">
                        <span>📊 Confidence: <strong>{conf_val}</strong></span> &nbsp;•&nbsp; 
                        <span>⏱️ Latency: <strong>{latency} detik</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # PENYELARASAN GANDA (DUAL-XAI ALIGNMENT)
        if explainer is not None and shap_res is not None:
            align_info = align_shap_with_cot(shap_res, exp_text)
            score = align_info["alignment_score"]
            is_al = align_info["is_aligned"]
            border_c = "#22c55e" if is_al else "#eab308"
            bg_c = "#f0fdf4" if is_al else "#fefce8"
            text_c = "#15803d" if is_al else "#854d0e"

            st.markdown(
                f"""
                <div style="margin-top: 14px; background-color: {bg_c}; border: 1px solid {border_c}; border-radius: 8px; padding: 12px 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; color: {text_c};">🛡️ Indeks Keselarasan Dual-XAI (SHAP ⟷ CoT): {score}%</span>
                        <span style="font-size: 12px; color: {text_c}; font-weight: 500;">
                            {"✅ Selaras Secara Kultural" if is_al else "ℹ️ Sebagian Selaras"}
                        </span>
                    </div>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: {text_c};">
                        {align_info['alignment_summary']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("🔍 Lihat Raw JSON Output"):
            st.json({k: v for k, v in res.items() if k not in ["comment", "timestamp"]})

    # History Section
    if st.session_state["single_history"]:
        st.markdown("---")
        st.markdown("### 📜 Riwayat Pengujian Sesi Ini")
        hist_df = pd.DataFrame(st.session_state["single_history"])
        display_cols = ["timestamp", "comment", "cyberbullying", "type", "emotion", "cultural_context", "implicit_level", "severity", "confidence", "explanation"]
        available_cols = [c for c in display_cols if c in hist_df.columns]
        
        st.dataframe(hist_df[available_cols], use_container_width=True, height=220)
        
        csv_data = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Unduh Riwayat Pengujian (CSV)",
            data=csv_data,
            file_name=f"single_test_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
