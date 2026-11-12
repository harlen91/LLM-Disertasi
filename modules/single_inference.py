from datetime import datetime
import pandas as pd
import streamlit as st

from config import DEFAULT_SUMOPOD_MODEL, MODELS_DIR, SAMPLE_COMMENTS, TAXONOMY
from core.model_engine import run_single_annotation
from core.shap_explainer import HybridSHAPExplainer, align_shap_with_cot, compute_faithfulness_comprehensiveness
from data_manager.dataset_handler import sanitize_html_text


@st.cache_resource
def get_cached_shap_explainer():
    """Loads the real SHAP explainer (prioritizing HybridFusionModel if trained, else IndoBERTweet baseline)."""
    hybrid_weights = MODELS_DIR / "hybrid_fusion_indobertweet" / "best_hybrid_model.pt"
    baseline_dir = MODELS_DIR / "indobertweet_cyberbullying" / "best_model"
    try:
        if hybrid_weights.exists():
            return HybridSHAPExplainer(model_path=hybrid_weights, model_type="hybrid"), None
        if baseline_dir.exists() and ((baseline_dir / "model.safetensors").exists() or (baseline_dir / "pytorch_model.bin").exists()):
            return HybridSHAPExplainer(model_path=baseline_dir, model_type="baseline"), None
        return None, "Checkpoint model terlatih belum ditemukan di output/models/."
    except Exception as exc:
        return None, str(exc)


def get_full_label(category: str, code: str) -> str:
    """Helper to convert code to human-friendly description."""
    cat_dict = TAXONOMY.get(category, {})
    if isinstance(cat_dict, dict):
        return cat_dict.get(code, code)
    return code


def render_single_inference_page():
    st.markdown("## 🧪 Playground Anotasi & Explainability Inspector")
    st.markdown(
        "Uji komentar individual secara instan untuk memeriksa klasifikasi multi-dimensi dan penalaran "
        "(*chain-of-thought explanation*) dari model LLM via **SumoPod AI Gateway** serta atribusi SHAP."
    )

    engine = st.session_state.get("active_engine")
    is_mock = getattr(engine, "is_mock", True) if engine else True

    if is_mock:
        st.warning(
            "⚡ **Mode Simulasi / Mock Fallback Aktif**: Hasil inferensi dihasilkan secara heuristik lokal untuk demonstrasi UI, "
            "bukan output SumoPod LLM sesungguhnya."
        )
    else:
        st.caption(f"🌐 Mode saat ini: **SumoPod AI Gateway** aktif (`{getattr(engine, 'model_name', DEFAULT_SUMOPOD_MODEL)}`).")

    if "single_history" not in st.session_state:
        st.session_state["single_history"] = []

    col_sel, col_clear = st.columns([4, 1])
    with col_sel:
        selected_sample = st.selectbox(
            "Pilih contoh komentar dari dataset riset:",
            ["-- Ketik komentar kustom di bawah --"] + SAMPLE_COMMENTS,
        )
    with col_clear:
        if st.button("🗑️ Bersihkan Riwayat", use_container_width=True):
            st.session_state["single_history"] = []
            st.rerun()

    default_text = "" if selected_sample.startswith("--") else selected_sample
    user_text = st.text_area(
        "Komentar Bahasa Indonesia yang diuji:",
        value=default_text,
        placeholder="Contoh: 'makannya segentong buset' atau ketik komentar apapun...",
        height=100,
    )

    btn_col1, _ = st.columns([1, 3])
    with btn_col1:
        analyze_btn = st.button("🔍 Jalankan Anotasi", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_text.strip():
            st.warning("Silakan masukkan teks komentar terlebih dahulu.")
            return

        with st.spinner("Menganalisis teks menggunakan taksonomi riset..."):
            try:
                temp = st.session_state.get("temperature", 0.1)
                max_tok = st.session_state.get("max_tokens", 220)
                sys_prompt = st.session_state.get("custom_system_prompt", None)

                result = run_single_annotation(
                    engine=engine,
                    text=user_text.strip(),
                    system_prompt=sys_prompt,
                    max_new_tokens=max_tok,
                    temperature=temp,
                )

                record = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "comment": user_text.strip(),
                    **result,
                }
                st.session_state["single_history"].insert(0, record)
                st.session_state["last_result"] = record
            except Exception as exc:
                st.error(f"❌ Gagal menjalankan anotasi: {exc}")
                return

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        st.markdown("---")
        st.markdown("### 📋 Hasil Anotasi & Penjelasan Model")

        cb_val = str(res.get("cyberbullying", "UNKNOWN")).upper()
        is_cb = cb_val == "YES"
        verdict_color = "#e53e3e" if is_cb else ("#38a169" if cb_val == "NO" else "#d97706")
        verdict_bg = "#fff5f5" if is_cb else ("#f0fff4" if cb_val == "NO" else "#fffbeb")
        verdict_border = "#feb2b2" if is_cb else ("#9ae6b4" if cb_val == "NO" else "#fcd34d")
        verdict_icon = (
            "⚠️ TERDETEKSI CYBERBULLYING"
            if is_cb
            else ("✅ BUKAN CYBERBULLYING" if cb_val == "NO" else "❔ TIDAK DIKETAHUI / PARSE ERROR")
        )
        source_badge = (
            "🧪 [SIMULASI MOCK]" if res.get("is_mock") else f"🌐 [SUMOPOD API: {sanitize_html_text(res.get('llm_model', DEFAULT_SUMOPOD_MODEL))}]"
        )
        safe_comment = sanitize_html_text(res.get("comment", ""))

        st.markdown(
            f"""
            <div style="background-color: {verdict_bg}; border: 2px solid {verdict_border}; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="color: {verdict_color}; margin: 0 0 6px 0; font-size: 22px;">{verdict_icon}</h3>
                    <span style="font-size: 12px; font-weight: 600; color: #475569;">{source_badge}</span>
                </div>
                <p style="margin: 0; color: #4a5568; font-size: 15px;"><strong>Teks Diuji:</strong> "{safe_comment}"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, key, title in [
            (c1, "type", "Tipe (Type)"),
            (c2, "emotion", "Emosi (Emotion)"),
            (c3, "cultural_context", "Konteks Kultural"),
            (c4, "implicit_level", "Keterbukaan"),
            (c5, "severity", "Tingkat Keparahan"),
        ]:
            with col:
                code = str(res.get(key, "UNKNOWN"))
                st.metric(title, code, help=get_full_label(key, code))
                st.caption(f"_{get_full_label(key, code)}_")

        st.markdown("#### 🔬 Dual-Modality Explainability Framework (Real SHAP + CoT Rationale)")
        st.caption(
            "Menampilkan kontribusi Shapley pada modalitas teks (Stream A) dan fitur pengetahuan LLM 33D (Stream B) secara terpisah, beserta rationale SumoPod LLM."
        )

        col_xai_left, col_xai_right = st.columns([1, 1])
        explainer, exp_err = get_cached_shap_explainer()
        dual_shap = None

        with col_xai_left:
            st.markdown("##### 📊 1. Kuantitatif: Token SHAP Attribution")
            if explainer is not None:
                with st.spinner("Menghitung Shapley values (library shap)..."):
                    dual_shap = explainer.explain_dual_modalities(
                        text=res.get("comment", ""),
                        annotation_dict=res if not res.get("is_mock") else None,
                    )
                    shap_res = dual_shap["textual_shap"]
                    st.markdown(shap_res["html_heatmap"], unsafe_allow_html=True)
                    st.caption(f"ℹ️ Ruang lingkup: {shap_res.get('conditional_scope', '')}")

                    pos_tokens = shap_res.get("top_positive_tokens", [])
                    neg_tokens = shap_res.get("top_negative_tokens", [])
                    if pos_tokens:
                        pos_str = ", ".join([f"`{t['token']}` (+{t['shap_value']:.3f})" for t in pos_tokens])
                        st.markdown(f"🔴 **Kata Pemicu Cyberbullying**: {pos_str}")
                    if neg_tokens:
                        neg_str = ", ".join([f"`{t['token']}` ({t['shap_value']:.3f})" for t in neg_tokens])
                        st.markdown(f"🟢 **Kata Peredam / Netral**: {neg_str}")

                    llm_shap = dual_shap.get("llm_vector_shap", {})
                    if llm_shap.get("available"):
                        with st.expander("🧠 Atribusi Modalitas Stream B (33D Teacher Features)"):
                            st.caption(llm_shap.get("conditional_scope", ""))
                            top_pos_f = llm_shap.get("top_positive_features", [])
                            if top_pos_f:
                                st.dataframe(pd.DataFrame(top_pos_f), use_container_width=True, hide_index=True)
            else:
                st.info(f"SHAP lokal tidak aktif: {exp_err}")

        with col_xai_right:
            st.markdown("##### 💬 2. Kualitatif: Chain-of-Thought Rationale (SumoPod LLM)")
            exp_text = str(res.get("explanation", "Tidak ada penjelasan yang dihasilkan."))
            safe_exp_html = sanitize_html_text(exp_text)
            conf_raw = res.get("confidence", None)
            conf_val = f"{conf_raw:.2f}" if isinstance(conf_raw, (int, float)) else "N/A"
            latency = res.get("latency_sec", 0)
            prompt_ver = sanitize_html_text(res.get("prompt_version", "legacy"))

            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 14px 18px; border-radius: 6px; min-height: 120px;">
                    <p style="font-size: 15px; line-height: 1.6; color: #1e293b; margin: 0;">
                        {safe_exp_html}
                    </p>
                    <div style="margin-top: 10px; font-size: 13px; color: #64748b;">
                        <span>📊 Confidence: <strong>{conf_val}</strong></span> &nbsp;•&nbsp;
                        <span>⏱️ Latency: <strong>{latency} detik</strong></span> &nbsp;•&nbsp;
                        <span>🏷️ Prompt: <code>{prompt_ver}</code></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if explainer is not None and dual_shap is not None:
            shap_res = dual_shap["textual_shap"]
            comp_val = compute_faithfulness_comprehensiveness(explainer, res.get("comment", ""), shap_res)
            align_info = align_shap_with_cot(shap_res, exp_text, comprehensiveness_score=comp_val)
            score = align_info["alignment_score"]
            is_al = align_info["is_aligned"]
            border_c = "#22c55e" if is_al else "#eab308"
            bg_c = "#f0fdf4" if is_al else "#fefce8"
            text_c = "#15803d" if is_al else "#854d0e"
            safe_summary = sanitize_html_text(align_info["alignment_summary"])
            comp_str = f"{comp_val:+.4f}" if comp_val is not None else "N/A"

            st.markdown(
                f"""
                <div style="margin-top: 14px; background-color: {bg_c}; border: 1px solid {border_c}; border-radius: 8px; padding: 12px 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; color: {text_c};">🛡️ Diagnostik Lexical Overlap: {score}% | Faithfulness (Δp): {comp_str}</span>
                        <span style="font-size: 12px; color: {text_c}; font-weight: 500;">
                            {"✅ Overlap ≥ 50%" if is_al else "ℹ️ Overlap Parsial"}
                        </span>
                    </div>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: {text_c};">
                        {safe_summary}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("🔍 Lihat Raw JSON Output & Metadata"):
            st.json({k: v for k, v in res.items() if k not in ["comment", "timestamp"]})

    if st.session_state["single_history"]:
        st.markdown("---")
        st.markdown("### 📜 Riwayat Pengujian Sesi Ini")
        hist_df = pd.DataFrame(st.session_state["single_history"])
        display_cols = [
            "timestamp", "annotation_source", "comment", "cyberbullying", "type",
            "emotion", "cultural_context", "implicit_level", "severity", "confidence", "explanation",
        ]
        available_cols = [c for c in display_cols if c in hist_df.columns]
        st.dataframe(hist_df[available_cols], use_container_width=True, height=220)

        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Unduh Riwayat Pengujian (CSV)",
            data=csv_data,
            file_name=f"single_test_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
