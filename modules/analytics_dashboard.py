import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import TAXONOMY
from data_manager.checkpoint_manager import list_existing_checkpoints
from data_manager.dataset_handler import load_dataset_file

def render_analytics_dashboard_page():
    st.markdown("## 📊 Dashboard Analitik & Statistik Riset")
    st.markdown("Visualisasi distribusi taksonomi multi-dimensi cyberbullying dan metrik agregat untuk penulisan bab hasil disertasi.")

    # 1. Dataset Selection
    current_df = st.session_state.get("latest_batch_df", None)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        sel_source = st.selectbox(
            "Pilih Dataset Analisis:",
            ["Dataset Batch Sesi Aktif", "Pilih dari Checkpoint", "Unggah Berkas CSV"]
        )

    with col2:
        if sel_source == "Pilih dari Checkpoint":
            ckpts = list_existing_checkpoints()
            if ckpts:
                c_names = [f["filename"] for f in ckpts]
                pick = st.selectbox("Pilih berkas checkpoint:", c_names)
                p_path = next(f["path"] for f in ckpts if f["filename"] == pick)
                if st.button("Muat untuk Analisis"):
                    loaded, err = load_dataset_file(p_path)
                    if not err:
                        current_df = loaded
            else:
                st.info("Belum ada file checkpoint.")
        elif sel_source == "Unggah Berkas CSV":
            u_csv = st.file_uploader("Unggah CSV hasil anotasi:", type=["csv"])
            if u_csv:
                loaded, err = load_dataset_file(u_csv)
                if not err:
                    current_df = loaded

    if current_df is None or "cyberbullying" not in current_df.columns:
        st.info("Silakan muat dataset yang telah memiliki kolom anotasi (`cyberbullying`, `type`, dll.).")
        return

    # Clean & standardize fields
    df = current_df.copy()
    df["cyberbullying"] = df["cyberbullying"].astype(str).str.upper().str.strip()

    st.markdown("---")
    
    # 2. Key High-level Metrics
    total_data = len(df)
    cb_count = (df["cyberbullying"] == "YES").sum()
    non_cb_count = (df["cyberbullying"] == "NO").sum()
    cb_pct = (cb_count / total_data * 100) if total_data > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Data Teranotasi", f"{total_data:,}")
    with k2:
        st.metric("Cyberbullying (YES)", f"{cb_count:,} ({cb_pct:.1f}%)")
    with k3:
        st.metric("Non-Cyberbullying (NO)", f"{non_cb_count:,} ({100 - cb_pct:.1f}%)")
    with k4:
        # High severity count
        if "severity" in df.columns:
            high_sev = (df["severity"].astype(str).str.upper() == "HIGH").sum()
            st.metric("Severity HIGH", f"{high_sev:,}")
        else:
            st.metric("Severity HIGH", "N/A")
    with k5:
        # Average Confidence Score
        if "confidence" in df.columns:
            num_conf = pd.to_numeric(df["confidence"], errors="coerce")
            avg_conf = num_conf.mean()
            st.metric("Rerata Confidence", f"{avg_conf:.2f}" if pd.notna(avg_conf) else "N/A")
        else:
            st.metric("Rerata Confidence", "N/A")

    # 3. Main Visualizations
    st.markdown("### 📈 Visualisasi Taksonomi Multi-Dimensi")
    
    tab_dist, tab_cross, tab_table = st.tabs(["📊 Distribusi Taksonomi", "🔗 Analisis Silang (Crosstab)", "📑 Ringkasan Tabel Riset"])

    with tab_dist:
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            # Cyberbullying Proportion
            cb_df = df["cyberbullying"].value_counts().reset_index()
            cb_df.columns = ["Label", "Jumlah"]
            fig_cb = px.pie(
                cb_df, values="Jumlah", names="Label",
                title="Proporsi Cyberbullying (YES vs NO)",
                color="Label",
                color_discrete_map={"YES": "#e53e3e", "NO": "#38a169"},
                hole=0.4
            )
            fig_cb.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_cb, use_container_width=True)

        with r1_c2:
            # Type Distribution for YES
            if "type" in df.columns:
                cb_only = df[df["cyberbullying"] == "YES"]
                type_counts = cb_only["type"].value_counts().reset_index()
                type_counts.columns = ["Kode Tipe", "Frekuensi"]
                # Map full label
                type_counts["Deskripsi"] = type_counts["Kode Tipe"].map(TAXONOMY["type"]).fillna(type_counts["Kode Tipe"])
                fig_type = px.bar(
                    type_counts, x="Frekuensi", y="Deskripsi",
                    orientation='h',
                    title="Distribusi Tipe Cyberbullying",
                    color="Frekuensi",
                    color_continuous_scale="Reds"
                )
                fig_type.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_type, use_container_width=True)

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            # Cultural Context
            if "cultural_context" in df.columns:
                cult_counts = df[df["cyberbullying"] == "YES"]["cultural_context"].value_counts().reset_index()
                cult_counts.columns = ["Kode", "Frekuensi"]
                cult_counts["Konteks"] = cult_counts["Kode"].map(TAXONOMY["cultural_context"]).fillna(cult_counts["Kode"])
                fig_cult = px.bar(
                    cult_counts, x="Konteks", y="Frekuensi",
                    title="Konteks Kultural Terkait Cyberbullying",
                    color="Frekuensi",
                    color_continuous_scale="Purples"
                )
                st.plotly_chart(fig_cult, use_container_width=True)

        with r2_c2:
            # Implicit Level
            if "implicit_level" in df.columns:
                imp_counts = df[df["cyberbullying"] == "YES"]["implicit_level"].value_counts().reset_index()
                imp_counts.columns = ["Kode", "Frekuensi"]
                imp_counts["Tingkat"] = imp_counts["Kode"].map(TAXONOMY["implicit_level"]).fillna(imp_counts["Kode"])
                fig_imp = px.pie(
                    imp_counts, values="Frekuensi", names="Tingkat",
                    title="Tingkat Keterbukaan Ujaran (Explicit vs Implicit)",
                    color_discrete_sequence=px.colors.sequential.Teal
                )
                st.plotly_chart(fig_imp, use_container_width=True)

    with tab_cross:
        st.markdown("#### Matriks Silang (Cultural Context vs Severity)")
        if "cultural_context" in df.columns and "severity" in df.columns:
            cb_subset = df[df["cyberbullying"] == "YES"]
            ct = pd.crosstab(cb_subset["cultural_context"], cb_subset["severity"])
            fig_heat = px.imshow(
                ct,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Severity", y="Cultural Context", color="Frekuensi"),
                title="Heatmap: Hubungan Konteks Kultural dan Tingkat Keparahan",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
            st.markdown("#### Matriks Silang (Type vs Implicit Level)")
            if "type" in df.columns and "implicit_level" in df.columns:
                ct2 = pd.crosstab(cb_subset["type"], cb_subset["implicit_level"])
                fig_heat2 = px.imshow(
                    ct2,
                    text_auto=True,
                    aspect="auto",
                    labels=dict(x="Implicit Level", y="Cyberbullying Type", color="Frekuensi"),
                    title="Heatmap: Tipe Cyberbullying vs Tingkat Keterbukaan",
                    color_continuous_scale="Plasma"
                )
                st.plotly_chart(fig_heat2, use_container_width=True)

    with tab_table:
        st.markdown("#### 📄 Tabel Ringkasan untuk Naskah Disertasi")
        st.markdown("Tabel frekuensi dan persentase yang siap disalin ke bab laporan penelitian:")
        
        # Prepare aggregated summary
        summary_rows = []
        for cat in ["type", "emotion", "cultural_context", "implicit_level", "severity"]:
            if cat in df.columns:
                vc = df[cat].value_counts()
                for val, cnt in vc.items():
                    pct = (cnt / total_data) * 100
                    summary_rows.append({
                        "Dimensi": cat.capitalize(),
                        "Nilai / Kategori": str(val),
                        "Frekuensi (N)": cnt,
                        "Persentase (%)": f"{pct:.2f}%"
                    })
        
        sum_df = pd.DataFrame(summary_rows)
        st.dataframe(sum_df, use_container_width=True, height=350)
        
        # LaTeX Generator
        latex_str = sum_df.to_latex(index=False)
        with st.expander("📝 Salin Kode LaTeX untuk Bab Disertasi"):
            st.code(latex_str, language="latex")
