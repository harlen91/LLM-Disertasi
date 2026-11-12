#!/bin/bash
# Script untuk menjalankan IICD Streamlit Annotation Studio
cd "$(dirname "$0")"

echo "=========================================================="
echo "🛡️ Memulai IICD LLM-Augmented Explainable Annotation Studio"
echo "=========================================================="

python3 -m streamlit run app.py --server.port 8501 --server.headless false
