import os
from pathlib import Path

# Base Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# SumoPod Gateway Configurations
DEFAULT_SUMOPOD_BASE_URL = "https://api.sumopod.com/v1"
DEFAULT_SUMOPOD_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SUMOPOD_POPULAR_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-7B-Instruct",
    "gpt-4o-mini"
]

# Default Dataset Path
DEFAULT_DATASET_PATH = PROJECT_ROOT / "Dataset-Research.csv"

# Output Paths
OUTPUT_DIR = APP_DIR / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
FINAL_DIR = OUTPUT_DIR / "final"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Taxonomy Reference
TAXONOMY = {
    "cyberbullying": ["YES", "NO"],
    "type": {
        "INS": "Insult (Penghinaan)",
        "BSH": "Body Shaming (Cercaan Fisik)",
        "SHM": "Social Humiliation (Pelecehan Sosial)",
        "SAR": "Sarcasm (Sarkasme / Sindiran)",
        "THR": "Threat (Ancaman)",
        "DIS": "Discrimination (Diskriminasi / SARA)",
        "SEX": "Sexual Harassment (Pelecehan Seksual)",
        "EXC": "Social Exclusion (Pengucilan Sosial)",
        "NONE": "None / Non-Cyberbullying"
    },
    "emotion": {
        "ANG": "Anger (Kemarahan)",
        "DIS": "Disgust (Jijik / Muak)",
        "HUM": "Humiliation (Malu / Dipermalukan)",
        "MOCK": "Mockery (Ejekan / Olok-olok)",
        "SAD": "Sadness (Kesedihan)",
        "NEU": "Neutral (Netral)"
    },
    "cultural_context": {
        "APP": "Appearance (Penampilan Fisik)",
        "INT": "Intelligence (Kecerdasan / Kemampuan)",
        "STA": "Social Status (Status Sosial / Ekonomi)",
        "FAM": "Family Background (Latar Belakang Keluarga)",
        "EDU": "Education (Pendidikan)",
        "REG": "Regional Identity (Asal Daerah / Suku)",
        "GEN": "Gender (Gender / Orientasi)",
        "NONE": "None / Tidak Ada Konteks Kultural Spesifik"
    },
    "implicit_level": {
        "E0": "Explicit (Sangat Terang-terangan)",
        "E1": "Semi Implicit (Semi Tersirat)",
        "E2": "Implicit (Tersirat / Kontekstual)"
    },
    "severity": {
        "LOW": "Low (Rendah / Ringan)",
        "MEDIUM": "Medium (Sedang)",
        "HIGH": "High (Tinggi / Berat)",
        "NONE": "None / Non-Cyberbullying"
    }
}

# Example comments for quick testing
SAMPLE_COMMENTS = [
    "makannya segentong buset",
    "mirip ursula di little mermaid",
    "stylenya dia tuh gmn sik??",
    "Beli alat gym aja kak",
    "dari mulutnya udh bisa bikin orang gulung tikar??????",
    "zar, gaada kepikiran mau pakai behel ??",
    "Hama Tiktok",
    "mandi minimal",
    "bayu terlihat tidak di urus",
    "Semangat terus ya kak, jangan dengarkan komentar jahat netizen!"
]
