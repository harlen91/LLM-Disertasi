from pathlib import Path

# Base Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# SumoPod Gateway Configurations
DEFAULT_SUMOPOD_BASE_URL = "https://api.sumopod.com/v1"
DEFAULT_SUMOPOD_MODEL = "deepseek-ai/DeepSeek-V3"
SUMOPOD_POPULAR_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-v4.1-flash:netra",
    "gpt-4o-mini",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]

# Default Dataset Path
DEFAULT_DATASET_PATH = PROJECT_ROOT / "Dataset-Research.csv"

# Output Paths
OUTPUT_DIR = APP_DIR / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
FINAL_DIR = OUTPUT_DIR / "final"
MODELS_DIR = OUTPUT_DIR / "models"
EVAL_REPORT_DIR = OUTPUT_DIR / "evaluation_report"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained Backbone & Research Versioning
INDOBERTWEET_PRETRAINED_MODEL = "indolem/indobertweet-base-uncased"
DEFAULT_RANDOM_SEED = 42
APP_VERSION = "2.0.0"
MODEL_MANIFEST_FILENAME = "model_manifest.json"

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
