"""
core/data_understanding.py
--------------------------
Modul Lapisan Pemahaman Data (Data Understanding Layer) & Ekstraksi Vektor
Representasi Pengetahuan LLM (SumoPod API) untuk Deteksi Cyberbullying Hibrida.

Fitur Utama:
1. Normalisasi Leksikon Slang / Bahasa Gaul Media Sosial Indonesia.
2. Demultiplexing & Semantik Emoji Sarkas/Penghinaan ke Representasi Teks.
3. Deteksi & Tagging Dialek Kedaerahan (Jawa, Sunda, Batak, Jaksel).
4. Reduksi Elongasi Karakter & Pembersihan Mention Pengguna.
5. Ekstraksi Vektor Probabilitas Taksonomi LLM (33 Dimensi) dari Hasil Anotasi SumoPod API.
6. Pemformatan Teks Penalaran Terstruktur (Structured CoT Rationale) untuk Dual-Sequence Transformer.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

# =============================================================================
# 1. LEKSIKON SLANG & NORMALISASI BAHASA GAUL INDONESIA
# =============================================================================

SLANG_DICTIONARY: Dict[str, str] = {
    # Kata tugas & singkatan umum
    "yg": "yang",
    "dgn": "dengan",
    "bgt": "banget",
    "bgtt": "banget",
    "kmrn": "kemarin",
    "kemaren": "kemarin",
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "kagak": "tidak",
    "kgk": "tidak",
    "tdk": "tidak",
    "bkn": "bukan",
    "tp": "tetapi",
    "tpi": "tetapi",
    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "dr": "dari",
    "dri": "dari",
    "sm": "sama",
    "sma": "sama",
    "utk": "untuk",
    "untuk": "untuk",
    "sy": "saya",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "elu": "kamu",
    "elo": "kamu",
    "krn": "karena",
    "karna": "karena",
    "jd": "jadi",
    "jdi": "jadi",
    "bener": "benar",
    "beneran": "benar",
    "dapet": "dapat",
    "dpt": "dapat",
    "bs": "bisa",
    "bisaa": "bisa",
    "blm": "belum",
    "belum": "belum",
    "gmn": "bagaimana",
    "gimana": "bagaimana",
    "knp": "kenapa",
    "kenape": "kenapa",
    "org": "orang",
    "orng": "orang",
    "bray": "kawan",
    "bro": "kawan",
    "cuy": "kawan",
    "bg": "abang",
    "bang": "abang",
    "kak": "kakak",
    "mba": "mbak",
    "mas": "mas",
    # Slang tren / gaul kontemporer
    "tbl": "takut banget",
    "ygy": "ya guys ya",
    "kuy": "ayo",
    "sans": "santai",
    "sabi": "bisa",
    "gercep": "gerak cepat",
    "mager": "malas gerak",
    "baper": "bawa perasaan",
    "pansos": "panjat sosial",
    "fomo": "takut ketinggalan",
    "cringe": "menjijikkan",
    "pick me": "mencari perhatian",
    "pikmi": "mencari perhatian",
    "glow up": "berubah menarik",
    "glowup": "berubah menarik",
    "red flag": "tanda bahaya",
    "redflag": "tanda bahaya",
    # Singkatan makian / celaan yang distandarkan
    "bego": "bodoh",
    "tolol": "bodoh",
    "idiot": "bodoh",
    "pekok": "bodoh",
    "bloon": "bodoh",
    "oon": "bodoh"
}

# =============================================================================
# 2. DEMULTIPLEXING SEMANTIK EMOJI (SARKASME & AFEKSI)
# =============================================================================

EMOJI_SEMANTICS: Dict[str, str] = {
    "🤡": " [emoji_badut_mengejek] ",
    "🗿": " [emoji_batu_sinis] ",
    "💀": " [emoji_tengkorak_mati] ",
    "☠️": " [emoji_tengkorak_racun] ",
    "🤮": " [emoji_muntah_jijik] ",
    "🤢": " [emoji_mual_muak] ",
    "💩": " [emoji_kotoran_hinaan] ",
    "🐷": " [emoji_babi_hinaan] ",
    "🐽": " [emoji_babi_hinaan] ",
    "🐶": " [emoji_anjing_hinaan] ",
    "🐕": " [emoji_anjing_hinaan] ",
    "🐒": " [emoji_monyet_hinaan] ",
    "🐵": " [emoji_monyet_hinaan] ",
    "🖕": " [emoji_jari_tengah] ",
    "🤬": " [emoji_marah_mengumpat] ",
    "😡": " [emoji_marah] ",
    "😠": " [emoji_kesal] ",
    "😏": " [emoji_senyum_sinis] ",
    "🙄": " [emoji_memutar_mata_meremehkan] ",
    "😒": " [emoji_tidak_senang] ",
    "😂": " [emoji_tertawa_menertawakan] ",
    "🤣": " [emoji_tertawa_terbahak] ",
    "😭": " [emoji_menangis] ",
    "🥺": " [emoji_memohon] ",
    "🙏": " [emoji_hormat_terima_kasih] ",
    "❤️": " [emoji_cinta] ",
    "🔥": " [emoji_api_semangat] ",
    "👍": " [emoji_jempol_setuju] "
}

# =============================================================================
# 3. INDIKATOR DIALEK KEDAERAHAN INDONESIA
# =============================================================================

DIALECT_LEXICON: Dict[str, List[str]] = {
    "JAWA": [
        "jancuk", "jancok", "asu", "pekok", "matamu", "ndasmu", "raimu",
        "bajingan", "bocah", "rek", "cuk", "cok", "karepmu", "monggo", "opo",
        "piye", "wis", "wes", "ora", "iso", "ndro", "dancuk", "ndeso"
    ],
    "SUNDA": [
        "belegug", "sia", "aing", "maneh", "goblog", "geuleuh", "kumaha",
        "teuing", "euy", "atuh", "punten", "naon", "kunaon", "kehed"
    ],
    "BATAK": [
        "bodat", "poda", "hosa", "begu", "horas", "dang", "songon", "lae"
    ],
    "JAKSEL": [
        "literally", "which is", "basically", "honestly", "prefer", "anxious",
        "burnout", "mental health", "toxic", "circle", "overwhelmed"
    ]
}

# =============================================================================
# 4. SKEMA TAKSONOMI & DIMENSI VEKTOR PENGETAHUAN LLM (33 DIMENSI)
# =============================================================================

TAXONOMY_VECTOR_MAP = {
    "cyberbullying": ["NO", "YES"],                     # 2 dimensi
    "type": [                                           # 9 dimensi
        "INS", "BSH", "SHM", "SAR", "THR", "DIS", "SEX", "EXC", "NONE"
    ],
    "emotion": [                                        # 6 dimensi
        "ANG", "DIS", "HUM", "MOCK", "SAD", "NEU"
    ],
    "cultural_context": [                               # 8 dimensi
        "APP", "INT", "STA", "FAM", "EDU", "REG", "GEN", "NONE"
    ],
    "implicit_level": [                                 # 3 dimensi
        "E0", "E1", "E2"
    ],
    "severity": [                                       # 4 dimensi
        "LOW", "MEDIUM", "HIGH", "NONE"
    ]
}

TOTAL_VECTOR_DIM = 2 + 9 + 6 + 8 + 3 + 4 + 1  # 33 Dimensi (32 categorical + 1 scalar confidence)


# =============================================================================
# 5. FUNGSI-FUNGSI UTAMA PREPROCESSING & PEMAHAMAN DATA
# =============================================================================

def normalize_slang(text: str) -> str:
    """Mengganti singkatan dan slang informal dengan bentuk standarnya."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    words = text.split()
    normalized_words = [SLANG_DICTIONARY.get(w.lower(), w) for w in words]
    return " ".join(normalized_words)


def demultiplex_emojis(text: str) -> str:
    """Menerjemahkan emoji ekspresif/sarkastik menjadi token semantik teks."""
    if not isinstance(text, str):
        return ""
    
    for em, sem in EMOJI_SEMANTICS.items():
        if em in text:
            text = text.replace(em, sem)
    return text


def detect_dialects(text: str) -> List[str]:
    """Mendeteksi apakah kalimat memuat kata serapan dialek kedaerahan."""
    if not isinstance(text, str):
        return []
    
    detected = []
    text_lower = text.lower()
    for dialect, tokens in DIALECT_LEXICON.items():
        for tok in tokens:
            if re.search(rf"\b{re.escape(tok)}\b", text_lower):
                detected.append(dialect)
                break
    return detected


def clean_and_normalize_social_text(
    text: str,
    handle_slang: bool = True,
    handle_emojis: bool = True,
    reduce_elongation: bool = True
) -> str:
    """
    Pipeline pembersihan menyeluruh untuk komentar media sosial Indonesia:
    1. Konversi emoji semantik.
    2. Hapus URL/tautan.
    3. Hapus awalan username '@user:'.
    4. Standarisasi mention tersisa '@user'.
    5. Reduksi elongasi karakter berulang (> 2 menjadi 2).
    6. Normalisasi slang kata leksikal.
    7. Pembersihan whitespace ganda.
    """
    if not isinstance(text, str):
        return ""

    # 1. Emoji Demultiplexing
    if handle_emojis:
        text = demultiplex_emojis(text)

    # 2. Hapus URL web
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Hapus awalan username percakapan seperti '@human:' atau '@User Name:'
    text = re.sub(r'^@[\w\s\.\-]{1,35}:\s*', '', text)

    # 4. Standardisasi mention tersisa
    text = re.sub(r'@\w+', '@user', text)

    # 5. Reduksi elongasi huruf berulang
    if reduce_elongation:
        text = re.sub(r'([a-zA-Z0-9])\1{2,}', r'\1\1', text)

    # 6. Normalisasi slang
    if handle_slang:
        text = normalize_slang(text)

    # 7. Whitespace cleanup
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =============================================================================
# 6. EKSTRAKSI TEACHER ANNOTATION FEATURES (33-DIMENSI) DARI SUMOPOD LLM
# =============================================================================

TEACHER_FEATURE_SCHEMA = [
    {"dim_start": 0, "dim_end": 2, "task": "cyberbullying", "classes": TAXONOMY_VECTOR_MAP["cyberbullying"], "origin": "LLM teacher binary prediction (or actual logprobs if provided)"},
    {"dim_start": 2, "dim_end": 11, "task": "type", "classes": TAXONOMY_VECTOR_MAP["type"], "origin": "LLM teacher bullying type prediction (or actual logprobs if provided)"},
    {"dim_start": 11, "dim_end": 17, "task": "emotion", "classes": TAXONOMY_VECTOR_MAP["emotion"], "origin": "LLM teacher emotion prediction (or actual logprobs if provided)"},
    {"dim_start": 17, "dim_end": 25, "task": "cultural_context", "classes": TAXONOMY_VECTOR_MAP["cultural_context"], "origin": "LLM teacher cultural context prediction (or actual logprobs if provided)"},
    {"dim_start": 25, "dim_end": 28, "task": "implicit_level", "classes": TAXONOMY_VECTOR_MAP["implicit_level"], "origin": "LLM teacher implicitness level prediction (or actual logprobs if provided)"},
    {"dim_start": 28, "dim_end": 32, "task": "severity", "classes": TAXONOMY_VECTOR_MAP["severity"], "origin": "LLM teacher severity prediction (or actual logprobs if provided)"},
    {"dim_start": 32, "dim_end": 33, "task": "confidence", "classes": ["scalar_confidence"], "origin": "LLM self-reported decimal confidence scalar in [0.0, 1.0] (0.0 if missing)"},
]


def _encode_task_teacher_scores(
    task_name: str,
    raw_label: Any,
    class_list: List[str],
    class_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[float]:
    """
    Encodes a single taxonomy dimension into numeric teacher scores.
    - If actual model probabilities (`class_probabilities[task_name]`) are supplied, uses them normalized.
    - Otherwise encodes a clean one-hot indicator (1.0 for predicted category, 0.0 elsewhere) without
      fabricating calibrated probabilities from self-reported confidence.
    - Missing/UNKNOWN labels remain an all-zero vector instead of defaulting to another valid class.
    """
    if isinstance(class_probabilities, dict) and isinstance(class_probabilities.get(task_name), dict):
        prob_map = class_probabilities[task_name]
        raw_probs = [max(0.0, float(prob_map.get(cls, 0.0))) for cls in class_list]
        total = sum(raw_probs)
        if total > 0.0:
            return [round(p / total, 6) for p in raw_probs]

    val = "" if raw_label is None else str(raw_label).upper().strip()
    vec = [0.0] * len(class_list)
    if val in class_list:
        vec[class_list.index(val)] = 1.0
    return vec


def extract_teacher_annotation_features(
    annotation_dict: Dict[str, Any],
    class_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    """
    Mengekstrak vektor 33-dimensi `teacher_scores` dari prediksi LLM SumoPod (BUKAN dari ground truth manusia):
    - [0:2]   : teacher_cyberbullying ([NO, YES])
    - [2:11]  : teacher_type (9 classes)
    - [11:17] : teacher_emotion (6 classes)
    - [17:25] : teacher_cultural_context (8 classes)
    - [25:28] : teacher_implicit_level (3 classes)
    - [28:32] : teacher_severity (4 classes)
    - [32]    : teacher_confidence_score scalar [0.0, 1.0] (0.0 jika tidak tersedia)

    Catatan Metodologis:
    - Apabila API mengembalikan probabilitas/logprobs aktual (`class_probabilities`), nilai aktual digunakan.
    - Apabila hanya tersedia kategori diskret + confidence scalar, fitur kategori dikodekan sebagai
      indikator teacher scores biner (one-hot) dan confidence diletakkan pada dimensi ke-32 tanpa
      mengklaim sebaran buatan sebagai probabilitas terkalibrasi.
    - Nilai UNKNOWN/missing dikodekan sebagai nol (zero vector).
    """
    # Prefer explicit `llm_*` keys if present so ground-truth columns are never touched
    def _pick(field: str) -> Any:
        llm_key = f"llm_{field}"
        return annotation_dict.get(llm_key, annotation_dict.get(field))

    raw_conf = _pick("confidence")
    try:
        conf = 0.0 if raw_conf is None or (isinstance(raw_conf, float) and np.isnan(raw_conf)) else float(raw_conf)
    except (TypeError, ValueError):
        conf = 0.0
    conf = float(max(0.0, min(1.0, conf)))

    vec: List[float] = []
    for task_key in ["cyberbullying", "type", "emotion", "cultural_context", "implicit_level", "severity"]:
        vec.extend(
            _encode_task_teacher_scores(
                task_name=task_key,
                raw_label=_pick(task_key),
                class_list=TAXONOMY_VECTOR_MAP[task_key],
                class_probabilities=class_probabilities,
            )
        )
    vec.append(conf)
    arr = np.array(vec, dtype=np.float32)
    assert arr.shape == (TOTAL_VECTOR_DIM,), f"Expected {TOTAL_VECTOR_DIM} features, got {arr.shape}"
    return arr


def extract_taxonomy_probability_vector(
    annotation_dict: Dict[str, Any],
    class_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    """
    Backward-compatible alias for `extract_teacher_annotation_features()`.
    Mengembalikan teacher annotation features 33-dimensi (menggunakan probabilitas aktual jika
    disediakan melalui `class_probabilities`, atau indikator kategori diskret + scalar confidence
    tanpa mengklaim sebagai probabilitas terkalibrasi).
    """
    return extract_teacher_annotation_features(
        annotation_dict=annotation_dict,
        class_probabilities=class_probabilities,
    )


def get_teacher_feature_names() -> List[str]:
    """Mengembalikan daftar nama fitur eksplisit untuk ke-33 dimensi teacher annotation features."""
    names: List[str] = []
    for k, classes in TAXONOMY_VECTOR_MAP.items():
        for c in classes:
            names.append(f"teacher_{k}_{c}")
    names.append("teacher_confidence_score")
    return names


def get_taxonomy_vector_feature_names() -> List[str]:
    """Backward-compatible alias returning the 33-dim feature names."""
    return get_teacher_feature_names()


def get_teacher_feature_schema() -> List[Dict[str, Any]]:
    """Returns full documentation metadata for every dimension group of the 33-dim vector."""
    return TEACHER_FEATURE_SCHEMA


def format_structured_rationale(annotation_dict: Dict[str, Any]) -> str:
    """
    Memformat teks penjelasan LLM (explanation) untuk tampilan analitik / Dual-XAI.
    PERINGATAN METODOLOGIS: Jangan menyisipkan tag label target ([Konteks: ... | Tipe: ...]) ke dalam
    teks input pelatihan/evaluasi ketika label target berasal dari anotasi yang sama (mencegah target leakage).
    """
    cult = annotation_dict.get("llm_cultural_context", annotation_dict.get("cultural_context", "NONE"))
    emo = annotation_dict.get("llm_emotion", annotation_dict.get("emotion", "NEU"))
    impl = annotation_dict.get("llm_implicit_level", annotation_dict.get("implicit_level", "E0"))
    btype = annotation_dict.get("llm_type", annotation_dict.get("type", "NONE"))
    explanation = str(annotation_dict.get("llm_explanation", annotation_dict.get("explanation", ""))).strip()

    prefix = f"[Konteks: {cult} | Emosi: {emo} | Tipe: {btype} | Implisit: {impl}]"
    if explanation:
        return f"{prefix} {explanation}"
    return prefix
