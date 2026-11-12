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
5. Ekstraksi Vektor Probabilitas Taksonomi LLM (33 Dimensi) dari Hasil Anotasi LLaMA-3.
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
# 6. EKSTRAKSI VEKTOR PENGETAHUAN LLM (33-DIMENSI) DARI SUMOPOD API
# =============================================================================

def extract_taxonomy_probability_vector(annotation_dict: Dict[str, Any]) -> np.ndarray:
    """
    Mengekstrak vektor probabilitas taksonomi 33-dimensi dari anotasi LLaMA-3 (SumoPod):
    - [0:2]   : Cyberbullying Binary ([NO, YES])
    - [2:11]  : Bullying Type (9 classes)
    - [11:17] : Emotion (6 classes)
    - [17:25] : Cultural Context (8 classes)
    - [25:28] : Implicitness Level (3 classes)
    - [28:32] : Severity Level (4 classes)
    - [32]    : Normalized Confidence Scalar [0.0, 1.0]
    
    Total Dimensi: 33
    """
    vec = []
    conf = float(annotation_dict.get("confidence", 0.90))
    conf = max(0.50, min(1.0, conf))

    # 1. Cyberbullying (Binary) -> 2 dims
    cb_val = str(annotation_dict.get("cyberbullying", "")).upper().strip()
    if cb_val == "YES":
        vec.extend([1.0 - conf, conf])
    elif cb_val == "NO":
        vec.extend([conf, 1.0 - conf])
    else:
        vec.extend([0.50, 0.50])

    # 2. Type (9 dims)
    t_val = str(annotation_dict.get("type", "")).upper().strip()
    t_list = TAXONOMY_VECTOR_MAP["type"]
    t_vec = [0.0] * len(t_list)
    if t_val in t_list:
        idx = t_list.index(t_val)
        t_vec[idx] = conf
        # Sebar sisa probabilitas ke kelas lain
        remainder = (1.0 - conf) / (len(t_list) - 1)
        for i in range(len(t_list)):
            if i != idx:
                t_vec[i] = remainder
    else:
        # Default to NONE
        t_vec[t_list.index("NONE")] = conf
    vec.extend(t_vec)

    # 3. Emotion (6 dims)
    e_val = str(annotation_dict.get("emotion", "")).upper().strip()
    e_list = TAXONOMY_VECTOR_MAP["emotion"]
    e_vec = [0.0] * len(e_list)
    if e_val in e_list:
        idx = e_list.index(e_val)
        e_vec[idx] = conf
        remainder = (1.0 - conf) / (len(e_list) - 1)
        for i in range(len(e_list)):
            if i != idx:
                e_vec[i] = remainder
    else:
        e_vec[e_list.index("NEU")] = conf
    vec.extend(e_vec)

    # 4. Cultural Context (8 dims)
    c_val = str(annotation_dict.get("cultural_context", "")).upper().strip()
    c_list = TAXONOMY_VECTOR_MAP["cultural_context"]
    c_vec = [0.0] * len(c_list)
    if c_val in c_list:
        idx = c_list.index(c_val)
        c_vec[idx] = conf
        remainder = (1.0 - conf) / (len(c_list) - 1)
        for i in range(len(c_list)):
            if i != idx:
                c_vec[i] = remainder
    else:
        c_vec[c_list.index("NONE")] = conf
    vec.extend(c_vec)

    # 5. Implicit Level (3 dims)
    i_val = str(annotation_dict.get("implicit_level", "")).upper().strip()
    i_list = TAXONOMY_VECTOR_MAP["implicit_level"]
    i_vec = [0.0] * len(i_list)
    if i_val in i_list:
        idx = i_list.index(i_val)
        i_vec[idx] = conf
        remainder = (1.0 - conf) / (len(i_list) - 1)
        for i in range(len(i_list)):
            if i != idx:
                i_vec[i] = remainder
    else:
        i_vec[i_list.index("E0")] = conf
    vec.extend(i_vec)

    # 6. Severity (4 dims)
    s_val = str(annotation_dict.get("severity", "")).upper().strip()
    s_list = TAXONOMY_VECTOR_MAP["severity"]
    s_vec = [0.0] * len(s_list)
    if s_val in s_list:
        idx = s_list.index(s_val)
        s_vec[idx] = conf
        remainder = (1.0 - conf) / (len(s_list) - 1)
        for i in range(len(s_list)):
            if i != idx:
                s_vec[i] = remainder
    else:
        s_vec[s_list.index("NONE")] = conf
    vec.extend(s_vec)

    # 7. Confidence Scalar (1 dim)
    vec.append(conf)

    return np.array(vec, dtype=np.float32)


def get_taxonomy_vector_feature_names() -> List[str]:
    """Mengembalikan daftar nama fitur untuk ke-33 dimensi vektor probabilitas taksonomi."""
    names = []
    for k, classes in TAXONOMY_VECTOR_MAP.items():
        for c in classes:
            names.append(f"{k}_{c}")
    names.append("confidence_score")
    return names


def format_structured_rationale(annotation_dict: Dict[str, Any]) -> str:
    """
    Memformat teks penalaran LLaMA-3 menjadi teks berstruktur ringkas
    untuk di-encode bersama teks komentar oleh Transformer Encoder:
    [CLS] komentar [SEP] rationale [SEP]
    """
    cult = annotation_dict.get("cultural_context", "NONE")
    emo = annotation_dict.get("emotion", "NEU")
    impl = annotation_dict.get("implicit_level", "E0")
    btype = annotation_dict.get("type", "NONE")
    explanation = str(annotation_dict.get("explanation", "")).strip()

    prefix = f"[Konteks: {cult} | Emosi: {emo} | Tipe: {btype} | Implisit: {impl}]"
    if explanation:
        return f"{prefix} {explanation}"
    return prefix
