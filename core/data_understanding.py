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
# 4. SKEMA TAKSONOMI TARGET & REPRESENTASI BUKTI KONTEKSTUAL LLM (33 DIMENSI)
# =============================================================================

# Peta label target untuk 6 Multi-Task Classification Heads (TIDAK disalin ke Stream B)
TAXONOMY_VECTOR_MAP = {
    "cyberbullying": ["NO", "YES"],                     # 2 kelas target
    "type": [                                           # 9 kelas target
        "INS", "BSH", "SHM", "SAR", "THR", "DIS", "SEX", "EXC", "NONE"
    ],
    "emotion": [                                        # 6 kelas target
        "ANG", "DIS", "HUM", "MOCK", "SAD", "NEU"
    ],
    "cultural_context": [                               # 8 kelas target
        "APP", "INT", "STA", "FAM", "EDU", "REG", "GEN", "NONE"
    ],
    "implicit_level": [                                 # 3 kelas target
        "E0", "E1", "E2"
    ],
    "severity": [                                       # 4 kelas target
        "LOW", "MEDIUM", "HIGH", "NONE"
    ]
}

# Stream B: Non-Leaky 33-Dim LLM Contextual Evidence Representation
# Menggambarkan bukti linguistik, budaya, emosi, pragmatik, dan kontekstual
# TANPA menyalin satu pun kolom label target klasifikasi.
CONTEXTUAL_EVIDENCE_GROUPS: Dict[str, List[str]] = {
    "linguistic": [  # Dims 0..6 (7D)
        "ling_derogatory_lexical",
        "ling_metaphor_simile",
        "ling_rhetorical_punct",
        "ling_elongation_emphasis",
        "ling_slang_register",
        "ling_second_person_target",
        "ling_negation_litotes",
    ],
    "cultural": [    # Dims 7..13 (7D) - Komponen augmentasi budaya non-target
        "cult_regional_dialect_cue",
        "cult_appearance_norm_cue",
        "cult_competence_norm_cue",
        "cult_status_hierarchy_cue",
        "cult_family_kinship_cue",
        "cult_education_norm_cue",
        "cult_gender_identity_cue",
    ],
    "emotional": [   # Dims 14..20 (7D) - Komponen augmentasi emosi non-target
        "emo_hostility_aggression",
        "emo_derision_ridicule",
        "emo_disgust_contempt",
        "emo_humiliation_shaming",
        "emo_sadness_distress",
        "emo_calm_neutral_register",
        "emo_sarcastic_emoji_affect",
    ],
    "pragmatic": [   # Dims 21..26 (6D)
        "prag_irony_sarcasm_signal",
        "prag_presupposition_insinuation",
        "prag_indirect_advice_framing",
        "prag_hyperbole_exaggeration",
        "prag_imperative_attack",
        "prag_benign_supportive",
    ],
    "contextual": [  # Dims 27..32 (6D)
        "ctx_direct_addressee_target",
        "ctx_public_platform_stage",
        "ctx_evaluative_judgment",
        "ctx_harm_impact_evidence",
        "ctx_provocation_reply",
        "ctx_teacher_confidence_raw",
    ],
}

CULTURAL_CUE_SLICE = slice(7, 14)
EMOTIONAL_CUE_SLICE = slice(14, 21)

CUE_FAMILY_SLICES: Dict[str, slice] = {
    "Linguistic": slice(0, 7),
    "Cultural": slice(7, 14),
    "Emotional": slice(14, 21),
    "Pragmatic": slice(21, 27),
    "Contextual": slice(27, 33),
}

TOTAL_VECTOR_DIM = 33  # 7 + 7 + 7 + 6 + 6 = 33 Dimensi Bukti Kontekstual Non-Target


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
# 6. EKSTRAKSI REPRESENTASI BUKTI KONTEKSTUAL LLM (33-DIM NON-TARGET)
# =============================================================================

CONTEXTUAL_EVIDENCE_SCHEMA = [
    {
        "dim_start": 0,
        "dim_end": 7,
        "family": "Linguistic",
        "features": CONTEXTUAL_EVIDENCE_GROUPS["linguistic"],
        "origin": "Surface lexical/morphological/rhetorical text markers + LLM contextual_cues.linguistic + CoT rationale lexical evidence",
    },
    {
        "dim_start": 7,
        "dim_end": 14,
        "family": "Cultural",
        "features": CONTEXTUAL_EVIDENCE_GROUPS["cultural"],
        "origin": "Indonesian regional dialect lexicon + LLM contextual_cues.cultural + CoT cultural norm evidence (zero copy of target cultural_context)",
    },
    {
        "dim_start": 14,
        "dim_end": 21,
        "family": "Emotional",
        "features": CONTEXTUAL_EVIDENCE_GROUPS["emotional"],
        "origin": "Emoji affective polarity + LLM contextual_cues.emotional + CoT affective tone evidence (zero copy of target emotion)",
    },
    {
        "dim_start": 21,
        "dim_end": 27,
        "family": "Pragmatic",
        "features": CONTEXTUAL_EVIDENCE_GROUPS["pragmatic"],
        "origin": "Illocutionary/pragmatic framing cues + LLM contextual_cues.pragmatic + CoT pragmatic speech-act evidence",
    },
    {
        "dim_start": 27,
        "dim_end": 33,
        "family": "Contextual",
        "features": CONTEXTUAL_EVIDENCE_GROUPS["contextual"],
        "origin": "Target-addressee/platform discourse markers + LLM CoT harm evidence + uncalibrated scalar confidence [0,1]",
    },
]

# Backward-compatible schema alias
TEACHER_FEATURE_SCHEMA = CONTEXTUAL_EVIDENCE_SCHEMA

# Leksikon pola bukti non-target (bahasa Indonesia + Inggris untuk CoT rationale & komentar)
_CUE_LEXICONS: Dict[str, List[str]] = {
    "ling_derogatory": [
        "bego", "tolol", "idiot", "goblok", "bodoh", "anjing", "babi", "bangsat", "bajingan",
        "hama", "sampah", "jelek", "dekil", "burik", "insult", "derogatory", "profanity",
        "offensive", "slur", "name-calling", "calling someone", "penghinaan", "makian", "cercaan",
    ],
    "ling_metaphor": [
        "mirip", "seperti", "kayak", "bagai", "laksana", "compares", "comparison", "metaphor",
        "simile", "analog", "membandingkan", "umpama", "ursula", "segentong", "monster",
    ],
    "cult_appearance": [
        "gemuk", "gendut", "kurus", "pendek", "jerawat", "behel", "gigi", "wajah", "muka",
        "badan", "tubuh", "weight", "overweight", "ugly", "appearance", "looks", "physical",
        "body shaming", "eating habits", "braces", "teeth", "outfit", "makeup", "fisik", "penampilan",
    ],
    "cult_competence": [
        "otak", "iq", "pintar", "cerdas", "bodoh", "tolol", "bego", "dungu", "intelligence",
        "stupid", "worth", "damage", "incompetent", "ability", "kecerdasan", "kemampuan", "nilai",
    ],
    "cult_status": [
        "miskin", "kaya", "gembel", "kampungan", "udik", "norak", "status", "economic",
        "wealth", "class", "social standing", "ekonomi", "strata", "pansos", "sok kaya",
    ],
    "cult_family": [
        "yatim", "piatu", "anak haram", "ibu", "bapak", "ortu", "keluarga", "family",
        "parents", "upbringing", "keturunan", "orang tua",
    ],
    "cult_education": [
        "sekolah", "kuliah", "kampus", "sd", "smp", "sma", "sarjana", "skripsi", "guru",
        "dosen", "education", "school", "academic", "pendidikan", "lulusan",
    ],
    "cult_gender": [
        "bencong", "banci", "lonte", "jalang", "pelacur", "perawan", "janda", "cewek",
        "gender", "sexist", "feminine", "masculine", "harassment", "seksual",
    ],
    "emo_hostility": [
        "bunuh", "mati", "hajar", "gampar", "angry", "anger", "hostil", "aggres", "threat",
        "violent", "rage", "marah", "geram", "ancam", "benci", "hate",
    ],
    "emo_derision": [
        "haha", "wkwk", "wk", "kwkw", "mock", "mocking", "ridicul", "teas", "laugh",
        "ejek", "olok", "menertawakan", "cemooh", "ngeledek", "ledek",
    ],
    "emo_disgust": [
        "jijik", "muak", "muntah", "najis", "cringe", "disgust", "repuls", "gross", "sick",
    ],
    "emo_humiliation": [
        "malu", "dipermalukan", "aib", "humiliat", "embarrass", "shaming", "degrad", "worthless",
        "rendah", "hina",
    ],
    "emo_sadness": [
        "sedih", "menangis", "kasihan", "miris", "sad", "distress", "pity", "sorrow", "hurt",
    ],
    "emo_neutral_calm": [
        "neutral", "netral", "question", "asking", "suggestion", "reminder", "benign",
        "without any", "no insulting", "does not contain", "tanpa indikasi", "pertanyaan biasa",
        "informatif", "positif", "semangat", "mendukung", "helpful",
    ],
    "prag_irony": [
        "sarcas", "ironi", "irony", "satir", "sindir", "sinis", "rhetoric", "retoris",
        "exaggerat", "cynic", "sindiran", "bercanda kasar",
    ],
    "prag_presupposition": [
        "impl", "suggest", "presuppos", "menyirat", "tersirat", "tersembunyi", "indirect",
        "subtle", "semi-implicit", "berarti", "menuduh", "insinuat",
    ],
    "prag_indirect_advice": [
        "minimal", "beli ", "pake ", "pakai ", "mending", "coba ", "sebaiknya", "advice",
        "reminder", "recommend", "suggestion", "kepikir", "kepikiran",
    ],
    "prag_hyperbole": [
        "banget", "segentong", "sedunia", "seumur", "sampai mati", "gulung tikar",
        "exaggerat", "overstat", "hyperbol", "excessively",
    ],
    "prag_imperative": [
        "mati aja", "diam", "minggat", "pergi", "jangan", "mampus", "sana", "command",
        "imperative", "telling them to",
    ],
    "prag_benign": [
        "no insulting", "does not contain", "no harmful", "neutral question", "supportive",
        "encourag", "tanpa unsur", "bukan perundungan", "positif", "semangat", "saran baik",
    ],
    "ctx_public_stage": [
        "tiktok", "ig", "instagram", "twitter", "x ", "youtube", "netizen", "followers",
        "viral", "fyp", "konten", "public", "sosmed", "media sosial",
    ],
    "ctx_evaluative": [
        "stylenya", "kelakuan", "sifat", "mental", "judge", "evaluat", "critic", "commenting on",
        "penilaian", "menilai", "menghakimi",
    ],
    "ctx_harm_evidence": [
        "harm", "cyberbullying", "bullying", "insulting", "humiliating", "attack",
        "perundungan", "merendahkan", "menyakiti", "buruk", "negative impact",
    ],
}

_SARCASTIC_EMOJIS = ["🤡", "🗿", "💀", "☠️", "🤮", "🤢", "💩", "🐷", "🐶", "🐒", "🖕", "🤬", "😏", "🙄", "😂", "🤣"]


def _keyword_score(text_blob: str, keywords: List[str], cap: int = 3) -> float:
    """Computes bounded [0.0, 1.0] lexical evidence score from keyword hits."""
    if not text_blob:
        return 0.0
    hits = sum(1 for kw in keywords if kw in text_blob)
    if hits <= 0:
        return 0.0
    return round(min(1.0, hits / float(cap)), 4)


def _parse_cues_dict(raw_cues: Any) -> Dict[str, Dict[str, float]]:
    if isinstance(raw_cues, dict):
        return raw_cues
    if isinstance(raw_cues, str) and raw_cues.strip().startswith("{"):
        try:
            import json as _json
            obj = _json.loads(raw_cues)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _blend_cue(llm_val: Optional[float], heuristic_val: float) -> float:
    """Prefers explicit LLM contextual_cues when provided, gently blended with observed surface signal."""
    if llm_val is not None:
        val = max(0.0, min(1.0, float(llm_val)))
        return round(0.75 * val + 0.25 * heuristic_val, 4) if heuristic_val > 0 else round(val, 4)
    return round(max(0.0, min(1.0, heuristic_val)), 4)


def extract_contextual_evidence_vector(
    annotation_dict: Dict[str, Any],
    text: Optional[str] = None,
) -> np.ndarray:
    """
    Membangun vektor Representasi Bukti Kontekstual LLM 33-dimensi (Stream B) secara NON-LEAKY.

    KONTRAK METODOLOGIS KRITIS:
    - Fungsi ini secara eksplisit DILARANG membaca kolom label target klasifikasi
      (`cyberbullying`, `type`, `emotion`, `cultural_context`, `implicit_level`, `severity`, `target_*`, `label`).
    - Sumber bukti berasal murni dari:
      1. Sinyal teks komentar (`comment` / `comment_normalized`) dan leksikon sosio-pragmatik/dialek/emoji.
      2. Teks penalaran CoT LLM (`explanation` / `llm_explanation`).
      3. Kamus terstruktur `contextual_cues` dari SumoPod (`iicd-v3.0-sumopod`) bila tersedia.
      4. Skalar kepercayaan mentah LLM (`confidence` terkalibrasi-terbatas dalam [0.0, 1.0] sebagai dimensi prior ke-33).
    """
    raw_comment = str(
        text if text is not None else annotation_dict.get("comment", annotation_dict.get("comment_normalized", ""))
    )
    comment_lower = raw_comment.lower()
    norm_comment = clean_and_normalize_social_text(raw_comment).lower()
    cot_exp = str(annotation_dict.get("llm_explanation", annotation_dict.get("explanation", ""))).lower().strip()
    evidence_blob = f"{norm_comment} {cot_exp}".strip()

    cues = _parse_cues_dict(annotation_dict.get("contextual_cues"))
    ling_c = cues.get("linguistic", {}) if isinstance(cues.get("linguistic"), dict) else {}
    cult_c = cues.get("cultural", {}) if isinstance(cues.get("cultural"), dict) else {}
    emo_c = cues.get("emotional", {}) if isinstance(cues.get("emotional"), dict) else {}
    prag_c = cues.get("pragmatic", {}) if isinstance(cues.get("pragmatic"), dict) else {}
    ctx_c = cues.get("contextual", {}) if isinstance(cues.get("contextual"), dict) else {}

    def _get_cue(sub: Dict[str, Any], key: str) -> Optional[float]:
        val = sub.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return None

    # --- 1. LINGUISTIC CUES (Dims 0..6, 7D) ---
    tokens = norm_comment.split()
    n_tok = max(1, len(tokens))
    ling_derog = _blend_cue(_get_cue(ling_c, "derogatory_lexical"), _keyword_score(evidence_blob, _CUE_LEXICONS["ling_derogatory"], cap=2))
    ling_meta = _blend_cue(_get_cue(ling_c, "metaphor_simile"), _keyword_score(evidence_blob, _CUE_LEXICONS["ling_metaphor"], cap=2))
    q_ex_count = raw_comment.count("?") + raw_comment.count("!")
    upper_ratio = sum(1 for c in raw_comment if c.isupper()) / max(1, len(raw_comment))
    rhet_heur = min(1.0, (q_ex_count / 3.0) + (0.4 if upper_ratio > 0.35 and len(raw_comment) > 5 else 0.0))
    ling_rhet = _blend_cue(_get_cue(ling_c, "rhetorical_punct"), rhet_heur)
    elong_hits = len(re.findall(r'([a-zA-Z])\1{2,}', comment_lower))
    ling_elong = round(min(1.0, elong_hits / 2.0), 4)
    slang_hits = sum(1 for tok in re.findall(r'\b\w+\b', comment_lower) if tok in SLANG_DICTIONARY)
    ling_slang = round(min(1.0, slang_hits / max(2.0, n_tok * 0.5)), 4)
    second_person_toks = {"lu", "lo", "elu", "elo", "kamu", "kau", "mu", "kalian", "luw", "dia", "situ"}
    ling_2nd = 1.0 if any(tok in second_person_toks for tok in re.findall(r'\b\w+\b', comment_lower)) else 0.0
    neg_hits = sum(1 for kw in ["tidak", "bukan", "gak", "ga", "gk", "nggak", "gaada", "kagak"] if re.search(rf"\b{kw}\b", comment_lower))
    ling_neg = round(min(1.0, neg_hits / 2.0), 4)

    linguistic_vec = [ling_derog, ling_meta, ling_rhet, ling_elong, ling_slang, ling_2nd, ling_neg]

    # --- 2. CULTURAL CUES (Dims 7..13, 7D) [NON-TARGET] ---
    detected_dialects = detect_dialects(raw_comment)
    cult_dialect = _blend_cue(_get_cue(cult_c, "regional_dialect"), min(1.0, len(detected_dialects) / 1.5))
    cult_app = _blend_cue(_get_cue(cult_c, "appearance_norm"), _keyword_score(evidence_blob, _CUE_LEXICONS["cult_appearance"], cap=2))
    cult_comp = _blend_cue(_get_cue(cult_c, "competence_norm"), _keyword_score(evidence_blob, _CUE_LEXICONS["cult_competence"], cap=2))
    cult_stat = _blend_cue(_get_cue(cult_c, "status_hierarchy"), _keyword_score(evidence_blob, _CUE_LEXICONS["cult_status"], cap=2))
    cult_fam = _blend_cue(_get_cue(cult_c, "family_kinship"), _keyword_score(evidence_blob, _CUE_LEXICONS["cult_family"], cap=2))
    cult_edu = round(_keyword_score(evidence_blob, _CUE_LEXICONS["cult_education"], cap=2), 4)
    cult_gen = _blend_cue(_get_cue(cult_c, "gender_identity"), _keyword_score(evidence_blob, _CUE_LEXICONS["cult_gender"], cap=2))

    cultural_vec = [cult_dialect, cult_app, cult_comp, cult_stat, cult_fam, cult_edu, cult_gen]

    # --- 3. EMOTIONAL CUES (Dims 14..20, 7D) [NON-TARGET] ---
    emo_host = _blend_cue(_get_cue(emo_c, "hostility_aggression"), _keyword_score(evidence_blob, _CUE_LEXICONS["emo_hostility"], cap=2))
    emo_deris = _blend_cue(_get_cue(emo_c, "derision_ridicule"), _keyword_score(evidence_blob, _CUE_LEXICONS["emo_derision"], cap=2))
    emo_disg = _blend_cue(_get_cue(emo_c, "disgust_contempt"), _keyword_score(evidence_blob, _CUE_LEXICONS["emo_disgust"], cap=2))
    emo_humil = _blend_cue(_get_cue(emo_c, "humiliation_shaming"), _keyword_score(evidence_blob, _CUE_LEXICONS["emo_humiliation"], cap=2))
    emo_sad = _blend_cue(_get_cue(emo_c, "sadness_distress"), _keyword_score(evidence_blob, _CUE_LEXICONS["emo_sadness"], cap=2))
    emo_neu = round(_keyword_score(cot_exp if cot_exp else evidence_blob, _CUE_LEXICONS["emo_neutral_calm"], cap=2), 4)
    em_hits = sum(1 for em in _SARCASTIC_EMOJIS if em in raw_comment)
    emo_emoji = round(min(1.0, em_hits / 2.0), 4)

    emotional_vec = [emo_host, emo_deris, emo_disg, emo_humil, emo_sad, emo_neu, emo_emoji]

    # --- 4. PRAGMATIC CUES (Dims 21..26, 6D) ---
    prag_irony = _blend_cue(_get_cue(prag_c, "irony_sarcasm"), _keyword_score(evidence_blob, _CUE_LEXICONS["prag_irony"], cap=2))
    prag_presup = _blend_cue(_get_cue(prag_c, "presupposition_insinuation"), _keyword_score(evidence_blob, _CUE_LEXICONS["prag_presupposition"], cap=2))
    prag_advice = _blend_cue(_get_cue(prag_c, "indirect_advice"), _keyword_score(evidence_blob, _CUE_LEXICONS["prag_indirect_advice"], cap=2))
    prag_hyper = round(_keyword_score(evidence_blob, _CUE_LEXICONS["prag_hyperbole"], cap=2), 4)
    prag_imper = round(_keyword_score(evidence_blob, _CUE_LEXICONS["prag_imperative"], cap=2), 4)
    prag_benign = _blend_cue(_get_cue(prag_c, "benign_supportive"), _keyword_score(cot_exp if cot_exp else evidence_blob, _CUE_LEXICONS["prag_benign"], cap=2))

    pragmatic_vec = [prag_irony, prag_presup, prag_advice, prag_hyper, prag_imper, prag_benign]

    # --- 5. CONTEXTUAL & EPISTEMIC CUES (Dims 27..32, 6D) ---
    addressee_heur = 1.0 if ("@" in raw_comment or ling_2nd > 0 or "target" in cot_exp or "person" in cot_exp) else 0.0
    ctx_addr = _blend_cue(_get_cue(ctx_c, "direct_addressee"), addressee_heur)
    ctx_stage = _blend_cue(_get_cue(ctx_c, "public_stage"), _keyword_score(evidence_blob, _CUE_LEXICONS["ctx_public_stage"], cap=2))
    ctx_eval = _blend_cue(_get_cue(ctx_c, "evaluative_judgment"), _keyword_score(evidence_blob, _CUE_LEXICONS["ctx_evaluative"], cap=2))
    ctx_harm = _blend_cue(_get_cue(ctx_c, "harm_impact_evidence"), _keyword_score(cot_exp if cot_exp else evidence_blob, _CUE_LEXICONS["ctx_harm_evidence"], cap=2))
    ctx_reply = 1.0 if (raw_comment.strip().startswith("@") or "?" in raw_comment or any(k in comment_lower for k in ["balas", "reply", "makanya", "dari mulut"])) else 0.0

    raw_conf = annotation_dict.get("llm_confidence", annotation_dict.get("confidence", None))
    conf_val = 0.0
    if raw_conf is not None and not (isinstance(raw_conf, float) and np.isnan(raw_conf)):
        try:
            fval = float(raw_conf)
            if 1.0 < fval <= 100.0:
                fval = fval / 100.0
            if 0.0 <= fval <= 1.0:
                conf_val = round(fval, 4)
        except (TypeError, ValueError):
            conf_val = 0.0

    contextual_vec = [ctx_addr, ctx_stage, ctx_eval, ctx_harm, ctx_reply, conf_val]

    vector = np.array(
        linguistic_vec + cultural_vec + emotional_vec + pragmatic_vec + contextual_vec,
        dtype=np.float32,
    )
    assert vector.shape == (TOTAL_VECTOR_DIM,), f"Expected {TOTAL_VECTOR_DIM} features, got {vector.shape}"
    return vector


def extract_teacher_annotation_features(
    annotation_dict: Dict[str, Any],
    class_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
    text: Optional[str] = None,
) -> np.ndarray:
    """
    Wrapper terstandarisasi untuk ekstraksi fitur Stream B 33-Dimensi.
    Menghasilkan Representasi Bukti Kontekstual Non-Target (Linguistic, Cultural,
    Emotional, Pragmatic, Contextual) tanpa menyalin kolom label target klasifikasi.
    """
    _ = class_probabilities  # Target probabilities are intentionally NOT copied into Stream B.
    return extract_contextual_evidence_vector(annotation_dict, text=text)


def extract_taxonomy_probability_vector(
    annotation_dict: Dict[str, Any],
    class_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
    text: Optional[str] = None,
) -> np.ndarray:
    """Alias kompatibilitas yang memanggil ekstraktor bukti kontekstual non-target 33-dimensi."""
    return extract_contextual_evidence_vector(annotation_dict, text=text)


def extract_cue_family_summary(vec: np.ndarray) -> Dict[str, float]:
    """Mengagregasi vektor 33D menjadi 5 skor ringkas famili bukti (mean activation per grup)."""
    arr = np.asarray(vec, dtype=float).ravel()
    if len(arr) < TOTAL_VECTOR_DIM:
        padded = np.zeros(TOTAL_VECTOR_DIM, dtype=float)
        padded[: len(arr)] = arr
        arr = padded
    return {
        family: round(float(np.mean(arr[slc])), 4)
        for family, slc in CUE_FAMILY_SLICES.items()
    }


def get_teacher_feature_names() -> List[str]:
    """Mengembalikan daftar 33 nama fitur bukti kontekstual non-target."""
    names: List[str] = []
    for grp in ("linguistic", "cultural", "emotional", "pragmatic", "contextual"):
        names.extend(CONTEXTUAL_EVIDENCE_GROUPS[grp])
    return names


def get_taxonomy_vector_feature_names() -> List[str]:
    """Backward-compatible alias returning the 33-dim contextual evidence feature names."""
    return get_teacher_feature_names()


def get_teacher_feature_schema() -> List[Dict[str, Any]]:
    """Returns full documentation metadata for every dimension group of the 33-dim vector."""
    return CONTEXTUAL_EVIDENCE_SCHEMA


def format_structured_rationale(annotation_dict: Dict[str, Any]) -> str:
    """
    Memformat ringkasan bukti penalaran LLM (explanation) untuk tampilan analitik / Dual-XAI.
    Bebas target leakage (tidak menyertakan kode label target ke teks encoder).
    """
    explanation = str(annotation_dict.get("llm_explanation", annotation_dict.get("explanation", ""))).strip()
    return explanation
