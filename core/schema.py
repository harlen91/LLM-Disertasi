import re
from typing import Optional, Any
from pydantic import BaseModel, Field

class AnnotationSchema(BaseModel):
    cyberbullying: str = Field(description="YES or NO")
    type: str = Field(description="INS, BSH, SHM, SAR, THR, DIS, SEX, EXC, or NONE")
    emotion: str = Field(description="ANG, DIS, HUM, MOCK, SAD, NEU")
    cultural_context: str = Field(description="APP, INT, STA, FAM, EDU, REG, GEN, or NONE")
    implicit_level: str = Field(description="E0, E1, E2")
    severity: str = Field(description="LOW, MEDIUM, HIGH, or NONE")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Numeric confidence score in range 0.0 to 1.0")
    explanation: Optional[str] = Field(default="", description="Chain-of-thought explanation of the classification")

def parse_confidence_score(val: Any, default: float = 0.85) -> float:
    """
    Converts diverse confidence formats (float, int, string numbers, 'HIGH'/'MEDIUM'/'LOW')
    into a standardized float strictly in the range [0.0, 1.0].
    """
    if val is None:
        return default

    # If already a number
    if isinstance(val, (int, float)):
        score = float(val)
        if score > 1.0 and score <= 100.0:
            score = score / 100.0
        return round(max(0.0, min(1.0, score)), 2)

    s = str(val).strip().upper()
    if not s or s in ["-", "N/A", "NONE", "NULL"]:
        return default

    # Map qualitative linguistic labels to numeric scores
    qualitative_map = {
        "HIGH": 0.90,
        "VERY HIGH": 0.95,
        "TINGGI": 0.90,
        "SANGAT TINGGI": 0.95,
        "MEDIUM": 0.70,
        "MODERATE": 0.70,
        "SEDANG": 0.70,
        "LOW": 0.50,
        "RENDAH": 0.50,
        "VERY LOW": 0.30,
        "SANGAT RENDAH": 0.30
    }
    for k, score in qualitative_map.items():
        if k in s:
            return score

    # Extract numeric pattern like 0.85, .9, 85%, etc.
    s_clean = s.replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)", s_clean)
    if match:
        try:
            num = float(match.group(1))
            if "%" in s or num > 1.0:
                num = num / 100.0
            return round(max(0.0, min(1.0, num)), 2)
        except ValueError:
            pass

    return default

def sanitize_annotation_dict(data: dict) -> dict:
    """
    Normalize extracted fields to ensure clean presentation in UI and datasets.
    Guarantees:
    1. Confidence is strictly float between 0.0 and 1.0.
    2. No fields are empty or blank (empty strings replaced with 'NONE', 'NEU', 'E0', etc.).
    """
    # 1. Cyberbullying (YES / NO)
    cb_raw = str(data.get("cyberbullying", "")).strip().upper()
    if "YES" in cb_raw:
        cb = "YES"
    elif "NO" in cb_raw:
        cb = "NO"
    else:
        cb = "NO" if cb_raw in ["0", "FALSE", "BUKAN", "TIDAK", "NEGATIVE"] else "YES"

    # 2. Type
    valid_types = {"INS", "BSH", "SHM", "SAR", "THR", "DIS", "SEX", "EXC", "NONE"}
    t_raw = str(data.get("type", "")).strip().upper()
    if t_raw in ["", "-", "NA", "N/A", "NULL", "NONE"]:
        t_val = "NONE" if cb == "NO" else "INS"
    else:
        # Check if code exists in string
        matched_type = next((vt for vt in valid_types if vt in t_raw), None)
        t_val = matched_type if matched_type else ("NONE" if cb == "NO" else "INS")

    # 3. Emotion
    valid_emotions = {"ANG", "DIS", "HUM", "MOCK", "SAD", "NEU"}
    e_raw = str(data.get("emotion", "")).strip().upper()
    if e_raw in ["", "-", "NA", "N/A", "NULL", "NONE"]:
        e_val = "NEU"
    else:
        matched_emo = next((ve for ve in valid_emotions if ve in e_raw), None)
        e_val = matched_emo if matched_emo else "NEU"

    # 4. Cultural Context
    valid_cults = {"APP", "INT", "STA", "FAM", "EDU", "REG", "GEN", "NONE"}
    c_raw = str(data.get("cultural_context", "")).strip().upper()
    if c_raw in ["", "-", "NA", "N/A", "NULL", "NONE"]:
        c_val = "NONE"
    else:
        matched_cult = next((vc for vc in valid_cults if vc in c_raw), None)
        c_val = matched_cult if matched_cult else "NONE"

    # 5. Implicit Level
    valid_implicits = {"E0", "E1", "E2"}
    i_raw = str(data.get("implicit_level", "")).strip().upper()
    if i_raw in ["", "-", "NA", "N/A", "NULL", "NONE"]:
        i_val = "E0"
    else:
        matched_imp = next((vi for vi in valid_implicits if vi in i_raw), None)
        i_val = matched_imp if matched_imp else "E0"

    # 6. Severity
    valid_severities = {"LOW", "MEDIUM", "HIGH", "NONE"}
    s_raw = str(data.get("severity", "")).strip().upper()
    if s_raw in ["", "-", "NA", "N/A", "NULL", "NONE"]:
        s_val = "NONE" if cb == "NO" else "LOW"
    else:
        if "HIGH" in s_raw:
            s_val = "HIGH"
        elif "MED" in s_raw:
            s_val = "MEDIUM"
        elif "LOW" in s_raw:
            s_val = "LOW"
        elif "NONE" in s_raw:
            s_val = "NONE"
        else:
            s_val = "NONE" if cb == "NO" else "LOW"

    # 7. Confidence Score (Float 0.0 - 1.0)
    conf_raw = data.get("confidence", 0.85)
    conf_val = parse_confidence_score(conf_raw, default=0.85)

    # 8. Explanation (Non-empty rationale)
    exp_val = str(data.get("explanation", "")).strip()
    if not exp_val or exp_val in ["-", "N/A", "NULL"]:
        if cb == "NO":
            exp_val = "Teks merupakan ujaran netral tanpa indikasi perundungan siber atau ujaran kebencian."
        else:
            exp_val = f"Teks teridentifikasi memuat unsur perundungan siber tipe {t_val}."

    return {
        "cyberbullying": cb,
        "type": t_val,
        "emotion": e_val,
        "cultural_context": c_val,
        "implicit_level": i_val,
        "severity": s_val,
        "confidence": conf_val,
        "explanation": exp_val
    }
