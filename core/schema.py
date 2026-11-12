import re
from typing import Optional, Any, Dict, List, Tuple
from pydantic import BaseModel, Field

MISSING_LABEL = "UNKNOWN"

VALID_CYBERBULLYING = {"YES", "NO"}
VALID_TYPES = {"INS", "BSH", "SHM", "SAR", "THR", "DIS", "SEX", "EXC", "NONE"}
VALID_EMOTIONS = {"ANG", "DIS", "HUM", "MOCK", "SAD", "NEU"}
VALID_CULTURAL_CONTEXTS = {"APP", "INT", "STA", "FAM", "EDU", "REG", "GEN", "NONE"}
VALID_IMPLICIT_LEVELS = {"E0", "E1", "E2"}
VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "NONE"}


class AnnotationSchema(BaseModel):
    cyberbullying: str = Field(description="YES or NO (or UNKNOWN if missing)")
    type: str = Field(description="INS, BSH, SHM, SAR, THR, DIS, SEX, EXC, NONE, or UNKNOWN")
    emotion: str = Field(description="ANG, DIS, HUM, MOCK, SAD, NEU, or UNKNOWN")
    cultural_context: str = Field(description="APP, INT, STA, FAM, EDU, REG, GEN, NONE, or UNKNOWN")
    implicit_level: str = Field(description="E0, E1, E2, or UNKNOWN")
    severity: str = Field(description="LOW, MEDIUM, HIGH, NONE, or UNKNOWN")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Numeric confidence score in range 0.0 to 1.0"
    )
    explanation: Optional[str] = Field(default="", description="Chain-of-thought explanation of the classification")


def parse_confidence_score(val: Any, default: Optional[float] = 0.85) -> Optional[float]:
    """
    Converts diverse confidence formats (float, int, string numbers, 'HIGH'/'MEDIUM'/'LOW')
    into a standardized float strictly in [0.0, 1.0]. Returns `default` if invalid/missing.
    """
    if val is None or (isinstance(val, float) and val != val):
        return default

    if isinstance(val, (int, float)):
        score = float(val)
        if 1.0 < score <= 100.0:
            score = score / 100.0
        if score < 0.0 or score > 100.0:
            return default
        return round(max(0.0, min(1.0, score)), 4)

    s = str(val).strip().upper()
    if not s or s in {"-", "N/A", "NA", "NONE", "NULL", "NAN"}:
        return default

    qualitative_map = {
        "VERY HIGH": 0.95,
        "SANGAT TINGGI": 0.95,
        "HIGH": 0.90,
        "TINGGI": 0.90,
        "MEDIUM": 0.70,
        "MODERATE": 0.70,
        "SEDANG": 0.70,
        "VERY LOW": 0.30,
        "SANGAT RENDAH": 0.30,
        "LOW": 0.50,
        "RENDAH": 0.50,
    }
    for k, score in qualitative_map.items():
        if k in s:
            return score

    s_clean = s.replace(",", ".")
    match = re.search(r"(-?\d+(?:\.\d+)?)", s_clean)
    if match:
        try:
            num = float(match.group(1))
            if "%" in s or (1.0 < num <= 100.0):
                num = num / 100.0
            if num < 0.0 or num > 1.0:
                return default
            return round(num, 4)
        except ValueError:
            pass

    return default


def _match_valid_code(raw_val: Any, valid_codes: set) -> Optional[str]:
    """Extracts an exact or contained taxonomy code; returns None if missing/invalid."""
    if raw_val is None or (isinstance(raw_val, float) and raw_val != raw_val):
        return None
    s = str(raw_val).strip().upper()
    if not s or s in {"-", "NA", "N/A", "NULL", "NAN", "UNKNOWN", "MISSING"}:
        return None
    if s in valid_codes:
        return s
    # Word-boundary lookup first, then substring fallback
    tokens = re.findall(r"[A-Z0-9_]+", s)
    for tok in tokens:
        if tok in valid_codes:
            return tok
    for vc in sorted(valid_codes, key=len, reverse=True):
        if vc in s:
            return vc
    return None


def validate_annotation_record(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Checks raw taxonomy & cross-field consistency of an annotation dictionary."""
    issues: List[str] = []
    cb_raw = str(data.get("cyberbullying", "")).strip().upper()
    cb = None
    if cb_raw in {"YES", "1", "TRUE"} or "YES" in cb_raw:
        cb = "YES"
    elif cb_raw in {"NO", "0", "FALSE"} or "NO" in cb_raw:
        cb = "NO"
    else:
        issues.append(f"invalid_cyberbullying:{cb_raw or 'EMPTY'}")

    t_val = _match_valid_code(data.get("type"), VALID_TYPES)
    if t_val is None:
        issues.append("missing_or_invalid_type")
    e_val = _match_valid_code(data.get("emotion"), VALID_EMOTIONS)
    if e_val is None:
        issues.append("missing_or_invalid_emotion")
    c_val = _match_valid_code(data.get("cultural_context"), VALID_CULTURAL_CONTEXTS)
    if c_val is None:
        issues.append("missing_or_invalid_cultural_context")
    i_val = _match_valid_code(data.get("implicit_level"), VALID_IMPLICIT_LEVELS)
    if i_val is None:
        issues.append("missing_or_invalid_implicit_level")

    s_raw = str(data.get("severity", "")).strip().upper()
    s_val = _match_valid_code("MEDIUM" if "MED" in s_raw and "MEDIUM" not in s_raw else s_raw, VALID_SEVERITIES)
    if s_val is None:
        issues.append("missing_or_invalid_severity")

    conf = parse_confidence_score(data.get("confidence"), default=None)
    if conf is None or not (0.0 <= conf <= 1.0):
        issues.append("invalid_confidence")

    if cb == "NO":
        if t_val not in (None, "NONE"):
            issues.append(f"inconsistent_type_for_non_cb:{t_val}")
        if s_val not in (None, "NONE", "LOW"):
            issues.append(f"inconsistent_severity_for_non_cb:{s_val}")
    elif cb == "YES":
        if t_val == "NONE":
            issues.append("inconsistent_type_NONE_for_cb_YES")
        if s_val == "NONE":
            issues.append("inconsistent_severity_NONE_for_cb_YES")

    return len(issues) == 0, issues


def sanitize_annotation_dict(
    data: dict,
    preserve_missing: bool = True,
) -> dict:
    """
    Normalizes extracted fields into standard taxonomy codes.
    When `preserve_missing=True` (recommended for scientific curation), unknown/missing
    fields are explicitly preserved as 'UNKNOWN' (and confidence as None when absent/invalid)
    rather than fabricating default valid labels.
    """
    # 1. Cyberbullying (YES / NO / UNKNOWN)
    cb_raw = "" if data.get("cyberbullying") is None else str(data.get("cyberbullying", "")).strip().upper()
    if cb_raw in {"YES", "1", "TRUE", "YA", "POSITIVE_CB"} or "YES" in cb_raw:
        cb = "YES"
    elif cb_raw in {"NO", "0", "FALSE", "BUKAN", "TIDAK", "NEGATIVE"} or re.search(r"\bNO\b", cb_raw):
        cb = "NO"
    else:
        cb = MISSING_LABEL if preserve_missing else "NO"

    # 2. Type
    t_match = _match_valid_code(data.get("type"), VALID_TYPES)
    if t_match is not None:
        t_val = t_match
    elif cb == "NO" and str(data.get("type", "")).strip().upper() in {"", "-", "NA", "N/A", "NULL", "NONE"}:
        t_val = "NONE"
    else:
        t_val = MISSING_LABEL if preserve_missing else ("NONE" if cb == "NO" else MISSING_LABEL)

    # 3. Emotion
    e_match = _match_valid_code(data.get("emotion"), VALID_EMOTIONS)
    e_val = e_match if e_match is not None else (MISSING_LABEL if preserve_missing else "NEU")

    # 4. Cultural Context
    c_match = _match_valid_code(data.get("cultural_context"), VALID_CULTURAL_CONTEXTS)
    if c_match is not None:
        c_val = c_match
    elif cb == "NO" and str(data.get("cultural_context", "")).strip().upper() in {"", "-", "NA", "N/A", "NULL", "NONE"}:
        c_val = "NONE"
    else:
        c_val = MISSING_LABEL if preserve_missing else "NONE"

    # 5. Implicit Level
    i_match = _match_valid_code(data.get("implicit_level"), VALID_IMPLICIT_LEVELS)
    i_val = i_match if i_match is not None else (MISSING_LABEL if preserve_missing else "E0")

    # 6. Severity
    s_raw = "" if data.get("severity") is None else str(data.get("severity", "")).strip().upper()
    if "MED" in s_raw and "MEDIUM" not in s_raw:
        s_raw = "MEDIUM"
    s_match = _match_valid_code(s_raw, VALID_SEVERITIES)
    if s_match is not None:
        s_val = s_match
    elif cb == "NO" and s_raw in {"", "-", "NA", "N/A", "NULL", "NONE"}:
        s_val = "NONE"
    else:
        s_val = MISSING_LABEL if preserve_missing else ("NONE" if cb == "NO" else MISSING_LABEL)

    # Strict cross-field consistency enforcement for clean NO rows
    if cb == "NO":
        if t_val != "NONE" and t_val != MISSING_LABEL:
            t_val = "NONE"
        if s_val not in {"NONE", MISSING_LABEL}:
            s_val = "NONE"

    # 7. Confidence Score
    conf_val = parse_confidence_score(
        data.get("confidence"),
        default=None if preserve_missing else 0.0,
    )

    # 8. Explanation
    raw_exp = data.get("explanation", "")
    exp_val = "" if raw_exp is None else str(raw_exp).strip()
    if not exp_val or exp_val.upper() in {"-", "N/A", "NA", "NULL", "NAN", "NONE"}:
        if preserve_missing:
            exp_val = ""
        elif cb == "NO":
            exp_val = "Teks merupakan ujaran netral tanpa indikasi perundungan siber."
        else:
            exp_val = f"Teks teridentifikasi memuat unsur perundungan siber ({t_val})."

    return {
        "cyberbullying": cb,
        "type": t_val,
        "emotion": e_val,
        "cultural_context": c_val,
        "implicit_level": i_val,
        "severity": s_val,
        "confidence": conf_val,
        "explanation": exp_val,
    }
