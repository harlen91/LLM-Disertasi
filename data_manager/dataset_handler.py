import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

MAX_UPLOAD_SIZE_MB: int = 50
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_DATASET_EXTENSIONS = (".csv", ".xlsx", ".xls")


def sanitize_html_text(text: Any) -> str:
    """Escapes HTML special characters from LLM or user text before rendering inside HTML blocks."""
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def validate_upload_file_size(
    file_path_or_buffer,
    max_bytes: int = MAX_UPLOAD_SIZE_BYTES,
) -> Optional[str]:
    """Validates file size (if determinable) and extension before loading."""
    if isinstance(file_path_or_buffer, (str, Path)):
        path = Path(file_path_or_buffer)
        if not path.exists():
            return f"File tidak ditemukan: {path}"
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_DATASET_EXTENSIONS:
            return f"Format file '{suffix}' tidak didukung. Gunakan CSV atau Excel (.csv, .xlsx, .xls)."
        size = path.stat().st_size
        if size == 0:
            return "File dataset kosong (0 bytes)."
        if size > max_bytes:
            return f"Ukuran file ({size / (1024 * 1024):.1f} MB) melebihi batas maksimum {max_bytes // (1024 * 1024)} MB."
        return None

    filename = getattr(file_path_or_buffer, "name", "")
    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix and suffix not in ALLOWED_DATASET_EXTENSIONS:
        return f"Format file '{suffix}' tidak didukung. Gunakan CSV atau Excel (.csv, .xlsx, .xls)."

    size = getattr(file_path_or_buffer, "size", None)
    if size is not None:
        if size == 0:
            return "File dataset kosong (0 bytes)."
        if size > max_bytes:
            return f"Ukuran file ({size / (1024 * 1024):.1f} MB) melebihi batas maksimum {max_bytes // (1024 * 1024)} MB."
    return None


def validate_dataset_dataframe(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    required_cols: Optional[List[str]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates a loaded dataset DataFrame for structural integrity.
    Returns (is_valid, warnings, errors).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        return False, warnings, ["Objek dataset bukan DataFrame yang valid."]
    if df.empty or len(df) == 0:
        errors.append("Dataset tidak memiliki baris data (0 baris).")
        return False, warnings, errors
    if len(df.columns) == 0:
        errors.append("Dataset tidak memiliki kolom.")
        return False, warnings, errors

    # Check duplicate column names
    if len(df.columns) != len(set(df.columns)):
        errors.append("Dataset memiliki nama kolom duplikat.")

    if required_cols:
        missing_req = [c for c in required_cols if c not in df.columns]
        if missing_req:
            errors.append(f"Kolom wajib tidak ditemukan: {', '.join(missing_req)}")

    if text_col is not None:
        if text_col not in df.columns:
            errors.append(f"Kolom teks '{text_col}' tidak ditemukan di dataset.")
        else:
            series = df[text_col]
            null_count = int(series.isna().sum())
            non_null_str = series.dropna().astype(str).str.strip()
            empty_count = int((non_null_str == "").sum())
            invalid_str_count = int(non_null_str.str.lower().isin({"nan", "none", "null"}).sum())
            if null_count > 0:
                warnings.append(f"Ditemukan {null_count} nilai kosong (NaN) pada kolom '{text_col}'.")
            if empty_count > 0:
                warnings.append(f"Ditemukan {empty_count} baris teks kosong/spasi pada kolom '{text_col}'.")
            if invalid_str_count > 0:
                warnings.append(f"Ditemukan {invalid_str_count} baris bernilai literal 'nan'/'null' pada '{text_col}'.")
            valid_rows = len(df) - null_count - empty_count - invalid_str_count
            if valid_rows <= 0:
                errors.append(f"Tidak ada teks komentar yang valid pada kolom '{text_col}'.")

    return len(errors) == 0, warnings, errors


def load_dataset_file(
    file_path_or_buffer,
    max_bytes: int = MAX_UPLOAD_SIZE_BYTES,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Loads and validates CSV or Excel dataset into a pandas DataFrame."""
    size_err = validate_upload_file_size(file_path_or_buffer, max_bytes=max_bytes)
    if size_err:
        return None, size_err

    try:
        if hasattr(file_path_or_buffer, "seek"):
            try:
                file_path_or_buffer.seek(0)
            except Exception:
                pass

        if isinstance(file_path_or_buffer, (str, Path)):
            path_str = str(file_path_or_buffer).lower()
            if path_str.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)
        else:
            filename = getattr(file_path_or_buffer, "name", "").lower()
            if filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)

        if df is None or df.empty:
            return None, "Dataset kosong atau tidak memiliki baris data."
        # Strip surrounding whitespace in column names for safety
        df.columns = [str(c).strip() for c in df.columns]
        return df, None
    except pd.errors.EmptyDataError:
        return None, "File CSV kosong atau tidak memiliki header yang valid."
    except Exception as e:
        return None, f"Gagal membaca dataset: {e}"


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detects the most probable column containing text/comments."""
    if df is None or len(df.columns) == 0:
        return None

    priority_candidates = [
        "comment", "komentar", "text", "teks", "tweet", "content",
        "sentence", "clean_text", "raw_text", "message",
    ]
    cols_lower = {str(col).lower(): col for col in df.columns}

    for cand in priority_candidates:
        if cand in cols_lower:
            return cols_lower[cand]

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "string":
            return col

    return df.columns[0]


def detect_ground_truth_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detects an explicit validated human ground-truth column if present.
    Important: upstream raw 'sentiment' or LLM 'cyberbullying' columns are NOT treated as human ground truth.
    """
    if df is None or len(df.columns) == 0:
        return None

    candidates = [
        "human_ground_truth",
        "human_label",
        "gold_label",
        "ground_truth",
        "validated_label",
        "manual_label",
        "true_label",
    ]
    cols_lower = {str(col).lower(): col for col in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None
