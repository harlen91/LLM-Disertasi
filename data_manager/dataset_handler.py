import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

def load_dataset_file(file_path_or_buffer) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Loads CSV or Excel dataset into a pandas DataFrame."""
    try:
        if isinstance(file_path_or_buffer, (str, Path)):
            path_str = str(file_path_or_buffer)
            if path_str.endswith(".csv"):
                df = pd.read_csv(file_path_or_buffer)
            elif path_str.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)
        else:
            # UploadedFile from Streamlit
            filename = file_path_or_buffer.name.lower()
            if filename.endswith(".csv"):
                df = pd.read_csv(file_path_or_buffer)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)
        return df, None
    except Exception as e:
        return None, str(e)


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detects the most probable column containing text/comments."""
    priority_candidates = [
        "comment", "komentar", "text", "teks", "tweet", "content", 
        "sentence", "clean_text", "raw_text", "message"
    ]
    cols_lower = {col.lower(): col for col in df.columns}
    
    for cand in priority_candidates:
        if cand in cols_lower:
            return cols_lower[cand]
            
    # Fallback to the first string/object column
    for col in df.columns:
        if df[col].dtype == "object":
            return col
            
    return df.columns[0] if len(df.columns) > 0 else None
