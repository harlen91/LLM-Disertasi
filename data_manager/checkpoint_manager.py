import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from config import CHECKPOINTS_DIR, FINAL_DIR

def get_checkpoint_filepath(dataset_name: str = "IICD_batch") -> Path:
    """Generates standardized checkpoint path."""
    clean_name = Path(dataset_name).stem.replace(" ", "_")
    return CHECKPOINTS_DIR / f"checkpoint_{clean_name}.csv"

def save_checkpoint(df: pd.DataFrame, dataset_name: str = "IICD_batch") -> str:
    """Saves current dataframe progress to checkpoint CSV."""
    filepath = get_checkpoint_filepath(dataset_name)
    df.to_csv(filepath, index=False, encoding="utf-8")
    return str(filepath)

def load_checkpoint(dataset_name: str = "IICD_batch") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Loads existing checkpoint if present."""
    filepath = get_checkpoint_filepath(dataset_name)
    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            return df, str(filepath)
        except Exception as e:
            return None, str(e)
    return None, None

def list_existing_checkpoints() -> List[Dict[str, str]]:
    """Lists all available checkpoints in the checkpoints folder."""
    files = list(CHECKPOINTS_DIR.glob("*.csv"))
    results = []
    for f in sorted(files, key=os.path.getmtime, reverse=True):
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = round(f.stat().st_size / 1024, 1)
        results.append({
            "filename": f.name,
            "path": str(f),
            "last_modified": mtime,
            "size_kb": f"{size_kb} KB"
        })
    return results

def export_final_dataset(df: pd.DataFrame, base_filename: str = "IICD_SumoPod_annotated") -> Dict[str, str]:
    """Exports dataset to final directory in CSV and Excel formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = f"{base_filename}_{timestamp}"
    
    csv_path = FINAL_DIR / f"{clean_name}.csv"
    excel_path = FINAL_DIR / f"{clean_name}.xlsx"
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df.to_excel(excel_path, index=False)
    except Exception:
        pass
        
    return {
        "csv": str(csv_path),
        "excel": str(excel_path)
    }
