import sys
import os
import shutil
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.schema import parse_confidence_score, sanitize_annotation_dict
from core.parser import extract_json_from_llama

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes confidence to numeric float [0.0, 1.0] and fills missing/empty fields
    with standard taxonomy tokens ('NONE', 'NEU', 'E0', etc.).
    """
    cleaned_df = df.copy()

    # Identify annotation columns
    annotation_cols = [
        "cyberbullying", "type", "emotion", "cultural_context",
        "implicit_level", "severity", "confidence", "explanation"
    ]
    for col in annotation_cols:
        if col not in cleaned_df.columns:
            cleaned_df[col] = None

    for idx in range(len(cleaned_df)):
        row_dict = {col: cleaned_df.at[idx, col] for col in annotation_cols}

        # Check if row was a "No JSON brackets found" error and try to re-parse
        exp_str = str(row_dict.get("explanation", ""))
        err_str = str(cleaned_df.at[idx, "error"]) if "error" in cleaned_df.columns else ""
        if "No JSON brackets found" in exp_str or "No JSON brackets found" in err_str:
            repaired = extract_json_from_llama(exp_str)
            if repaired.get("error") != "No JSON brackets found":
                for k, v in repaired.items():
                    if k in annotation_cols:
                        row_dict[k] = v
                if "error" in cleaned_df.columns:
                    cleaned_df.at[idx, "error"] = None

        # Sanitize dictionary
        sanitized = sanitize_annotation_dict(row_dict)

        for col, val in sanitized.items():
            cleaned_df.at[idx, col] = val

    # Enforce float type on confidence column
    cleaned_df["confidence"] = pd.to_numeric(cleaned_df["confidence"], errors="coerce").fillna(0.85).round(2)

    return cleaned_df

def process_file(file_path: Path):
    print(f"Cleaning checkpoint: {file_path.name}...")
    df = pd.read_csv(file_path)
    total_before = len(df)
    
    # Check if confidence has non-numeric values
    non_numeric_conf = df["confidence"].apply(lambda x: not isinstance(x, (int, float)) and not str(x).replace(".", "", 1).isdigit() if pd.notna(x) else True).sum()
    print(f"  Total rows: {total_before}")
    print(f"  Rows with non-numeric or missing confidence before: {non_numeric_conf}")

    # Backup original
    backup_path = file_path.with_suffix(".csv.bak")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  Created backup at: {backup_path.name}")

    # Clean
    cleaned = clean_dataframe(df)

    # Save cleaned
    cleaned.to_csv(file_path, index=False, encoding="utf-8")
    
    # Stats after
    empty_type = (cleaned["type"] == "").sum() + cleaned["type"].isna().sum()
    empty_cult = (cleaned["cultural_context"] == "").sum() + cleaned["cultural_context"].isna().sum()
    conf_types = cleaned["confidence"].apply(type).value_counts().to_dict()
    print(f"  Cleaned successfully!")
    print(f"  Empty 'type' remaining: {empty_type}")
    print(f"  Empty 'cultural_context' remaining: {empty_cult}")
    print(f"  Confidence data types: {conf_types}")
    print(f"  Sample confidence values: {cleaned['confidence'].head(5).tolist()}")

def main():
    checkpoints_dir = PROJECT_DIR / "output" / "checkpoints"
    csv_files = list(checkpoints_dir.glob("*.csv"))
    if not csv_files:
        print("No checkpoint files found in output/checkpoints.")
        return

    for csv_file in csv_files:
        process_file(csv_file)

if __name__ == "__main__":
    main()
