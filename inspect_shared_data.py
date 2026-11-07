
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = BASE_DIR / "data" / "shared"

FILES = [
    "clean_drugs.csv",
    "interaction_template_catalog.csv",
    "validated_interaction_labels.csv",
    "validated_pair_summary.csv"
]

print("=" * 75)
print("DDIVOLVE — SHARED DATASET SCHEMA INSPECTION")
print("=" * 75)

for filename in FILES:

    path = SHARED_DIR / filename

    print("\n" + "=" * 75)
    print(f"FILE: {filename}")
    print("=" * 75)

    if not path.exists():
        print("ERROR: File not found")
        continue

    df = pd.read_csv(path, low_memory=False)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nCOLUMN NAMES:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nDATA TYPES:")
    print(df.dtypes.to_string())

    print("\nMISSING VALUES:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing):
        print(missing.to_string())
    else:
        print("No missing values")

    print("\nFIRST 3 RECORDS:")
    print(df.head(3).to_string(index=False))

    print("\nDUPLICATE ROWS:", df.duplicated().sum())

print("\n" + "=" * 75)
print("SCHEMA INSPECTION COMPLETED")
print("=" * 75)
