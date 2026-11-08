from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/processed/consolidated_features/"
    "consolidated_molecular_features.csv"
)

df = pd.read_csv(INPUT)

print("\n========== CONSOLIDATED DATASET INSPECTION ==========")

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes.to_string())

print("\nFirst 5 records:")
print(df.head().to_string())

print("\nMissing values:")
print(df.isna().sum().to_string())

print("\nDuplicate rows:", df.duplicated().sum())

print("\n========== INSPECTION COMPLETE ==========")