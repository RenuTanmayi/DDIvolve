
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/processed/stage2/duplicate_pair_labels.csv"

df = pd.read_csv(FILE)

print("\n========== EXACT DUPLICATE PAIR RECORDS ==========")

print("Rows:", len(df))
print("\nAvailable columns:")
print(df.columns.tolist())

for pair_id, group in df.groupby("canonical_pair_id"):
    print("\n----------------------------------------")
    print("Canonical pair:", pair_id)
    print(group.to_string(index=False))

print("\n========== INSPECTION COMPLETE ==========")
