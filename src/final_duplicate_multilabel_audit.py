
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/stage2/validated_interaction_labels.csv"
OUT = ROOT / "data/processed/stage2"

df = pd.read_csv(INPUT)

print("========== FINAL LABEL AUDIT ==========")
print("Total records:", len(df))

# 1. Exact duplicate inspection
duplicates = df[df.duplicated(keep=False)]

print("\nExact duplicate rows:", len(duplicates))
print("Duplicate pairs:",
      duplicates["canonical_pair_id"].nunique())

duplicates.to_csv(
    OUT / "final_exact_duplicates.csv", index=False
)

# 2. Multiple records per canonical pair
pair_counts = (
    df.groupby("canonical_pair_id")
      .size()
      .reset_index(name="record_count")
)

multi_pairs = pair_counts[pair_counts["record_count"] > 1]

print("\nPairs with multiple records:", len(multi_pairs))

# 3. Distinct interaction labels per pair
label_cols = [
    "interaction_family",
    "direction",
    "affected_drug_key"
]

multi_labels = (
    df.groupby("canonical_pair_id")
      .apply(
          lambda g: g[label_cols].drop_duplicates().shape[0],
          include_groups=False
      )
      .reset_index(name="unique_label_count")
)

multi_labels = multi_labels[
    multi_labels["unique_label_count"] > 1
]

print("Pairs with multiple distinct labels:", len(multi_labels))

# Save reports
multi_pairs.to_csv(
    OUT / "final_multiple_record_pairs.csv", index=False
)

multi_labels.to_csv(
    OUT / "final_multiple_label_pairs.csv", index=False
)

print("\n========== AUDIT COMPLETE ==========")
