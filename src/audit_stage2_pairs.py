
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/stage2/pair_labels.csv"
OUT = ROOT / "data/processed/stage2"

df = pd.read_csv(INPUT)

target_cols = [
    "interaction_family",
    "direction",
    "affected_drug"
]

print("\n========== PAIR-LEVEL AUDIT ==========")
print("Total records:", len(df))
print("Unique canonical pairs:", df["canonical_pair_id"].nunique())

# 1. Find exact duplicate pair-label records
duplicate_cols = ["canonical_pair_id"] + target_cols

duplicates = df[
    df.duplicated(subset=duplicate_cols, keep=False)
].sort_values(duplicate_cols)

duplicates.to_csv(
    OUT / "duplicate_pair_labels.csv",
    index=False
)

print("\nExact duplicate pair-label rows:", len(duplicates))
print("Saved: duplicate_pair_labels.csv")

# 2. Identify pairs with multiple distinct label combinations
df["_label_combination"] = (
    df[target_cols].astype(str).agg(" | ".join, axis=1)
)

label_counts = (
    df.groupby("canonical_pair_id")["_label_combination"]
    .nunique()
)

multi_label_ids = label_counts[label_counts > 1].index

multi_label_pairs = df[
    df["canonical_pair_id"].isin(multi_label_ids)
].sort_values("canonical_pair_id")

multi_label_pairs = multi_label_pairs.drop(
    columns=["_label_combination"]
)

multi_label_pairs.to_csv(
    OUT / "pairs_with_multiple_labels.csv",
    index=False
)

print("\nPairs with multiple label combinations:", len(multi_label_ids))
print("Records belonging to these pairs:", len(multi_label_pairs))
print("Saved: pairs_with_multiple_labels.csv")

# 3. Summarize records per canonical pair
pair_sizes = df.groupby("canonical_pair_id").size()

print("\n========== RECORDS PER PAIR ==========")
print("Pairs with one record:", (pair_sizes == 1).sum())
print("Pairs with multiple records:", (pair_sizes > 1).sum())
print("Maximum records for one pair:", pair_sizes.max())

print("\n========== AUDIT COMPLETE ==========")
