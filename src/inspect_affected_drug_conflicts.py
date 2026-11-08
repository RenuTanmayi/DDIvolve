
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/stage2/pair_labels_with_affected_drug_key.csv"
OUTPUT = ROOT / "data/processed/stage2"

df = pd.read_csv(INPUT)

label_cols = [
    "canonical_pair_id",
    "interaction_family",
    "direction",
    "affected_drug_key"
]

# --------------------------------------------------
# 1. Inspect duplicate labels
# --------------------------------------------------

duplicate_mask = df.duplicated(
    subset=label_cols,
    keep=False
)

duplicates = df[duplicate_mask].sort_values(label_cols)

duplicates.to_csv(
    OUTPUT / "duplicate_label_inspection.csv",
    index=False
)

print("\n========== DUPLICATE LABEL INSPECTION ==========")
print("Duplicate rows:", len(duplicates))

if not duplicates.empty:
    print(
        duplicates[
            label_cols + ["Drug 1", "Drug 2", "Interaction Description"]
        ].to_string(index=False)
    )

# --------------------------------------------------
# 2. Identify pairs with multiple affected drugs
# --------------------------------------------------

affected_counts = df.groupby("canonical_pair_id")[
    "affected_drug_key"
].nunique()

multi_pair_ids = affected_counts[
    affected_counts > 1
].index

multi_pairs = df[
    df["canonical_pair_id"].isin(multi_pair_ids)
].copy()

multi_pairs = multi_pairs.sort_values(
    ["canonical_pair_id", "interaction_family", "direction"]
)

multi_pairs.to_csv(
    OUTPUT / "multiple_affected_drug_pairs.csv",
    index=False
)

# --------------------------------------------------
# 3. Generate pair-level summary
# --------------------------------------------------

summary = multi_pairs.groupby("canonical_pair_id").agg(
    drug1=("Drug 1", "first"),
    drug2=("Drug 2", "first"),

    affected_drug_keys=(
        "affected_drug_key",
        lambda x: " | ".join(sorted(set(x.dropna().astype(str))))
    ),

    relative_labels=(
        "affected_drug",
        lambda x: " | ".join(sorted(set(x.dropna().astype(str))))
    ),

    interaction_families=(
        "interaction_family",
        lambda x: " | ".join(sorted(set(x.dropna().astype(str))))
    ),

    directions=(
        "direction",
        lambda x: " | ".join(sorted(set(x.dropna().astype(str))))
    ),

    row_count=("canonical_pair_id", "size"),

    unique_descriptions=(
        "Interaction Description",
        "nunique"
    )
).reset_index()

summary.to_csv(
    OUTPUT / "multiple_affected_drug_pair_summary.csv",
    index=False
)

print("\n========== MULTIPLE AFFECTED DRUG SUMMARY ==========")

print("Pairs:", len(summary))
print("\nRows involved:", len(multi_pairs))

print("\nSample of 20 pairs:")
print(summary.head(20).to_string(index=False))

print("\n========== FILES SAVED ==========")
print("1. duplicate_label_inspection.csv")
print("2. multiple_affected_drug_pairs.csv")
print("3. multiple_affected_drug_pair_summary.csv")

print("\n========== INSPECTION COMPLETE ==========")
