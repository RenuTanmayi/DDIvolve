
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/processed/stage2/"
    "pair_labels_with_affected_drug_key.csv"
)

OUTPUT = ROOT / "data/processed/stage2"

df = pd.read_csv(INPUT)

# Remove only exact duplicate records.
# Distinct descriptions are preserved.
before = len(df)

df = df.drop_duplicates(
    subset=[
        "canonical_pair_id",
        "interaction_family",
        "direction",
        "affected_drug_key",
        "Interaction Description"
    ]
).copy()

removed = before - len(df)

# Validate essential labels.
required = [
    "canonical_pair_id",
    "drug1_key",
    "drug2_key",
    "interaction_family",
    "direction",
    "affected_drug_key"
]

for col in required:
    missing = df[col].isna().sum()
    print(f"{col}: {missing} missing")

# Check valid affected-drug identities.
valid_targets = (
    (df["affected_drug_key"] == "drug_pair") |
    (df["affected_drug_key"] == df["drug1_key"]) |
    (df["affected_drug_key"] == df["drug2_key"])
)

invalid = df[~valid_targets]

print("\n========== LABEL CONSOLIDATION ==========")
print("Original rows:", before)
print("Exact duplicate rows removed:", removed)
print("Final rows:", len(df))
print("Invalid affected-drug identities:", len(invalid))

# Preserve all distinct interaction statements.
df.to_csv(
    OUTPUT / "validated_interaction_labels.csv",
    index=False
)

# Pair-level statistics.
pair_summary = df.groupby("canonical_pair_id").agg(
    interaction_count=("interaction_family", "size"),
    unique_families=("interaction_family", "nunique"),
    unique_affected_targets=("affected_drug_key", "nunique")
).reset_index()

pair_summary.to_csv(
    OUTPUT / "validated_pair_summary.csv",
    index=False
)

print("\nUnique canonical pairs:", df["canonical_pair_id"].nunique())
print("Interaction families:")
print(df["interaction_family"].value_counts())

print("\nSaved:")
print("validated_interaction_labels.csv")
print("validated_pair_summary.csv")

print("\n========== COMPLETE ==========")
