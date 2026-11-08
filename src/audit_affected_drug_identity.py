
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/stage2/pair_labels.csv"
OUTPUT = ROOT / "data/processed/stage2"

df = pd.read_csv(INPUT)

# Convert the relative label into the actual affected drug key.
df["affected_drug_key"] = None

df.loc[
    df["affected_drug"] == "Drug 1",
    "affected_drug_key"
] = df["drug1_key"]

df.loc[
    df["affected_drug"] == "Drug 2",
    "affected_drug_key"
] = df["drug2_key"]

df.loc[
    df["affected_drug"] == "Drug pair",
    "affected_drug_key"
] = "drug_pair"

print("\n========== AFFECTED DRUG IDENTITY AUDIT ==========")

print("\nMissing actual affected-drug identities:",
      df["affected_drug_key"].isna().sum())

print("\nActual affected-drug distribution:")
print(df["affected_drug_key"].value_counts().head(20))

# Check duplicate labels using actual drug identity.
label_cols = [
    "canonical_pair_id",
    "interaction_family",
    "direction",
    "affected_drug_key"
]

duplicates = df[
    df.duplicated(subset=label_cols, keep=False)
].sort_values(label_cols)

duplicates.to_csv(
    OUTPUT / "duplicates_by_actual_affected_drug.csv",
    index=False
)

print("\nDuplicate rows after actual identity correction:",
      len(duplicates))

# Check pairs with multiple actual affected-drug labels.
counts = df.groupby("canonical_pair_id")[
    "affected_drug_key"
].nunique()

print("\nPairs with multiple actual affected-drug identities:",
      (counts > 1).sum())

# Save an audited copy; preserve the original file.
df.to_csv(
    OUTPUT / "pair_labels_with_affected_drug_key.csv",
    index=False
)

print("\nSaved: pair_labels_with_affected_drug_key.csv")
print("\n========== AUDIT COMPLETE ==========")
