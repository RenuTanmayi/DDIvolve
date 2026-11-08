
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/stage2/pair_labels_with_affected_drug_key.csv"
OUTPUT = ROOT / "data/processed/stage2"

df = pd.read_csv(INPUT)

counts = df.groupby("canonical_pair_id")[
    "affected_drug_key"
].nunique()

multi_ids = counts[counts > 1].index

multi = df[df["canonical_pair_id"].isin(multi_ids)].copy()

# Preserve original descriptions and all label fields.
columns = [
    "canonical_pair_id",
    "Drug 1",
    "Drug 2",
    "drug1_key",
    "drug2_key",
    "interaction_family",
    "direction",
    "affected_drug",
    "affected_drug_key",
    "Interaction Description"
]

multi = multi[columns].sort_values(
    ["canonical_pair_id", "interaction_family"]
)

multi.to_csv(
    OUTPUT / "multilabel_description_audit.csv",
    index=False
)

# Print first 10 complete pairs.
print("\n========== MULTILABEL DESCRIPTION AUDIT ==========")

for pair_id, group in multi.groupby("canonical_pair_id"):
    print("\nPAIR:", pair_id)

    for _, row in group.iterrows():
        print("\nFamily:", row["interaction_family"])
        print("Direction:", row["direction"])
        print("Affected drug:", row["affected_drug_key"])
        print("Description:", row["Interaction Description"])

    if pair_id == multi["canonical_pair_id"].drop_duplicates().iloc[9]:
        break

print("\nTotal multi-label pairs:", len(multi_ids))
print("Total rows:", len(multi))

print("\nSaved: multilabel_description_audit.csv")
