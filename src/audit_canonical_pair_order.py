
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/processed/stage2/"
    "validated_interaction_labels.csv"
)

df = pd.read_csv(INPUT)

# Verify canonical pair IDs against normalized drug keys.
expected = (
    df[["drug1_key", "drug2_key"]]
    .apply(lambda row: " || ".join(sorted(row)), axis=1)
)

df["expected_canonical_pair_id"] = expected

mismatch = df[
    df["canonical_pair_id"] != df["expected_canonical_pair_id"]
]

print("\n========== CANONICAL ORDER AUDIT ==========")

print("Total records:", len(df))
print("Canonical ordering mismatches:", len(mismatch))

if not mismatch.empty:
    print(mismatch[
        [
            "canonical_pair_id",
            "drug1_key",
            "drug2_key",
            "expected_canonical_pair_id"
        ]
    ].head(20).to_string(index=False))

# Check whether affected-drug identity belongs to the pair.
valid = (
    (df["affected_drug_key"] == "drug_pair") |
    (df["affected_drug_key"] == df["drug1_key"]) |
    (df["affected_drug_key"] == df["drug2_key"])
)

print("Affected drug identity consistency:", valid.all())

mismatch.to_csv(
    ROOT / "data/processed/stage2/"
    "canonical_order_mismatches.csv",
    index=False
)

print("\n========== AUDIT COMPLETE ==========")
