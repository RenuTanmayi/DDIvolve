
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/stage2"

# Load arrays without loading everything into RAM
X_desc = np.load(
    OUT / "pair_descriptors.npy",
    mmap_mode="r"
)

X_fp = np.load(
    OUT / "pair_fingerprints.npy",
    mmap_mode="r"
)

labels = pd.read_csv(OUT / "pair_labels.csv")

print("\n========== STAGE 2 OUTPUT VALIDATION ==========")

print("Descriptor matrix:", X_desc.shape)
print("Fingerprint matrix:", X_fp.shape)
print("Label records:", len(labels))

# 1. Check row alignment
assert X_desc.shape[0] == len(labels)
assert X_fp.shape[0] == len(labels)

print("\nRow alignment: PASSED")

# 2. Validate descriptors in chunks
chunk_size = 5000
descriptor_errors = 0

for start in range(0, len(X_desc), chunk_size):
    chunk = X_desc[start:start + chunk_size]
    descriptor_errors += int((~np.isfinite(chunk)).sum())

print("Invalid descriptor values:", descriptor_errors)

# 3. Validate fingerprints in chunks
invalid_fp = 0

for start in range(0, len(X_fp), chunk_size):
    chunk = X_fp[start:start + chunk_size]

    invalid_fp += int(
        np.count_nonzero((chunk != 0) & (chunk != 1))
    )

print("Invalid fingerprint values:", invalid_fp)

# 4. Check label completeness
target_columns = [
    "interaction_family",
    "direction",
    "affected_drug"
]

print("\n========== LABEL VALIDATION ==========")

print("Missing labels:")
print(labels[target_columns].isna().sum())

print("\nInteraction family distribution:")
print(labels["interaction_family"].value_counts())

print("\nDirection distribution:")
print(labels["direction"].value_counts())

print("\nAffected drug distribution:")
print(labels["affected_drug"].value_counts())

# 5. Pair validation
print("\n========== PAIR VALIDATION ==========")

print("Unique canonical pairs:",
      labels["canonical_pair_id"].nunique())

print("Duplicate pair-label records:",
      labels.duplicated(
          subset=[
              "canonical_pair_id",
              "interaction_family",
              "direction",
              "affected_drug"
          ]
      ).sum())

# 6. Final status
passed = (
    descriptor_errors == 0
    and invalid_fp == 0
    and labels[target_columns].notna().all().all()
    and X_desc.shape[0] == X_fp.shape[0] == len(labels)
)

print("\n========== FINAL STATUS ==========")
print("STAGE 2 VALIDATION:", "PASSED" if passed else "FAILED")
