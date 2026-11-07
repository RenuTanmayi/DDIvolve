
from pathlib import Path
import pandas as pd
import json
import re

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "severity_module" / "data" / "raw" / "ddinter2"
OUTPUT_DIR = ROOT / "severity_module" / "data" / "processed" / "ddinter2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("ddinter_downloads_code_*.csv"))

if len(files) != 8:
    raise FileNotFoundError(
        f"Expected 8 DDInter CSV files, but found {len(files)} "
        f"in {INPUT_DIR}"
    )

frames = []

for file in files:
    print(f"Reading: {file.name}")

    df = pd.read_csv(file, encoding="utf-8-sig")

    if len(df.columns) < 5:
        raise ValueError(
            f"Unexpected column structure in {file.name}: "
            f"{list(df.columns)}"
        )

    # Rename by column position because DDInterID occurs twice.
    df = df.iloc[:, :5].copy()
    df.columns = [
        "drug_a_id",
        "drug_a",
        "drug_b_id",
        "drug_b",
        "source_level"
    ]

    df["source_file"] = file.name

    for col in [
        "drug_a_id", "drug_a", "drug_b_id",
        "drug_b", "source_level"
    ]:
        df[col] = df[col].astype("string").str.strip()

    frames.append(df)

# Preserve every original interaction record.
combined = pd.concat(frames, ignore_index=True)

# Normalize names only for comparison.
def normalize_name(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value

combined["drug_a_norm"] = combined["drug_a"].apply(normalize_name)
combined["drug_b_norm"] = combined["drug_b"].apply(normalize_name)

# Canonical pair key: order-independent.
combined["canonical_drug_pair"] = combined.apply(
    lambda row: " | ".join(sorted([
        row["drug_a_norm"],
        row["drug_b_norm"]
    ])),
    axis=1
)

# Identify incomplete records.
invalid_mask = (
    (combined["drug_a_norm"] == "") |
    (combined["drug_b_norm"] == "") |
    (combined["source_level"].isna())
)

# Retain all rows in combined output.
combined.to_csv(
    OUTPUT_DIR / "ddinter_combined.csv",
    index=False,
    encoding="utf-8-sig"
)

# Summarize records by unordered drug pair.
valid = combined.loc[~invalid_mask].copy()

pair_summary = (
    valid.groupby("canonical_drug_pair")
    .agg(
        drug_a=("drug_a", "first"),
        drug_b=("drug_b", "first"),
        record_count=("canonical_drug_pair", "size"),
        reported_levels=(
            "source_level",
            lambda x: " | ".join(sorted(x.dropna().unique()))
        ),
        source_files=(
            "source_file",
            lambda x: " | ".join(sorted(x.unique()))
        )
    )
    .reset_index()
)

pair_summary["level_count"] = pair_summary[
    "reported_levels"
].apply(lambda x: len(x.split(" | ")) if x else 0)

pair_summary.to_csv(
    OUTPUT_DIR / "ddinter_pair_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# Audit statistics.
level_counts = (
    combined["source_level"]
    .fillna("MISSING")
    .value_counts()
    .to_dict()
)

report = {
    "files_processed": len(files),
    "file_names": [f.name for f in files],
    "total_records": int(len(combined)),
    "unique_unordered_drug_pairs": int(
        valid["canonical_drug_pair"].nunique()
    ),
    "duplicate_records_beyond_unique_pairs": int(
        len(valid) - valid["canonical_drug_pair"].nunique()
    ),
    "incomplete_records": int(invalid_mask.sum()),
    "unique_drugs": int(
        pd.concat([
            valid["drug_a_norm"],
            valid["drug_b_norm"]
        ]).nunique()
    ),
    "reported_level_counts": {
        str(k): int(v) for k, v in level_counts.items()
    },
    "pairs_with_multiple_reported_levels": int(
        (pair_summary["level_count"] > 1).sum()
    )
}

with open(
    OUTPUT_DIR / "ddinter_audit_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=4)

print("\n" + "=" * 55)
print("DDINTER CONSOLIDATION COMPLETE")
print("=" * 55)

for key, value in report.items():
    print(f"{key}: {value}")

print("\nGenerated files:")
print(OUTPUT_DIR / "ddinter_combined.csv")
print(OUTPUT_DIR / "ddinter_pair_summary.csv")
print(OUTPUT_DIR / "ddinter_audit_report.json")

print("\nOriginal DDInter files were not modified.")
