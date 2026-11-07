
from pathlib import Path
import pandas as pd
import json
import re

ROOT = Path(__file__).resolve().parent

DDI_FILE = (
    ROOT / "data" / "shared" /
    "validated_interaction_labels.csv"
)

DDINTER_FILE = (
    ROOT / "severity_module" / "data" / "processed" /
    "ddinter2" / "ddinter_combined.csv"
)

OUT = ROOT / "severity_module" / "evaluation" / "ddinter_coverage"
OUT.mkdir(parents=True, exist_ok=True)


def normalize_name(value):
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def make_pair_key(a, b):
    a = normalize_name(a)
    b = normalize_name(b)

    if not a or not b:
        return ""

    return " || ".join(sorted([a, b]))


print("Loading DDIvolve labels...")
ddi = pd.read_csv(DDI_FILE, low_memory=False)

print("Loading DDInter records...")
ddinter = pd.read_csv(DDINTER_FILE, low_memory=False)

required_ddi = {"Drug 1", "Drug 2"}
required_ddinter = {"drug_a", "drug_b", "source_level"}

if not required_ddi.issubset(ddi.columns):
    raise ValueError(f"Missing DDIvolve columns: {required_ddi - set(ddi.columns)}")

if not required_ddinter.issubset(ddinter.columns):
    raise ValueError(
        f"Missing DDInter columns: "
        f"{required_ddinter - set(ddinter.columns)}"
    )

# Build order-independent pair keys.
ddi["comparison_pair_key"] = ddi.apply(
    lambda r: make_pair_key(r["Drug 1"], r["Drug 2"]),
    axis=1
)

ddinter["comparison_pair_key"] = ddinter.apply(
    lambda r: make_pair_key(r["drug_a"], r["drug_b"]),
    axis=1
)

ddi = ddi[ddi["comparison_pair_key"] != ""].copy()
ddinter = ddinter[ddinter["comparison_pair_key"] != ""].copy()

# Unique DDIvolve pairs.
ddi_pairs = (
    ddi.groupby("comparison_pair_key")
    .agg(
        drug_a=("Drug 1", "first"),
        drug_b=("Drug 2", "first"),
        interaction_records=("comparison_pair_key", "size"),
        interaction_families=(
            "interaction_family",
            lambda x: " | ".join(sorted(x.dropna().astype(str).unique()))
        ),
        descriptions=(
            "Interaction Description",
            lambda x: " || ".join(sorted(x.dropna().astype(str).unique()))
        )
    )
    .reset_index()
)

# Unique DDInter pairs.
ddinter_pairs = (
    ddinter.groupby("comparison_pair_key")
    .agg(
        ddinter_drug_a=("drug_a", "first"),
        ddinter_drug_b=("drug_b", "first"),
        source_level=("source_level", "first"),
        ddinter_record_count=("comparison_pair_key", "size"),
        source_files=(
            "source_file",
            lambda x: " | ".join(sorted(x.dropna().astype(str).unique()))
        )
    )
    .reset_index()
)

# Exact normalized-name intersection.
comparison = ddi_pairs.merge(
    ddinter_pairs,
    on="comparison_pair_key",
    how="left",
    indicator=True
)

matched = comparison[comparison["_merge"] == "both"].copy()
unmatched = comparison[comparison["_merge"] == "left_only"].copy()

ddinter_only = ddinter_pairs[
    ~ddinter_pairs["comparison_pair_key"].isin(
        ddi_pairs["comparison_pair_key"]
    )
].copy()

# Save pair-level comparison files.
matched.to_csv(
    OUT / "ddinter_matched_pairs.csv",
    index=False,
    encoding="utf-8-sig"
)

unmatched.to_csv(
    OUT / "ddinvolve_unmatched_pairs.csv",
    index=False,
    encoding="utf-8-sig"
)

ddinter_only.to_csv(
    OUT / "ddinter_only_pairs.csv",
    index=False,
    encoding="utf-8-sig"
)

# Exact duplicate record audit.
exact_duplicate_count = int(
    ddinter.duplicated(
        subset=["drug_a_norm", "drug_b_norm", "source_level"],
        keep="first"
    ).sum()
) if {"drug_a_norm", "drug_b_norm"}.issubset(ddinter.columns) else None

# Source-level counts among matched unique pairs.
matched_level_counts = (
    matched["source_level"]
    .fillna("MISSING")
    .value_counts()
    .to_dict()
)

total_ddi_pairs = len(ddi_pairs)
matched_count = len(matched)

report = {
    "ddi_interaction_rows": int(len(ddi)),
    "ddi_unique_pairs": int(total_ddi_pairs),
    "ddinter_interaction_rows": int(len(ddinter)),
    "ddinter_unique_pairs": int(len(ddinter_pairs)),
    "exact_name_matched_pairs": int(matched_count),
    "ddinvolve_unmatched_pairs": int(len(unmatched)),
    "ddinter_only_pairs": int(len(ddinter_only)),
    "ddinvolve_pair_coverage_percent": round(
        matched_count / total_ddi_pairs * 100, 2
    ) if total_ddi_pairs else 0,
    "matched_pairs_by_ddinter_level": {
        str(k): int(v) for k, v in matched_level_counts.items()
    },
    "exact_duplicate_ddinter_records": exact_duplicate_count
}

with open(OUT / "coverage_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\n" + "=" * 60)
print("DDIVOLVE × DDINTER COVERAGE REPORT")
print("=" * 60)

for key, value in report.items():
    print(f"{key}: {value}")

print("\nOutput folder:", OUT)
print("Original datasets were not modified.")
