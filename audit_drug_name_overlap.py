
from pathlib import Path
from collections import Counter
import pandas as pd
import re
import json

ROOT = Path(__file__).resolve().parent

DDI_FILE = ROOT / "data/shared/validated_interaction_labels.csv"
DDINTER_FILE = (
    ROOT / "severity_module/data/processed/ddinter2/"
    "ddinter_combined.csv"
)

OUT = ROOT / "severity_module/evaluation/ddinter_coverage"
OUT.mkdir(parents=True, exist_ok=True)


def normalize(value):
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


print("Loading datasets...")
ddi = pd.read_csv(DDI_FILE, low_memory=False)
ddinter = pd.read_csv(DDINTER_FILE, low_memory=False)

# Use unique DDIvolve pairs to avoid inflating drug counts
# because some pairs have multiple interaction descriptions.
ddi_pairs = ddi[["Drug 1", "Drug 2"]].drop_duplicates().copy()

ddi_pairs["drug1_norm"] = ddi_pairs["Drug 1"].apply(normalize)
ddi_pairs["drug2_norm"] = ddi_pairs["Drug 2"].apply(normalize)

ddinter_pairs = ddinter[["drug_a", "drug_b"]].drop_duplicates().copy()

ddinter_pairs["drug1_norm"] = ddinter_pairs["drug_a"].apply(normalize)
ddinter_pairs["drug2_norm"] = ddinter_pairs["drug_b"].apply(normalize)

# Count how many unique pairs contain each drug.
ddi_counts = Counter()
for _, row in ddi_pairs.iterrows():
    for drug in {row["drug1_norm"], row["drug2_norm"]}:
        if drug:
            ddi_counts[drug] += 1

ddinter_counts = Counter()
for _, row in ddinter_pairs.iterrows():
    for drug in {row["drug1_norm"], row["drug2_norm"]}:
        if drug:
            ddinter_counts[drug] += 1

ddi_names = set(ddi_counts)
ddinter_names = set(ddinter_counts)

common = ddi_names & ddinter_names
ddi_only = ddi_names - ddinter_names
ddinter_only = ddinter_names - ddi_names

def export_names(names, counts, filename):
    result = pd.DataFrame([
        {
            "normalized_drug_name": name,
            "unique_pairs_containing_drug": counts[name]
        }
        for name in names
    ])

    if not result.empty:
        result = result.sort_values(
            "unique_pairs_containing_drug",
            ascending=False
        )

    result.to_csv(
        OUT / filename,
        index=False,
        encoding="utf-8-sig"
    )

export_names(
    ddi_only,
    ddi_counts,
    "ddinvolve_drugs_not_in_ddinter.csv"
)

export_names(
    ddinter_only,
    ddinter_counts,
    "ddinter_drugs_not_in_ddinvolve.csv"
)

export_names(
    common,
    ddi_counts,
    "shared_drug_names.csv"
)

# Correctly verify repeated DDInter pair-level records.
def pair_key(a, b):
    return " || ".join(sorted([normalize(a), normalize(b)]))

ddinter["pair_key_audit"] = ddinter.apply(
    lambda r: pair_key(r["drug_a"], r["drug_b"]),
    axis=1
)

ddinter["level_normalized"] = (
    ddinter["source_level"].fillna("MISSING").astype(str).str.strip()
)

exact_pair_level_duplicates = int(
    ddinter.duplicated(
        subset=["pair_key_audit", "level_normalized"],
        keep="first"
    ).sum()
)

report = {
    "ddi_unique_drugs": len(ddi_names),
    "ddinter_unique_drugs": len(ddinter_names),
    "shared_exact_normalized_drug_names": len(common),
    "ddi_drugs_not_found_in_ddinter": len(ddi_only),
    "ddinter_drugs_not_found_in_ddi": len(ddinter_only),
    "ddi_drug_name_overlap_percent": round(
        len(common) / len(ddi_names) * 100, 2
    ) if ddi_names else 0,
    "ddinter_order_independent_pair_level_duplicate_records":
        exact_pair_level_duplicates
}

with open(OUT / "drug_name_overlap_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("\n" + "=" * 55)
print("DRUG NAME OVERLAP AUDIT")
print("=" * 55)

for key, value in report.items():
    print(f"{key}: {value}")

print("\nTop 20 DDIvolve drugs not found in DDInter:")
for name, count in Counter({
    n: ddi_counts[n] for n in ddi_only
}).most_common(20):
    print(f"{name}: {count} pairs")

print("\nGenerated files:", OUT)
print("Original datasets were not modified.")
