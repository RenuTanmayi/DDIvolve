
from pathlib import Path
import pandas as pd
import json
import re

ROOT = Path(__file__).resolve().parent

DDI_FILE = ROOT / "data/shared/validated_interaction_labels.csv"
MATCHED_FILE = ROOT / "severity_module/evaluation/ddinter_coverage/ddinter_matched_pairs.csv"

OUT_DIR = ROOT / "severity_module/data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_name(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())

def pair_key(a, b):
    return tuple(sorted([normalize_name(a), normalize_name(b)]))

# Load datasets
ddi = pd.read_csv(DDI_FILE)
matched = pd.read_csv(MATCHED_FILE)

print("DDIvolve columns:", list(ddi.columns))
print("Matched DDInter columns:", list(matched.columns))

# Check required DDIvolve columns
required = [
    "Drug 1", "Drug 2", "drug1_key", "drug2_key",
    "canonical_pair_id", "Interaction Description",
    "interaction_family"
]

missing = [c for c in required if c not in ddi.columns]
if missing:
    raise ValueError(f"Missing DDIvolve columns: {missing}")

# Identify DDInter name and severity columns
def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find any of these columns: {candidates}")

dd_a = find_column(matched, ["ddinter_drug_a", "drug_a", "Drug_A"])
dd_b = find_column(matched, ["ddinter_drug_b", "drug_b", "Drug_B"])
level_col = find_column(matched, ["source_level", "Level", "level"])

source_files_col = next(
    (c for c in ["source_files", "source_file"] if c in matched.columns),
    None
)

# Build DDInter lookup using order-independent exact normalized names
ddinter_lookup = {}

for _, row in matched.iterrows():
    key = pair_key(row[dd_a], row[dd_b])
    if not all(key):
        continue

    ddinter_lookup[key] = {
        "source_reported_level": str(row[level_col]).strip(),
        "ddinter_drug_a": row[dd_a],
        "ddinter_drug_b": row[dd_b],
        "source_files": (
            row[source_files_col] if source_files_col else ""
        )
    }

# Group original DDIvolve records by canonical pair
records = []

for canonical_id, group in ddi.groupby("canonical_pair_id", dropna=False):
    first = group.iloc[0]
    key = pair_key(first["Drug 1"], first["Drug 2"])

    if key not in ddinter_lookup:
        continue

    evidence = ddinter_lookup[key]
    reported_level = evidence["source_reported_level"]

    allowed_levels = {"Minor", "Moderate", "Major", "Unknown"}
    if reported_level not in allowed_levels:
        reported_level = "Unknown"

    if reported_level == "Unknown":
        status = "Source reports unknown"
    else:
        status = "Source level available; independent clinical review pending"

    records.append({
        "canonical_pair_id": canonical_id,
        "drug1_key": first["drug1_key"],
        "drug2_key": first["drug2_key"],
        "drug1_name": first["Drug 1"],
        "drug2_name": first["Drug 2"],
        "interaction_description": " | ".join(
            sorted(group["Interaction Description"].dropna().astype(str).unique())
        ),
        "interaction_families": " | ".join(
            sorted(group["interaction_family"].dropna().astype(str).unique())
        ),
        "severity": reported_level,
        "source_reported_level": reported_level,
        "clinical_effect": "",
        "clinical_significance": "",
        "management": "",
        "evidence_source": "DDInter 2.0",
        "evidence_quality": "Not independently graded",
        "assessment_status": status,
        "ddinter_drug_a": evidence["ddinter_drug_a"],
        "ddinter_drug_b": evidence["ddinter_drug_b"],
        "source_files": evidence["source_files"],
        "evidence_scope": (
            "Source-reported interaction level only; "
            "independent clinical verification not performed"
        )
    })

result = pd.DataFrame(records)

output_file = OUT_DIR / "clinical_evidence.csv"
result.to_csv(output_file, index=False)

# Audit report
audit = {
    "matched_canonical_pairs": int(len(result)),
    "severity_counts": {
        str(k): int(v)
        for k, v in result["severity"].value_counts().items()
    },
    "evidence_source": "DDInter 2.0",
    "independent_clinical_verification": False,
    "unmatched_pairs_excluded": True,
    "clinical_effect_and_management_populated": False
}

with open(OUT_DIR / "clinical_evidence_audit.json", "w") as f:
    json.dump(audit, f, indent=4)

print("\nCLINICAL EVIDENCE TABLE CREATED")
print("Rows:", len(result))
print("\nSeverity counts:")
print(result["severity"].value_counts())
print("\nSaved:", output_file)
print("Audit:", OUT_DIR / "clinical_evidence_audit.json")
