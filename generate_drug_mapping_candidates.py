
from pathlib import Path
import pandas as pd
import re
from difflib import SequenceMatcher

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
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def similarity(a, b):
    return round(100 * SequenceMatcher(None, a, b).ratio(), 2)


print("Loading drug datasets...")

ddi = pd.read_csv(DDI_FILE, low_memory=False)
ddinter = pd.read_csv(DDINTER_FILE, low_memory=False)

# Count DDIvolve pairs containing each drug.
drug_counts = {}

for col in ["Drug 1", "Drug 2"]:
    for name in ddi[col].dropna().unique():
        key = normalize(name)
        if key:
            drug_counts[key] = drug_counts.get(key, 0) + (
                ddi.loc[
                    ddi[col].apply(normalize) == key,
                    "pair_id"
                ].nunique()
            )

ddi_names = set(drug_counts)

ddinter_names = {
    normalize(x)
    for x in pd.concat([
        ddinter["drug_a"],
        ddinter["drug_b"]
    ]).dropna()
}

# Keep one display name for each normalized DDInter name.
display_names = {}

for name in pd.concat([
    ddinter["drug_a"],
    ddinter["drug_b"]
]).dropna().unique():
    key = normalize(name)
    display_names.setdefault(key, str(name))

unmatched = sorted(ddi_names - ddinter_names)

print("Unmatched DDIvolve drug names:", len(unmatched))
print("Generating top-five similarity candidates...")

rows = []

for drug in unmatched:
    candidates = []

    for dd_name in ddinter_names:
        score = similarity(drug, dd_name)
        candidates.append((score, dd_name))

    candidates.sort(reverse=True)

    for rank, (score, candidate) in enumerate(candidates[:5], start=1):
        rows.append({
            "ddi_drug_name": drug,
            "ddi_pairs_affected": drug_counts[drug],
            "candidate_rank": rank,
            "ddinter_candidate_name": display_names[candidate],
            "similarity_score": score,
            "mapping_decision": "REVIEW_REQUIRED"
        })

result = pd.DataFrame(rows)

result.to_csv(
    OUT / "drug_name_mapping_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTop candidate examples:")
print(result.head(25).to_string(index=False))

print("\nGenerated:")
print(OUT / "drug_name_mapping_candidates.csv")
print("\nNo automatic drug mappings were accepted.")
