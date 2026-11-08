
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]

INTERACTION_FILE = ROOT / "data/processed/clean_interactions.csv"
CATALOG_FILE = ROOT / "data/processed/interaction_template_catalog.csv"

OUTPUT_DIR = ROOT / "data/processed/stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load datasets
df = pd.read_csv(INTERACTION_FILE)
catalog = pd.read_csv(CATALOG_FILE)

# 2. Normalize text
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.!?]+$", "", text)
    return text

# 3. Replace actual drug names with placeholders
def extract_template(row):
    description = str(row["Interaction Description"])

    replacements = [
        (str(row["Drug 1"]), "drug1"),
        (str(row["Drug 2"]), "drug2"),
    ]

    # Replace longer drug names first to avoid substring conflicts
    replacements.sort(key=lambda x: len(x[0]), reverse=True)

    for drug, placeholder in replacements:
        if drug.strip():
            description = re.sub(
                re.escape(drug),
                placeholder,
                description,
                flags=re.IGNORECASE
            )

    return clean_text(description)

print("Extracting interaction templates...")

df["extracted_template"] = df.apply(extract_template, axis=1)

catalog["normalized_template"] = (
    catalog["template"].apply(clean_text)
)

# 4. Compare extracted templates with catalogue
catalog_templates = set(catalog["normalized_template"])

df["template_matched"] = (
    df["extracted_template"].isin(catalog_templates)
)

print("\n========== TEMPLATE MATCHING ==========")

print("Total interaction records:", len(df))
print("Unique extracted templates:", df["extracted_template"].nunique())
print("Catalogue templates:", len(catalog_templates))

matched = df[df["template_matched"]].copy()
unmatched = df[~df["template_matched"]].copy()

print("Matched records:", len(matched))
print("Unmatched records:", len(unmatched))

print(
    "Template matching percentage:",
    round(100 * len(matched) / len(df), 2), "%"
)

# 5. Save extracted template frequencies
template_counts = (
    df.groupby("extracted_template")
    .size()
    .reset_index(name="record_count")
    .sort_values("record_count", ascending=False)
)

template_counts.to_csv(
    OUTPUT_DIR / "extracted_template_distribution.csv",
    index=False
)

# 6. Save unmatched descriptions
unmatched[
    ["Drug 1", "Drug 2", "Interaction Description",
     "extracted_template"]
].to_csv(
    OUTPUT_DIR / "unmatched_templates.csv",
    index=False
)

# 7. Attach catalogue information to matched records
labelled = matched.merge(
    catalog[
        [
            "normalized_template",
            "interaction_family",
            "direction",
            "affected_drug"
        ]
    ],
    left_on="extracted_template",
    right_on="normalized_template",
    how="left",
    validate="many_to_one"
)

labelled.drop(
    columns=["normalized_template"],
    inplace=True
)

labelled.to_csv(
    OUTPUT_DIR / "template_label_analysis.csv",
    index=False
)

# 8. Family distribution
print("\n========== INTERACTION FAMILY DISTRIBUTION ==========")

print(labelled["interaction_family"].value_counts(dropna=False))

print("\n========== DIRECTION DISTRIBUTION ==========")

print(labelled["direction"].value_counts(dropna=False))

print("\n========== AFFECTED DRUG DISTRIBUTION ==========")

print(labelled["affected_drug"].value_counts(dropna=False))

# 9. Show unmatched examples
print("\n========== UNMATCHED TEMPLATE EXAMPLES ==========")

for template in unmatched["extracted_template"].drop_duplicates().head(20):
    print("-", template)

# 10. Show most frequent extracted templates
print("\n========== TOP 15 EXTRACTED TEMPLATES ==========")

print(template_counts.head(15).to_string(index=False))

print("\nGenerated files:")
print("- extracted_template_distribution.csv")
print("- unmatched_templates.csv")
print("- template_label_analysis.csv")

print("\nInteraction label inspection completed.")
