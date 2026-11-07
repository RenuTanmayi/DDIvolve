
import pandas as pd
import re

df = pd.read_csv(
    "data/processed/integrated_ddi_dataset.csv"
)

# Normalize descriptions for identifying recurring mechanisms
def normalize_description(row):
    text = str(row["Interaction Description"])

    names = sorted(
        [str(row["Drug 1"]), str(row["Drug 2"])],
        key=len,
        reverse=True
    )

    for i, name in enumerate(names):
        text = re.sub(
            re.escape(name),
            f"DRUG{i+1}",
            text,
            flags=re.IGNORECASE
        )

    return text.lower().strip()


df["template"] = df.apply(normalize_description, axis=1)


# Broad mechanism classification
def classify(text):

    if re.search(r"risk or severity of .*adverse effects", text):
        return "Adverse Effect Risk"

    if "serum concentration" in text:
        return "Serum Concentration"

    if "metabolism" in text:
        return "Drug Metabolism"

    if "therapeutic efficacy" in text:
        return "Therapeutic Efficacy"

    if "absorption" in text:
        return "Drug Absorption"

    if "bioavailability" in text:
        return "Bioavailability"

    if "excretion rate" in text:
        return "Drug Excretion"

    if "may increase" in text or "may decrease" in text:
        return "Pharmacological Activity"

    return "Other / Unclassified"


df["mechanism_category"] = df[
    "Interaction Description"
].apply(classify)


print("\n========== MECHANISM CATEGORY DISTRIBUTION ==========")

counts = df["mechanism_category"].value_counts()

print(counts.to_string())

print("\n========== UNIQUE DESCRIPTION TEMPLATES ==========")

templates = (
    df.groupby("template")
    .size()
    .sort_values(ascending=False)
)

print("Total unique templates:", len(templates))

print("\nTop 30 templates:")

print(templates.head(30).to_string())

print("\n========== UNCLASSIFIED DESCRIPTIONS ==========")

unclassified = df[
    df["mechanism_category"] == "Other / Unclassified"
]

print("Unclassified records:", len(unclassified))

print(
    unclassified["Interaction Description"]
    .head(30)
    .to_string(index=False)
)

print("\n========== ORDERED PAIR ANALYSIS ==========")

ordered_pairs = df[
    ["drug1_key", "drug2_key"]
].drop_duplicates()

unordered_pairs = df.apply(
    lambda r: tuple(sorted([
        str(r["drug1_key"]),
        str(r["drug2_key"])
    ])),
    axis=1
).nunique()

print("Unique ordered pairs:", len(ordered_pairs))
print("Unique unordered pairs:", unordered_pairs)

print("\nAnalysis completed.")
