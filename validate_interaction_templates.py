
import pandas as pd
import re

df = pd.read_csv(
    "data/processed/integrated_ddi_dataset.csv"
)

# Correctly normalize drug names while preserving
# their original Drug 1 / Drug 2 identities.

def make_template(row):
    text = str(row["Interaction Description"])

    drug_map = {
        str(row["Drug 1"]).casefold(): "DRUG1",
        str(row["Drug 2"]).casefold(): "DRUG2"
    }

    names = sorted(drug_map.keys(), key=len, reverse=True)

    pattern = re.compile(
        "|".join(re.escape(name) for name in names),
        flags=re.IGNORECASE
    )

    return pattern.sub(
        lambda match: drug_map[match.group().casefold()],
        text
    ).lower().strip()


df["correct_template"] = df.apply(make_template, axis=1)


print("\n========== 1. CORRECTED TEMPLATE ANALYSIS ==========")

templates = df["correct_template"].value_counts()

print("Unique templates:", len(templates))

print("\nAll template counts:")

print(templates.to_string())


print("\n========== 2. UNCLASSIFIED PATTERN INSPECTION ==========")

def broad_category(text):

    if re.search(r"risk or severity of", text, re.I):
        return "Risk / Severity"

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

    if re.search(r"\bmay (increase|decrease)\b", text, re.I):
        return "Pharmacological Activity"

    return "Other / Unclassified"


df["correct_category"] = df[
    "Interaction Description"
].apply(broad_category)

print("\nCorrected category distribution:")

print(df["correct_category"].value_counts().to_string())

unclassified = df[
    df["correct_category"] == "Other / Unclassified"
]

print("\nUnclassified records:", len(unclassified))

print("\nDistinct unclassified descriptions:")

print(
    unclassified["correct_template"]
    .value_counts()
    .to_string()
)


print("\n========== 3. REVERSE-ORDER PAIR ANALYSIS ==========")

df["ordered_pair"] = list(zip(
    df["drug1_key"],
    df["drug2_key"]
))

df["unordered_pair"] = df.apply(
    lambda row: tuple(sorted([
        str(row["drug1_key"]),
        str(row["drug2_key"])
    ])),
    axis=1
)

orientation_counts = (
    df.groupby("unordered_pair")["ordered_pair"]
    .nunique()
)

reverse_pairs = orientation_counts[
    orientation_counts > 1
]

print("Pairs appearing in both orientations:", len(reverse_pairs))

print("\nSample reverse-order pairs:")

sample_keys = reverse_pairs.head(20).index

print(
    df[df["unordered_pair"].isin(sample_keys)][
        ["Drug 1", "Drug 2", "Interaction Description"]
    ].to_string(index=False)
)


print("\nAnalysis completed.")
