
import pandas as pd
import re
from pathlib import Path

path = Path("data/processed/integrated_ddi_dataset.csv")
df = pd.read_csv(path)

def make_template(row):
    text = str(row["Interaction Description"])

    drug_map = {
        str(row["Drug 1"]).casefold(): "DRUG1",
        str(row["Drug 2"]).casefold(): "DRUG2"
    }

    names = sorted(drug_map, key=len, reverse=True)

    pattern = re.compile(
        "|".join(re.escape(n) for n in names),
        re.I
    )

    return pattern.sub(
        lambda m: drug_map[m.group().casefold()],
        text
    ).lower().strip()


df["template"] = df.apply(make_template, axis=1)


def classify_family(text):

    if any(x in text for x in [
        "risk or severity",
        "risk of a hypersensitivity"
    ]):
        return "Adverse Effect Risk"

    if "serum concentration" in text:
        return "Serum Concentration"

    if "metabolism" in text:
        return "Drug Metabolism"

    if "therapeutic efficacy" in text:
        return "Therapeutic Efficacy"

    if "protein binding" in text:
        return "Protein Binding"

    if "absorption" in text:
        return "Drug Absorption"

    if "bioavailability" in text:
        return "Bioavailability"

    if "excretion rate" in text:
        return "Drug Excretion"

    if "may increase" in text or "may decrease" in text:
        return "Pharmacological Activity"

    return "Other"


def get_direction(text):

    if re.search(r"\b(increased|increase)\b", text):
        return "Increase"

    if re.search(r"\b(decreased|decrease|reduced)\b", text):
        return "Decrease"

    return "Unspecified"


def get_affected_drug(text):

    if "risk or severity" in text or "risk of a hypersensitivity" in text:
        return "Drug pair"

    if "drug2" in text and (
        "of drug2" in text or
        "activities of drug2" in text or
        "drug2 can be" in text
    ):
        return "Drug 2"

    if "drug1" in text and (
        "of drug1" in text or
        "activities of drug1" in text or
        "drug1 can be" in text
    ):
        return "Drug 1"

    return "Review"


catalog = (
    df.groupby("template")
    .size()
    .reset_index(name="record_count")
)

catalog["interaction_family"] = catalog[
    "template"
].apply(classify_family)

catalog["direction"] = catalog[
    "template"
].apply(get_direction)

catalog["affected_drug"] = catalog[
    "template"
].apply(get_affected_drug)

catalog = catalog.sort_values(
    "record_count",
    ascending=False
)

output = Path("data/processed/interaction_template_catalog.csv")

catalog.to_csv(output, index=False)

print("\n========== INTERACTION CATALOG ==========")

print("Total templates:", len(catalog))

print("\nFamily distribution:")
print(
    catalog.groupby("interaction_family")["record_count"]
    .sum()
    .sort_values(ascending=False)
    .to_string()
)

print("\nDirection distribution:")
print(
    catalog.groupby("direction")["record_count"]
    .sum()
    .to_string()
)

print("\nAffected drug distribution:")
print(
    catalog.groupby("affected_drug")["record_count"]
    .sum()
    .to_string()
)

print("\nSaved catalogue to:", output)
print("\nFirst 20 catalogue entries:")
print(catalog.head(20).to_string(index=False))
