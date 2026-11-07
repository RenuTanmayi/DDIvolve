
import pandas as pd
from pathlib import Path

df = pd.read_csv(
    "data/processed/integrated_ddi_dataset.csv"
)

print("\n========== 1. EXACT DUPLICATES ==========")

print("Exact duplicate rows:", df.duplicated().sum())

print("\n========== 2. DUPLICATE PAIR ANALYSIS ==========")

duplicates = df[
    df.duplicated(subset=["pair_id"], keep=False)
].sort_values("pair_id")

print("Rows belonging to repeated pairs:", len(duplicates))

if not duplicates.empty:
    print("\nSample repeated pairs:")
    print(
        duplicates[
            ["Drug 1", "Drug 2", "Interaction Description", "pair_id"]
        ].head(15).to_string(index=False)
    )

    print("\nRepeated pairs with multiple descriptions:")

    desc_counts = (
        duplicates.groupby("pair_id")["Interaction Description"]
        .nunique()
    )

    print((desc_counts > 1).sum())


print("\n========== 3. REVERSE PAIR ANALYSIS ==========")

# Treat A+B and B+A as the same unordered pair
df["unordered_pair"] = df.apply(
    lambda row: " || ".join(
        sorted([str(row["drug1_key"]), str(row["drug2_key"])])
    ),
    axis=1
)

print(
    "Unique unordered drug pairs:",
    df["unordered_pair"].nunique()
)

print(
    "Rows belonging to repeated unordered pairs:",
    df.duplicated(subset=["unordered_pair"]).sum()
)

print("\nSample repeated unordered pairs:")

reverse = df[
    df.duplicated(subset=["unordered_pair"], keep=False)
]

print(
    reverse[
        ["Drug 1", "Drug 2", "Interaction Description"]
    ].head(20).to_string(index=False)
)


print("\n========== 4. SELF-INTERACTIONS ==========")

self_pairs = df[df["drug1_key"] == df["drug2_key"]]

print("Self-interaction records:", len(self_pairs))


print("\n========== 5. INTERACTION SENTENCE PATTERNS ==========")

descriptions = df["Interaction Description"].dropna()

# Remove drug names to identify recurring sentence templates
patterns = descriptions.copy()

for col in ["Drug 1", "Drug 2"]:
    names = sorted(
        df[col].dropna().astype(str).unique(),
        key=len,
        reverse=True
    )

    for name in names:
        patterns = patterns.str.replace(
            name, "DRUG", regex=False
        )

print("\nMost common sentence templates:")

print(
    patterns.value_counts().head(30).to_string()
)

print("\n========== 6. SAMPLE DESCRIPTIONS ==========")

for description in descriptions.sample(
    min(25, len(descriptions)),
    random_state=42
):
    print("-", description)

print("\nInteraction analysis completed.")
