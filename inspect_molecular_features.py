
import pandas as pd
import os

file_path = "data/processed/clean_drugs.csv"

if not os.path.exists(file_path):
    print("ERROR: clean_drugs.csv not found!")
    exit()

df = pd.read_csv(file_path)

print("\n========== MOLECULAR FEATURE INSPECTION ==========")

print("\nDataset shape:", df.shape)

print("\nAvailable columns:")
for col in df.columns:
    print("-", col)

print("\n========== FEATURE STATUS ==========")

features = {
    "SMILES": ["smiles", "canonical_smiles"],
    "Molecular descriptors": [
        "molecular_formula",
        "molecular_weight",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "num_atoms",
        "num_rings"
    ],
    "Morgan fingerprints": [
        "morgan_fingerprint",
        "fingerprint"
    ],
    "Molecular graphs": [
        "molecular_graph",
        "graph"
    ],
    "2D structures": [
        "2d_structure",
        "image_path"
    ],
    "3D conformers": [
        "3d_conformer",
        "conformer_path"
    ]
}

columns_lower = {col.lower(): col for col in df.columns}

for category, possible_columns in features.items():
    found = [columns_lower[c] for c in possible_columns if c in columns_lower]

    if found:
        print(f"\n{category}: FOUND")
        for col in found:
            print("  -", col)
    else:
        print(f"\n{category}: NOT FOUND")

print("\n========== DATA QUALITY ==========")

for col in ["smiles", "canonical_smiles"]:
    if col in df.columns:
        print(f"{col} non-null:", df[col].notna().sum())
        print(f"{col} missing:", df[col].isna().sum())

print("\n========== SAMPLE RECORD ==========")
print(df.head(3).to_string())

print("\nInspection completed.")
