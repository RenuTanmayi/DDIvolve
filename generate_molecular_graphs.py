
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import rdchem

# ==========================================
# PATHS
# ==========================================

INPUT_FILE = "data/processed/clean_drugs.csv"
OUTPUT_DIR = "data/processed/molecular_features/molecular_graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(INPUT_FILE)

# ==========================================
# FEATURE DEFINITIONS
# ==========================================

def get_atom_features(atom):

    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        int(atom.GetHybridization()),
        atom.GetMass()
    ]


def get_bond_features(bond):

    return [
        int(bond.GetBondType()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo())
    ]


# ==========================================
# GENERATE MOLECULAR GRAPHS
# ==========================================

successful = 0
failed = []

graph_index = []

print("Generating molecular graphs...\n")

for idx, row in df.iterrows():

    drug_name = row["drug_name"]
    smiles = row["canonical_smiles"]

    if pd.isna(smiles):
        failed.append(drug_name)
        continue

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None or mol.GetNumAtoms() == 0:
        failed.append(drug_name)
        continue

    # Extract node features
    node_features = np.array(
        [get_atom_features(atom) for atom in mol.GetAtoms()],
        dtype=np.float32
    )

    # Extract edge features
    edge_list = []
    edge_features = []

    for bond in mol.GetBonds():

        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        features = get_bond_features(bond)

        # Add both directions for an undirected molecular bond
        edge_list.append([start, end])
        edge_features.append(features)

        edge_list.append([end, start])
        edge_features.append(features)

    # Handle molecules without bonds
    if len(edge_list) == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 4), dtype=np.float32)
    else:
        edge_index = np.array(
            edge_list, dtype=np.int64
        ).T

        edge_attr = np.array(
            edge_features, dtype=np.float32
        )

    # Save individual molecular graph
    filename = f"graph_{idx:04d}.npz"

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, filename),
        drug_name=np.array(drug_name),
        smiles=np.array(str(smiles)),
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    graph_index.append({
        "drug_name": drug_name,
        "smiles": str(smiles),
        "graph_file": filename,
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds()
    })

    successful += 1

# ==========================================
# SAVE GRAPH INDEX
# ==========================================

index_df = pd.DataFrame(graph_index)

index_df.to_csv(
    os.path.join(OUTPUT_DIR, "molecular_graph_index.csv"),
    index=False
)

# ==========================================
# RESULTS
# ==========================================

print("\n========== MOLECULAR GRAPH RESULTS ==========")

print("Total drugs:", len(df))
print("Successfully generated:", successful)
print("Failed:", len(failed))

print("\nGraph feature dimensions:")

if successful > 0:

    sample = np.load(
        os.path.join(OUTPUT_DIR, graph_index[0]["graph_file"])
    )

    print("Node features:", sample["node_features"].shape)
    print("Edge index:", sample["edge_index"].shape)
    print("Edge features:", sample["edge_attr"].shape)

print("\nSaved to:", OUTPUT_DIR)

print("\nGraph index saved successfully.")

if failed:
    print("\nDrugs requiring review:")
    print(failed)
