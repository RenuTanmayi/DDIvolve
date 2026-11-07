
import os
import re
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    AllChem,
    Draw,
    DataStructs
)

# ============================================================
# PHASE 2: MOLECULAR FEATURE EXTRACTION
# Project: DDIvolve
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "clean_drugs.csv"
OUTPUT_DIR = BASE_DIR / "data" / "features"

# Output directories
DESCRIPTOR_DIR = OUTPUT_DIR / "descriptors"
FINGERPRINT_DIR = OUTPUT_DIR / "fingerprints"
GRAPH_DIR = OUTPUT_DIR / "molecular_graphs"
STRUCTURE_2D_DIR = OUTPUT_DIR / "structures_2d"
STRUCTURE_3D_DIR = OUTPUT_DIR / "structures_3d"

for folder in [
    DESCRIPTOR_DIR,
    FINGERPRINT_DIR,
    GRAPH_DIR,
    STRUCTURE_2D_DIR,
    STRUCTURE_3D_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# Configuration
FP_RADIUS = 2
FP_BITS = 2048
RANDOM_SEED = 42
MAX_OPTIMIZATION_ITERATIONS = 500

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    """Create a filesystem-safe filename."""

    name = str(name).strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)

    return name[:100]


def get_molecule(smiles):
    """Convert SMILES into a valid RDKit molecule."""

    if pd.isna(smiles) or not str(smiles).strip():
        return None

    mol = Chem.MolFromSmiles(str(smiles))

    return mol


def extract_descriptors(mol):
    """Calculate molecular descriptors."""

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "exact_molecular_weight": Descriptors.ExactMolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "h_bond_donors": rdMolDescriptors.CalcNumHBD(mol),
        "h_bond_acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "ring_count": mol.GetRingInfo().NumRings(),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "atom_count": mol.GetNumAtoms(),
        "bond_count": mol.GetNumBonds(),
        "formal_charge": Chem.GetFormalCharge(mol),
        "qed": __import__(
            "rdkit.Chem.QED", fromlist=["qed"]
        ).qed(mol)
    }


def extract_fingerprint(mol):
    """Generate a 2048-bit Morgan fingerprint."""

    generator = AllChem.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_BITS
    )

    fingerprint = generator.GetFingerprint(mol)

    array = np.zeros((FP_BITS,), dtype=np.uint8)

    DataStructs.ConvertToNumpyArray(
        fingerprint, array
    )

    return array


def extract_graph(mol):
    """Represent a molecule as an atom-bond graph."""

    nodes = []

    for atom in mol.GetAtoms():

        nodes.append({
            "atom_index": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
            "degree": atom.GetDegree(),
            "formal_charge": atom.GetFormalCharge(),
            "hybridization": str(atom.GetHybridization()),
            "aromatic": atom.GetIsAromatic(),
            "chiral_tag": str(atom.GetChiralTag()),
            "total_valence": atom.GetTotalValence(),
            "total_hydrogens": atom.GetTotalNumHs()
        })

    edges = []

    for bond in mol.GetBonds():

        edges.append({
            "source": bond.GetBeginAtomIdx(),
            "target": bond.GetEndAtomIdx(),
            "bond_type": str(bond.GetBondType()),
            "bond_order": bond.GetBondTypeAsDouble(),
            "aromatic": bond.GetIsAromatic(),
            "conjugated": bond.GetIsConjugated(),
            "in_ring": bond.IsInRing(),
            "stereo": str(bond.GetStereo())
        })

    return {
        "number_of_nodes": len(nodes),
        "number_of_edges": len(edges),
        "nodes": nodes,
        "edges": edges
    }


def generate_2d_structure(mol, output_path):
    """Generate and save a 2D molecular structure."""

    mol_2d = Chem.Mol(mol)

    AllChem.Compute2DCoords(mol_2d)

    Draw.MolToFile(
        mol_2d,
        str(output_path),
        size=(600, 450),
        kekulize=True
    )


def generate_3d_structure(mol, drug_name, writer):
    """
    Generate a 3D conformer and optimize it using MMFF.
    Fall back to UFF if MMFF is unavailable.
    """

    mol_3d = Chem.AddHs(Chem.Mol(mol))

    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.useRandomCoords = False

    result = AllChem.EmbedMolecule(
        mol_3d,
        params
    )

    if result != 0:
        raise RuntimeError("3D conformer generation failed")

    optimization_method = None
    optimization_status = None

    if AllChem.MMFFHasAllMoleculeParams(mol_3d):

        optimization_status = AllChem.MMFFOptimizeMolecule(
            mol_3d,
            maxIters=MAX_OPTIMIZATION_ITERATIONS
        )

        optimization_method = "MMFF"

    elif AllChem.UFFHasAllMoleculeParams(mol_3d):

        optimization_status = AllChem.UFFOptimizeMolecule(
            mol_3d,
            maxIters=MAX_OPTIMIZATION_ITERATIONS
        )

        optimization_method = "UFF"

    else:
        raise RuntimeError("Neither MMFF nor UFF parameters available")

    mol_3d.SetProp("_Name", str(drug_name))
    mol_3d.SetProp("optimization_method", optimization_method)
    mol_3d.SetProp("optimization_status", str(optimization_status))

    writer.write(mol_3d)

    return optimization_method, optimization_status


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 65)
    print("       DDIVOLVE - PHASE 2 FEATURE EXTRACTION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    print("\nLoading drug dataset...")

    df = pd.read_csv(INPUT_FILE)

    required_columns = ["drug_name", "smiles"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    print(f"Total drug records: {len(df)}")

    # Use canonical SMILES when available.
    # Fall back to original SMILES if necessary.

    descriptors_list = []
    fingerprints_list = []
    processing_log = []

    success_count = 0
    failure_count = 0

    mmff_count = 0
    uff_count = 0

    # Combined 3D structure file
    sdf_path = STRUCTURE_3D_DIR / "all_drugs_3d.sdf"

    sdf_writer = Chem.SDWriter(str(sdf_path))

    try:

        for index, row in df.iterrows():

            drug_name = str(row["drug_name"])

            # Preserve a unique identifier even for repeated names.
            filename = (
                f"{index + 1:05d}_"
                f"{safe_filename(drug_name)}"
            )

            print(
                f"\n[{index + 1}/{len(df)}] Processing: {drug_name}"
            )

            record = {
                "source_row": index,
                "drug_name": drug_name,
                "status": "failed",
                "error": ""
            }

            try:

                # ------------------------------------------------
                # STEP 1: MOLECULAR VALIDATION
                # ------------------------------------------------

                smiles = None

                for column in [
                    "canonical_smiles",
                    "smiles"
                ]:

                    if column in df.columns:

                        candidate = row[column]

                        if pd.notna(candidate) and str(candidate).strip():

                            mol = get_molecule(candidate)

                            if mol is not None:
                                smiles = str(candidate)
                                break

                if mol is None:
                    raise ValueError("Invalid or missing SMILES")

                canonical_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=True
                )

                record["canonical_smiles"] = canonical_smiles

                # ------------------------------------------------
                # STEP 2: MOLECULAR DESCRIPTORS
                # ------------------------------------------------

                descriptors = extract_descriptors(mol)

                descriptor_record = {
                    "drug_name": drug_name,
                    "source_row": index,
                    "canonical_smiles": canonical_smiles,
                    **descriptors
                }

                descriptors_list.append(descriptor_record)

                # ------------------------------------------------
                # STEP 3: MORGAN FINGERPRINT
                # ------------------------------------------------

                fingerprint = extract_fingerprint(mol)

                fingerprints_list.append({
                    "drug_name": drug_name,
                    "source_row": index,
                    "fingerprint": fingerprint
                })

                # ------------------------------------------------
                # STEP 4: MOLECULAR GRAPH
                # ------------------------------------------------

                graph = extract_graph(mol)

                graph_data = {
                    "drug_name": drug_name,
                    "source_row": index,
                    "canonical_smiles": canonical_smiles,
                    **graph
                }

                graph_path = GRAPH_DIR / f"{filename}.json"

                with open(
                    graph_path,
                    "w",
                    encoding="utf-8"
                ) as file:

                    json.dump(
                        graph_data,
                        file,
                        indent=4
                    )

                # ------------------------------------------------
                # STEP 5: 2D STRUCTURE
                # ------------------------------------------------

                image_path = (
                    STRUCTURE_2D_DIR / f"{filename}.png"
                )

                generate_2d_structure(
                    mol,
                    image_path
                )

                # ------------------------------------------------
                # STEP 6: 3D CONFORMER GENERATION
                # ------------------------------------------------

                method, status = generate_3d_structure(
                    mol,
                    drug_name,
                    sdf_writer
                )

                if method == "MMFF":
                    mmff_count += 1
                else:
                    uff_count += 1

                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                success_count += 1

                record["status"] = "success"
                record["optimization_method"] = method
                record["optimization_status"] = status

                print(
                    f"  SUCCESS | MM descriptors | "
                    f"Fingerprint | Graph | 2D | 3D ({method})"
                )

            except Exception as error:

                failure_count += 1

                record["error"] = str(error)

                print(f"  FAILED: {error}")

            processing_log.append(record)

    finally:
        sdf_writer.close()

    # ============================================================
    # SAVE DESCRIPTORS
    # ============================================================

    print("\nSaving molecular descriptors...")

    descriptor_df = pd.DataFrame(descriptors_list)

    descriptor_path = (
        DESCRIPTOR_DIR / "molecular_descriptors.csv"
    )

    descriptor_df.to_csv(
        descriptor_path,
        index=False
    )

    # ============================================================
    # SAVE MORGAN FINGERPRINTS
    # ============================================================

    print("Saving Morgan fingerprints...")

    fingerprint_matrix = (
        np.stack([
            item["fingerprint"]
            for item in fingerprints_list
        ])
        if fingerprints_list
        else np.empty((0, FP_BITS), dtype=np.uint8)
    )

    fingerprint_columns = [
        f"bit_{i}"
        for i in range(FP_BITS)
    ]

    fingerprint_df = pd.DataFrame(
        fingerprint_matrix,
        columns=fingerprint_columns
    )

    if fingerprints_list:

        fingerprint_metadata = pd.DataFrame([
            {
                "drug_name": item["drug_name"],
                "source_row": item["source_row"]
            }
            for item in fingerprints_list
        ])

        fingerprint_df.insert(
            0,
            "drug_name",
            fingerprint_metadata["drug_name"]
        )

        fingerprint_df.insert(
            1,
            "source_row",
            fingerprint_metadata["source_row"]
        )

    fingerprint_path = (
        FINGERPRINT_DIR / "morgan_fingerprints.csv"
    )

    fingerprint_df.to_csv(
        fingerprint_path,
        index=False
    )

    # Also save compact binary representation for model training.
    np.savez_compressed(
        FINGERPRINT_DIR / "morgan_fingerprints.npz",
        fingerprints=fingerprint_matrix,
        drug_names=np.array([
            item["drug_name"]
            for item in fingerprints_list
        ]),
        source_rows=np.array([
            item["source_row"]
            for item in fingerprints_list
        ])
    )

    # ============================================================
    # SAVE PROCESSING LOG
    # ============================================================

    log_df = pd.DataFrame(processing_log)

    log_path = OUTPUT_DIR / "phase2_processing_log.csv"

    log_df.to_csv(
        log_path,
        index=False
    )

    # ============================================================
    # GENERATE PHASE 2 REPORT
    # ============================================================

    elapsed_time = time.time() - start_time

    report = f"""
============================================================
                 DDIVOLVE - PHASE 2 REPORT
============================================================

INPUT DATASET
-------------
File: {INPUT_FILE}
Total drug records: {len(df)}

PROCESSING SUMMARY
------------------
Successfully processed: {success_count}
Failed records: {failure_count}
Success rate: {(success_count / len(df) * 100) if len(df) else 0:.2f}%

FEATURE EXTRACTION
------------------
Molecular descriptors: {success_count}
Morgan fingerprints: {success_count}
Fingerprint radius: {FP_RADIUS}
Fingerprint size: {FP_BITS} bits
Molecular graphs: {success_count}
2D molecular structures: {success_count}
3D conformers: {success_count}

3D OPTIMIZATION
---------------
MMFF optimized: {mmff_count}
UFF optimized: {uff_count}

OUTPUT FILES
------------
Descriptors:
{descriptor_path}

Fingerprints:
{fingerprint_path}

Compressed fingerprints:
{FINGERPRINT_DIR / "morgan_fingerprints.npz"}

Molecular graphs:
{GRAPH_DIR}

2D structures:
{STRUCTURE_2D_DIR}

3D structures:
{sdf_path}

Processing log:
{log_path}

EXECUTION
---------
Execution time: {elapsed_time:.2f} seconds

STATUS
------
{"PHASE 2 COMPLETED SUCCESSFULLY" if failure_count == 0 else "PHASE 2 COMPLETED WITH SOME FAILED RECORDS"}

Note: A successful 3D optimization status of 0 indicates
that the force-field optimization converged. Status 1
indicates that the maximum iteration count was reached.

============================================================
"""

    report_path = OUTPUT_DIR / "phase2_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print(report)

    print("\nPhase 2 feature extraction finished.")
    print(f"Report saved at: {report_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
