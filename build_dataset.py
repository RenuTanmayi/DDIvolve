
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, QED


# --------------------------------------------------
# 1. FILE PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "interactions.csv"
OUTPUT_FILE = DATA_DIR / "drugs.csv"

# Wait between ordinary requests.
REQUEST_DELAY = 2.0

# Number of attempts for temporary failures.
MAX_ATTEMPTS = 4

# Longer cooldown after HTTP 429.
RATE_LIMIT_BASE_WAIT = 60

RETRY_STATUSES = {429, 500, 502, 503, 504}

session = requests.Session()
session.headers.update({
    "User-Agent": "DDIvolveStudentResearchPrototype/1.0"
})


# --------------------------------------------------
# 2. CREATE A STANDARD DRUG RECORD
# --------------------------------------------------

def empty_record(name):
    return {
        "drug_name": name,
        "smiles": "",
        "canonical_smiles": "",
        "molecular_formula": "",
        "pubchem_cid": "",
        "lookup_status": "pending",
        "valid_smiles": False,
        "molecular_weight": None,
        "logp": None,
        "h_bond_donors": None,
        "h_bond_acceptors": None,
        "tpsa": None,
        "rotatable_bonds": None,
        "qed": None,
        "atom_count": None,
        "bond_count": None,
        "lookup_detail": ""
    }


# --------------------------------------------------
# 3. CALCULATE MOLECULAR FEATURES USING RDKIT
# --------------------------------------------------

def calculate_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "valid_smiles": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 3),
        "logp": round(Descriptors.MolLogP(mol), 3),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "qed": round(QED.qed(mol), 3),
        "atom_count": mol.GetNumAtoms(),
        "bond_count": mol.GetNumBonds()
    }


# --------------------------------------------------
# 4. LOOK UP A DRUG IN PUBCHEM
# --------------------------------------------------

def get_pubchem_data(drug_name):
    """
    Returns:
        (data, status, detail)

    Possible statuses:
        success
        not_found
        retry_later
        lookup_error
    """

    encoded_name = quote(drug_name, safe="")

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded_name}/property/"
        "SMILES,ConnectivitySMILES,MolecularFormula/JSON"
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            response = session.get(url, timeout=25)

            # Successful response
            if response.status_code == 200:
                payload = response.json()
                properties = payload["PropertyTable"]["Properties"][0]

                smiles = (
                    properties.get("ConnectivitySMILES")
                    or properties.get("SMILES")
                )

                if not smiles:
                    return (
                        None,
                        "lookup_error",
                        "PubChem response did not contain a SMILES string"
                    )

                return {
                    "smiles": smiles,
                    "molecular_formula":
                        properties.get("MolecularFormula", ""),
                    "pubchem_cid": properties.get("CID", "")
                }, "success", "HTTP 200"

            # Read any structured error information.
            try:
                error_payload = response.json()
                fault = error_payload.get("Fault", {})
                error_code = str(fault.get("Code", ""))
                error_message = str(fault.get("Message", ""))
            except ValueError:
                error_code = ""
                error_message = response.text[:200]

            detail = (
                f"HTTP {response.status_code}: "
                f"{error_code} {error_message}"
            ).strip()

            # Only treat an explicit compound-not-found response
            # as not_found.
            if (
                response.status_code == 404
                or "NotFound" in error_code
            ):
                return None, "not_found", detail

            # Temporary server errors and rate limiting.
            if response.status_code in RETRY_STATUSES:

                if response.status_code == 429:
                    # Respect Retry-After if PubChem supplies it.
                    retry_after = response.headers.get("Retry-After")

                    try:
                        wait = max(
                            RATE_LIMIT_BASE_WAIT * attempt,
                            int(retry_after) if retry_after else 0
                        )
                    except ValueError:
                        wait = RATE_LIMIT_BASE_WAIT * attempt

                    print(
                        f"  HTTP 429 rate limit. "
                        f"Waiting {wait} seconds..."
                    )

                else:
                    wait = min(2 ** attempt, 30)
                    print(
                        f"  Temporary HTTP error. "
                        f"Waiting {wait} seconds..."
                    )

                if attempt < MAX_ATTEMPTS:
                    time.sleep(wait)
                    continue

                return None, "retry_later", detail

            # Do not incorrectly classify other errors as not_found.
            return None, "lookup_error", detail

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            IndexError,
            TypeError
        ) as error:

            detail = f"{type(error).__name__}: {error}"

            if attempt < MAX_ATTEMPTS:
                wait = min(2 ** attempt, 30)
                print(
                    f"  Request or response error. "
                    f"Waiting {wait} seconds..."
                )
                time.sleep(wait)
            else:
                return None, "retry_later", detail

    return None, "retry_later", "Maximum attempts exhausted"


# --------------------------------------------------
# 5. SAVE PROGRESS SAFELY
# --------------------------------------------------
# --------------------------------------------------
# 5. SAVE PROGRESS SAFELY
# --------------------------------------------------

def save_records(records):
    output = pd.DataFrame(list(records.values()))

    output = output.sort_values(
        "drug_name",
        key=lambda col: col.str.casefold()
    )

    temp_file = OUTPUT_FILE.with_name("drugs_temp.csv")

    # Write all records to a temporary CSV first.
    output.to_csv(temp_file, index=False)

    # Retry replacement in case Windows temporarily locks the file.
    for attempt in range(5):
        try:
            temp_file.replace(OUTPUT_FILE)
            return

        except PermissionError:
            if attempt < 4:
                wait = 3
                print(
                    f"\nFile is currently locked. "
                    f"Retrying in {wait} seconds..."
                )
                time.sleep(wait)

            else:
                print(
                    "\nERROR: Unable to update drugs.csv.\n"
                    "1. Close Excel or any application using the file.\n"
                    "2. Check whether another Python process is running.\n"
                    f"3. Your temporary data is preserved at: {temp_file}\n"
                    "Please resolve the file lock and try again."
                )
                raise

# --------------------------------------------------
# 6. MAIN DATASET BUILDING PROCESS
# --------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    interactions = pd.read_csv(INPUT_FILE)

    if interactions.shape[1] < 3:
        print(
            "ERROR: interactions.csv must contain at least "
            "three columns: drug A, drug B, description."
        )
        return

    # Collect unique drug names from the first two columns.
    names = pd.concat(
        [
            interactions.iloc[:, 0],
            interactions.iloc[:, 1]
        ],
        ignore_index=True
    )

    names = names.dropna().astype(str).str.strip()

    drug_names = sorted(
        {name for name in names if name},
        key=str.casefold
    )

    # Load existing records and preserve successful lookups.
    records = {}

    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE).fillna("")

        if "drug_name" in existing.columns:
            for row in existing.to_dict("records"):
                name = str(row.get("drug_name", "")).strip()

                if name:
                    records[name] = row

    # Add any new drugs that are absent from the existing CSV.
    for name in drug_names:
        if name not in records:
            records[name] = empty_record(name)

    # Ensure older records have the diagnostic column.
    for record in records.values():
        record.setdefault("lookup_detail", "")

    # Only look up records that do not already have a usable structure.
    pending = [
        name for name in drug_names
        if not (
            str(records[name].get("lookup_status", "")) == "success"
            and str(records[name].get("smiles", "")).strip()
        )
    ]

    successful_count = len(drug_names) - len(pending)

    print(f"Unique drugs in interactions: {len(drug_names)}")
    print(f"Existing successful records preserved: {successful_count}")
    print(f"Records to look up: {len(pending)}")
    print("\nStarting PubChem lookup...\n")

    # If there are no pending records, still report the final dataset.
    for index, name in enumerate(pending, start=1):

        print(f"[{index}/{len(pending)}] {name}")

        data, status, detail = get_pubchem_data(name)

        row = records[name].copy()
        row["drug_name"] = name
        row["lookup_status"] = status
        row["lookup_detail"] = detail

        if status == "success" and data:

            row["smiles"] = data["smiles"]
            row["molecular_formula"] = data["molecular_formula"]
            row["pubchem_cid"] = data["pubchem_cid"]

            features = calculate_features(data["smiles"])

            if features is None:
                row["lookup_status"] = "invalid_smiles"
                row["valid_smiles"] = False
                row["lookup_detail"] = (
                    "PubChem returned a structure that RDKit "
                    "could not parse"
                )

            else:
                row.update(features)
                row["lookup_status"] = "success"
                row["lookup_detail"] = (
                    "PubChem lookup and RDKit parsing succeeded"
                )

        print(f"  Result: {row['lookup_status']}")

        records[name] = row

        # Save each result so interrupted runs retain their progress.
        save_records(records)

        # Apply a delay between drugs, including after failures.
        time.sleep(REQUEST_DELAY)

    # Final report
    final_df = pd.read_csv(OUTPUT_FILE)

    print("\nDataset processing finished.")
    print(f"Saved file: {OUTPUT_FILE}")
    print("\nLookup status counts:")
    print(
        final_df["lookup_status"].value_counts(dropna=False).to_string()
    )

    if "lookup_detail" in final_df.columns:
        print("\nUnsuccessful lookup examples:")
        unsuccessful = final_df[
            final_df["lookup_status"] != "success"
        ]

        if unsuccessful.empty:
            print("All drug structures were retrieved successfully.")
        else:
            print(
                unsuccessful[
                    ["drug_name", "lookup_status", "lookup_detail"]
                ].head(15).to_string(index=False)
            )


if __name__ == "__main__":
    main()
