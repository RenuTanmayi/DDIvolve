
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# DDIvolve | Model loading and prediction
# ============================================================

@st.cache_resource(show_spinner="Loading DDIvolve XGBoost model...")
def load_ddi_predictor():
    from src.inference.predict import DDIvolvePredictor
    return DDIvolvePredictor()


def predict_ddi_family(drug_a: str, drug_b: str):
    record_a = get_drug_record(drug_a)
    record_b = get_drug_record(drug_b)

    if record_a is None or record_b is None:
        raise ValueError("Molecular records unavailable for one or both drugs.")

    def extract_smiles(record):
        for column in ("canonical_smiles", "smiles"):
            value = record.get(column)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
        return None

    smiles_a = extract_smiles(record_a)
    smiles_b = extract_smiles(record_b)

    if not smiles_a or not smiles_b:
        raise ValueError("Valid SMILES unavailable for one or both drugs.")

    return load_ddi_predictor().predict(smiles_a, smiles_b)


# ============================================================
# DDIvolve | Research dashboard
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

INTERACTIONS_FILE = DATA_DIR / "interactions.csv"
DRUGS_FILE = DATA_DIR / "drugs.csv"

RAW_DDIS_FILE = DATA_DIR / "ddis.csv"
RAW_SMILES_FILE = DATA_DIR / "drug_smiles.csv"

st.set_page_config(
    page_title="DDIvolve | Interaction Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Design system
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ddi-ink: #16263d;
        --ddi-muted: #64748b;
        --ddi-teal: #0f9f91;
        --ddi-blue: #4169e1;
        --ddi-line: rgba(100, 116, 139, 0.20);
        --ddi-panel: rgba(148, 163, 184, 0.075);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, rgba(15,159,145,0.09), transparent 30%),
            radial-gradient(ellipse at 95% 4%, rgba(65,105,225,0.08), transparent 28%);
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em;
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--ddi-line);
    }

    [data-testid="stMetric"] {
        background: var(--ddi-panel);
        border: 1px solid var(--ddi-line);
        padding: 1rem 1.1rem;
        border-radius: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--ddi-muted);
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.04em;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
    }

    .hero {
        padding: 1.8rem 1.9rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(118deg, #10253f 0%, #153b56 54%, #087f80 100%);
        margin-bottom: 1.3rem;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        right: -65px;
        top: -95px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.22);
        box-shadow: 0 0 0 25px rgba(255,255,255,0.035),
                    0 0 0 55px rgba(255,255,255,0.025);
    }

    .hero-kicker {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #a7f3e8;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.08;
        letter-spacing: -0.055em;
        font-weight: 800;
        margin: 0;
    }

    .hero-copy {
        color: rgba(255,255,255,0.82);
        max-width: 760px;
        margin-top: 0.8rem;
        font-size: 1.02rem;
        line-height: 1.65;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.35rem 0.65rem;
        border: 1px solid rgba(255,255,255,0.23);
        border-radius: 999px;
        font-size: 0.78rem;
        color: #e0fffa;
        background: rgba(255,255,255,0.07);
    }

    .section-eyebrow {
        color: var(--ddi-teal);
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .subtle {
        color: var(--ddi-muted);
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        background: rgba(15,159,145,0.12);
        color: #087f72;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .notice {
        border: 1px solid var(--ddi-line);
        background: var(--ddi-panel);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        line-height: 1.55;
    }

    div.stButton > button[kind="primary"] {
        border-radius: 11px;
        font-weight: 700;
        border: 0;
        background: linear-gradient(100deg, #0f9f91, #167d91);
    }

    div.stButton > button {
        border-radius: 10px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--ddi-line);
        border-radius: 12px;
        overflow: hidden;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Safe data utilities
# ============================================================

def read_csv_safely(path: Path) -> tuple[pd.DataFrame, str | None]:
    """Load a CSV with predictable errors instead of crashing the app."""
    if not path.exists():
        return pd.DataFrame(), f"File not found: {path.name}"

    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False), None
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin-1", low_memory=False), None
        except Exception as exc:
            return pd.DataFrame(), f"Could not read {path.name}: {exc}"
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), f"{path.name} is empty."
    except Exception as exc:
        return pd.DataFrame(), f"Could not read {path.name}: {exc}"


def clean_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)


@st.cache_data(show_spinner="Loading research data…")
def load_interaction_data() -> tuple[pd.DataFrame, str | None, str]:
    """
    Supports either app-ready interactions.csv or the existing ddis.csv.
    The raw TWOSIDES file contains d1, d2, type, and Neg samples; `type`
    is an encoded label, not a human-readable clinical description.
    """
    path = first_existing(INTERACTIONS_FILE, RAW_DDIS_FILE)
    if path is None:
        return pd.DataFrame(), (
            "No interaction file found. Add data/ddis.csv or "
            "data/interactions.csv."
        ), "missing"

    df, error = read_csv_safely(path)
    if error:
        return pd.DataFrame(), error, path.name

    columns_lower = {str(col).strip().casefold(): col for col in df.columns}

    # App-ready, descriptive schema.
    drug_a_col = next(
        (columns_lower[key] for key in ("drug_a", "drug1", "drug_1", "d1")
         if key in columns_lower),
        None,
    )
    drug_b_col = next(
        (columns_lower[key] for key in ("drug_b", "drug2", "drug_2", "d2")
         if key in columns_lower),
        None,
    )
    description_col = next(
        (columns_lower[key] for key in
         ("description", "interaction", "interaction_description", "type")
         if key in columns_lower),
        None,
    )

    if not drug_a_col or not drug_b_col:
        if len(df.columns) < 3:
            return pd.DataFrame(), (
                f"{path.name} needs at least three columns or recognizable "
                "drug-pair columns."
            ), path.name
        # Compatibility fallback for an app-ready CSV with arbitrary headers.
        if path.name == INTERACTIONS_FILE.name:
            drug_a_col, drug_b_col, description_col = (
                df.columns[0], df.columns[1], df.columns[2]
            )
        else:
            return pd.DataFrame(), (
                f"Could not identify drug-pair columns in {path.name}. "
                f"Found: {', '.join(map(str, df.columns))}"
            ), path.name

    result = pd.DataFrame({
        "Drug_A": clean_text_series(df[drug_a_col]),
        "Drug_B": clean_text_series(df[drug_b_col]),
    })

    if description_col:
        result["Evidence_Label"] = clean_text_series(df[description_col])
    else:
        result["Evidence_Label"] = "Label not supplied"

    if "Neg samples" in df.columns:
        result["Negative_Sample_ID"] = clean_text_series(df["Neg samples"])

    result = result[
        (result["Drug_A"] != "")
        & (result["Drug_B"] != "")
        & (result["Drug_A"].str.casefold() != "nan")
        & (result["Drug_B"].str.casefold() != "nan")
    ].copy()

    # Do not remove repeated drug pairs: multiple rows can represent
    # distinct interaction types/labels in a multi-label dataset.
    return result.reset_index(drop=True), None, path.name


@st.cache_data(show_spinner="Loading molecular records…")
def load_drug_data() -> tuple[pd.DataFrame, str | None, str]:
    """
    Prefer friend's enriched drugs.csv; otherwise adapt drug_smiles.csv
    so the molecular explorer can show IDs and SMILES until enrichment arrives.
    """
    path = first_existing(DRUGS_FILE, RAW_SMILES_FILE)
    if path is None:
        return pd.DataFrame(), (
            "No molecular file found. Add data/drug_smiles.csv or data/drugs.csv."
        ), "missing"

    df, error = read_csv_safely(path)
    if error:
        return pd.DataFrame(), error, path.name

    df.columns = [str(col).strip() for col in df.columns]

    if "drug_name" not in df.columns and "drug_id" in df.columns:
        df["drug_name"] = df["drug_id"].astype(str)

    if "drug_name" not in df.columns:
        return pd.DataFrame(), (
            f"{path.name} must contain drug_name or drug_id."
        ), path.name

    df["drug_name"] = clean_text_series(df["drug_name"])
    df = df[df["drug_name"] != ""].copy()
    df = df.drop_duplicates(subset=["drug_name"], keep="first")
    return df.reset_index(drop=True), None, path.name


interactions, interactions_error, interactions_source = load_interaction_data()
drugs, drugs_error, drugs_source = load_drug_data()


# ============================================================
# Data helpers
# ============================================================

def safe_text(value: Any, fallback: str = "Not available") -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<na>", "nat"}:
        return fallback
    return text


def safe_number(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value)
        if pd.isna(number):
            return "—"
        return f"{number:,.{decimals}f}"
    except (TypeError, ValueError, OverflowError):
        return "—"


def get_drug_names() -> list[str]:
    names: set[str] = set()
    if not interactions.empty:
        names.update(interactions["Drug_A"].tolist())
        names.update(interactions["Drug_B"].tolist())
    if not drugs.empty:
        names.update(drugs["drug_name"].tolist())
    return sorted(
        {name.strip() for name in names if name.strip()},
        key=str.casefold,
    )


def get_drug_record(name: str) -> pd.Series | None:
    if drugs.empty:
        return None
    match = drugs[
        drugs["drug_name"].astype(str).str.casefold() == name.casefold()
    ]
    return None if match.empty else match.iloc[0]


def find_pair_records(drug_a: str, drug_b: str) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame()
    a = interactions["Drug_A"].astype(str).str.casefold()
    b = interactions["Drug_B"].astype(str).str.casefold()
    first, second = drug_a.casefold(), drug_b.casefold()
    return interactions[
        ((a == first) & (b == second))
        | ((a == second) & (b == first))
    ].copy()


def display_dataset_status() -> None:
    st.markdown("#### Data sources")
    st.caption(f"Interaction source: `{interactions_source}`")
    st.caption(f"Molecular source: `{drugs_source}`")
    if interactions_error:
        st.warning(interactions_error)
    if drugs_error:
        st.info(drugs_error)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🧬 DDIvolve")
    st.caption("Interaction intelligence · Research workspace")
    st.divider()

    st.markdown("**Workspace status**")
    st.metric("Interaction rows", f"{len(interactions):,}")
    st.metric("Drug records", f"{len(drugs):,}")

    if not interactions.empty:
        pair_count = interactions[["Drug_A", "Drug_B"]].drop_duplicates().shape[0]
        st.metric("Unique drug pairs", f"{pair_count:,}")

    st.divider()
    st.markdown("**Current modules**")
    st.markdown(
        "- Interaction evidence search\n"
        "- Molecular descriptor viewer\n"
        "- Dataset exploration\n"
        "- Intervention research roadmap"
    )
    st.divider()
    st.caption(
        "Research prototype only. It is not a medical device and must not "
        "be used to start, stop, or change medication."
    )


# ============================================================
# Hero header
# ============================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Drug safety research platform</div>
      <div class="hero-title">DDIvolve</div>
      <div class="hero-copy">
        An intelligent framework for exploring drug–drug interaction evidence,
        molecular features, and future risk-mitigation research.
      </div>
      <span class="hero-tag">Evidence-aware · Molecularly informed · Research-first</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Overview metrics
# ============================================================

unique_drugs = len(get_drug_names())
unique_pairs = (
    interactions[["Drug_A", "Drug_B"]].drop_duplicates().shape[0]
    if not interactions.empty else 0
)
label_count = (
    interactions["Evidence_Label"].replace("", pd.NA).nunique(dropna=True)
    if not interactions.empty else 0
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Drug entities", f"{unique_drugs:,}", help="Unique names/IDs available across loaded files.")
m2.metric("Interaction records", f"{len(interactions):,}", help="Rows in the currently loaded interaction file.")
m3.metric("Unique drug pairs", f"{unique_pairs:,}", help="Unique unordered pairs represented in the file.")
m4.metric("Distinct labels", f"{label_count:,}", help="Distinct encoded labels/descriptions; not necessarily clinical severity categories.")

st.markdown("")


# ============================================================
# Main application navigation
# ============================================================

tab_overview, tab_checker, tab_molecular, tab_data, tab_explanation, tab_reevaluation, tab_research = st.tabs([
    "Overview",
    "Interaction checker",
    "Molecular explorer",
    "Data explorer",
    "AI Explanation",
    "Re-evaluation",
    "Intervention research",
])


# ============================================================
# Overview
# ============================================================

with tab_overview:
    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        st.markdown('<div class="section-eyebrow">Platform overview</div>', unsafe_allow_html=True)
        st.subheader("From evidence to research insight")
        st.markdown(
            """
            DDIvolve is being developed as a modular research framework.
            The current interface can inspect the supplied interaction records
            and molecular metadata; model-based predictions will be added after
            the training labels and evaluation split are validated.
            """
        )

        with st.container(border=True):
            st.markdown("**Current workflow**")
            st.markdown(
                "1. **Find evidence** — search the loaded dataset for a drug pair.\n"
                "2. **Inspect chemistry** — review available SMILES and descriptors.\n"
                "3. **Evaluate the model** — add validated predictions and performance metrics.\n"
                "4. **Review mitigation ideas** — organize research questions for expert assessment."
            )

    with right:
        with st.container(border=True):
            st.markdown("**Data readiness**")
            if interactions.empty:
                st.warning("Interaction data is not ready.")
            else:
                st.markdown('<span class="status-pill">Interaction file loaded</span>', unsafe_allow_html=True)
                st.write(f"{len(interactions):,} interaction rows available.")
            if drugs.empty:
                st.info("Molecular records are not available yet.")
            else:
                st.markdown('<span class="status-pill">Molecular file loaded</span>', unsafe_allow_html=True)
                st.write(f"{len(drugs):,} drug records available.")
            st.caption("A loaded file does not guarantee complete clinical coverage or validated predictions.")

    st.markdown("### Quick access")
    q1, q2, q3 = st.columns(3)
    with q1:
        with st.container(border=True):
            st.markdown("**⇄ Interaction checker**")
            st.caption("Search whether a pair appears in the supplied dataset.")
    with q2:
        with st.container(border=True):
            st.markdown("**⚛ Molecular explorer**")
            st.caption("Inspect SMILES and available chemical descriptors.")
    with q3:
        with st.container(border=True):
            st.markdown("**▤ Data explorer**")
            st.caption("Filter, inspect, and export research records.")


# ============================================================
# Interaction checker

# ============================================================


# ============================================================
# Interaction checker
# ============================================================

with tab_checker:
    st.markdown(
        '<div class="section-eyebrow">AI-powered interaction analysis</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Drug interaction checker")
    st.markdown(
        '<div class="subtle">Combines research dataset evidence, '
        'XGBoost interaction-family prediction, and DDInter clinical '
        'severity evidence.</div>',
        unsafe_allow_html=True,
    )

    names = get_drug_names()

    if len(names) < 2:
        st.warning("At least two drug entries are needed.")
        display_dataset_status()

    else:
        c1, c2 = st.columns(2)

        with c1:
            drug_a = st.selectbox(
                "First drug", names, key="checker_drug_a"
            )

        with c2:
            candidates = [
                n for n in names
                if n.casefold() != drug_a.casefold()
            ]
            drug_b = st.selectbox(
                "Second drug",
                candidates or names,
                key="checker_drug_b",
            )

        run_check = st.button(
            "Analyze drug combination",
            type="primary",
            use_container_width=True,
        )

        if run_check:
            if drug_a.casefold() == drug_b.casefold():
                st.warning("Select two different drugs.")

            else:
                # 1. Dataset evidence
                st.markdown("---")
                st.markdown("### 1. Research dataset evidence")

                matches = find_pair_records(drug_a, drug_b)

                if not matches.empty:
                    st.success(
                        f"Found {len(matches):,} matching dataset record(s)."
                    )

                    view_cols = [
                        "Drug_A", "Drug_B", "Evidence_Label"
                    ]

                    if "Negative_Sample_ID" in matches.columns:
                        view_cols.append("Negative_Sample_ID")

                    st.dataframe(
                        matches[view_cols].reset_index(drop=True),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "No matching pair record was found in the dataset."
                    )
                    st.caption(
                        "No record found does not establish that the "
                        "combination is safe."
                    )

                # 2. XGBoost prediction
                st.markdown("---")
                st.markdown("### 2. AI interaction-family prediction")

                try:
                    with st.spinner("Running XGBoost prediction..."):
                        prediction = predict_ddi_family(drug_a, drug_b)

                    st.success("XGBoost prediction completed.")

                    p1, p2 = st.columns(2)

                    p1.metric(
                        "Predicted interaction family",
                        prediction["interaction_family"],
                    )

                    p2.metric(
                        "Model confidence",
                        f'{prediction["confidence"]:.2%}',
                    )

                    probability_df = pd.DataFrame(
                        [
                            {
                                "Interaction family": name,
                                "Probability": probability * 100,
                            }
                            for name, probability
                            in prediction["probabilities"].items()
                        ]
                    ).sort_values(
                        "Probability", ascending=False
                    )

                    st.markdown("#### Nine-class prediction distribution")

                    st.bar_chart(
                        probability_df.set_index("Interaction family")
                    )

                    st.dataframe(
                        probability_df.style.format(
                            {"Probability": "{:.2f}%"}
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.caption(
                        "Model confidence is not a clinical severity "
                        "score or a probability of patient harm."
                    )

                except Exception as exc:
                    st.error(f"XGBoost prediction failed: {exc}")

                # 3. DDInter severity evidence
                st.markdown("---")
                st.markdown("### 3. Clinical severity evidence")

                try:
                    from severity_module.src.assess_severity import (
                        assess_severity,
                    )

                    severity = assess_severity(drug_a, drug_b)

                    if severity.get("matched"):
                        st.metric(
                            "DDInter source-reported severity",
                            severity.get("severity", "Unknown"),
                        )

                        st.write(
                            "**Evidence source:**",
                            severity.get("evidence_source", "Not available"),
                        )

                        st.write(
                            "**Evidence quality:**",
                            severity.get("evidence_quality", "Not available"),
                        )

                        st.write(
                            "**Assessment status:**",
                            severity.get("assessment_status", "Not available"),
                        )

                    else:
                        st.warning(
                            "No matching DDInter severity evidence found."
                        )

                    st.info(
                        "Severity is retrieved from DDInter source records. "
                        "It is not an independently validated clinical "
                        "severity prediction."
                    )

                except Exception as exc:
                    st.warning(
                        f"Severity evidence unavailable: {exc}"
                    )

    st.markdown("---")
    st.caption(
        "DDIvolve is a research prototype. Do not use its outputs "
        "to start, stop, or change medication."
    )




# ============================================================
# Molecular explorer
# ============================================================

with tab_molecular:
    import io
    import pandas as pd
    import streamlit as st

    st.markdown(
        '<div class="section-eyebrow">Chemical intelligence</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Molecular explorer")
    st.markdown(
        '<div class="subtle">Explore molecular identity, chemical descriptors, '
        '2D and 3D structures, molecular graphs, and Morgan fingerprints.</div>',
        unsafe_allow_html=True,
    )

    if drugs.empty:
        st.warning("Molecular dataset is unavailable.")
        display_dataset_status()

    else:
        drug_names = sorted(
            drugs["drug_name"].dropna().astype(str).unique(),
            key=str.casefold,
        )

        selected = st.selectbox(
            "Choose a drug",
            drug_names,
            key="molecular_selected_drug",
        )

        record = get_drug_record(selected)

        if record is not None:
            smiles = ""

            for column in (
                "canonical_smiles",
                "isomeric_smiles",
                "smiles",
            ):
                if column in record.index:
                    value = record.get(column)
                    if pd.notna(value) and str(value).strip():
                        smiles = str(value).strip()
                        break

            if not smiles:
                st.warning(
                    "No molecular structure is available for this drug. "
                    "Its identity may still be present in the dataset."
                )
                st.stop()

            try:
                from rdkit import Chem
                from rdkit.Chem import (
                    AllChem,
                    Descriptors,
                    Draw,
                    Lipinski,
                    rdMolDescriptors,
                )
                from rdkit import DataStructs
                import matplotlib.pyplot as plt
                import networkx as nx

                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    st.error("RDKit could not parse this SMILES structure.")
                    st.stop()

            except ImportError as exc:
                st.error(f"Required molecular visualization package missing: {exc}")
                st.stop()

            # ------------------------------------------------
            # Compound identity
            # ------------------------------------------------

            with st.container(border=True):
                identity_cols = st.columns(4)

                identity_cols[0].metric(
                    "Drug",
                    safe_text(record.get("drug_name")),
                )
                identity_cols[1].metric(
                    "PubChem CID",
                    safe_text(record.get("pubchem_cid"), "N/A"),
                )
                identity_cols[2].metric(
                    "Molecular formula",
                    rdMolDescriptors.CalcMolFormula(mol),
                )
                identity_cols[3].metric(
                    "Atoms",
                    mol.GetNumAtoms(),
                )

            st.code(smiles, language=None)

            # ------------------------------------------------
            # Calculate molecular properties
            # ------------------------------------------------

            # ------------------------------------------------
            # Calculate molecular properties
            # ------------------------------------------------

            from rdkit.Chem import QED

            descriptors = {
                "Molecular weight": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "H-bond donors": Lipinski.NumHDonors(mol),
                "H-bond acceptors": Lipinski.NumHAcceptors(mol),
                "TPSA": rdMolDescriptors.CalcTPSA(mol),
                "Rotatable bonds": Lipinski.NumRotatableBonds(mol),
                "QED": float(QED.qed(mol)),
                "Atom count": mol.GetNumAtoms(),
                "Bond count": mol.GetNumBonds(),
            }



            # ------------------------------------------------
            # Molecular explorer tabs
            # ------------------------------------------------

            tab_profile, tab_2d, tab_3d, tab_graph, tab_fp = st.tabs(
                [
                    "Chemical profile",
                    "2D structure",
                    "3D structure",
                    "Molecular graph",
                    "Morgan fingerprint",
                ]
            )

            # ------------------------------------------------
            # 1. Chemical profile
            # ------------------------------------------------

            with tab_profile:
                st.markdown("### Molecular descriptors")

                descriptor_df = pd.DataFrame(
                    [
                        {
                            "Descriptor": name,
                            "Value": round(float(value), 4),
                        }
                        for name, value in descriptors.items()
                    ]
                )

                metric_cols = st.columns(3)

                for i, (name, value) in enumerate(descriptors.items()):
                    metric_cols[i % 3].metric(
                        name,
                        f"{value:.3f}" if isinstance(value, float) else value,
                    )

                st.markdown("#### Descriptor comparison")

                st.bar_chart(
                    descriptor_df.set_index("Descriptor")
                )

                st.dataframe(
                    descriptor_df,
                    hide_index=True,
                    use_container_width=True,
                )

                st.caption(
                    "Descriptors are calculated from the selected molecular "
                    "structure using RDKit. They do not independently establish "
                    "clinical safety or interaction risk."
                )

            # ------------------------------------------------
            # 2. 2D molecular structure
            # ------------------------------------------------

            with tab_2d:
                st.markdown("### 2D molecular structure")

                mol_2d = Chem.Mol(mol)
                from rdkit.Chem import rdDepictor

                rdDepictor.Compute2DCoords(mol_2d)

                image = Draw.MolToImage(
                    mol_2d,
                    size=(850, 550),
                    kekulize=True,
                )

                st.image(
                    image,
                    caption=f"2D chemical structure of {selected}",
                    use_container_width=True,
                )

                st.markdown("#### Structure information")

                structure_cols = st.columns(3)
                structure_cols[0].metric(
                    "Atoms",
                    mol.GetNumAtoms(),
                )
                structure_cols[1].metric(
                    "Bonds",
                    mol.GetNumBonds(),
                )
                structure_cols[2].metric(
                    "Ring count",
                    rdMolDescriptors.CalcNumRings(mol),
                )

                st.caption(
                    "2D coordinates are generated from the molecular "
                    "connectivity for visualization."
                )

            # ------------------------------------------------
            # 3. 3D molecular structure
            # ------------------------------------------------

            with tab_3d:
                st.markdown("### Interactive 3D molecular structure")

                st.write(
                    "Generate a 3D conformer using RDKit ETKDG and "
                    "optimize its geometry using MMFF when available."
                )

                cache_key = (
                    "ddvolve_3d_"
                    + str(selected).replace(" ", "_").replace("/", "_")
                )

                if st.button(
                    "Generate 3D structure",
                    key="generate_3d_" + str(selected),
                    type="primary",
                ):
                    with st.spinner("Generating and optimizing 3D conformer..."):
                        mol_3d = Chem.AddHs(Chem.Mol(mol))

                        params = AllChem.ETKDGv3()
                        params.randomSeed = 42

                        conformer_result = AllChem.EmbedMolecule(
                            mol_3d,
                            params,
                        )

                        if conformer_result != 0:
                            st.session_state[cache_key] = None
                            st.error(
                                "RDKit could not generate a 3D conformer "
                                "for this structure."
                            )
                        else:
                            optimization_method = "Not optimized"

                            if AllChem.MMFFHasAllMoleculeParams(mol_3d):
                                AllChem.MMFFOptimizeMolecule(mol_3d)
                                optimization_method = "MMFF"
                            elif AllChem.UFFHasAllMoleculeParams(mol_3d):
                                AllChem.UFFOptimizeMolecule(mol_3d)
                                optimization_method = "UFF"

                            mol_block = Chem.MolToMolBlock(mol_3d)

                            st.session_state[cache_key] = {
                                "mol_block": mol_block,
                                "optimization": optimization_method,
                                "atom_count": mol_3d.GetNumAtoms(),
                            }

                result_3d = st.session_state.get(cache_key)

                if result_3d:
                    st.success("3D conformer generated successfully.")

                    st.write(
                        f"**Geometry optimization:** "
                        f"{result_3d['optimization']}"
                    )

                    st.write(
                        f"**Atoms including hydrogens:** "
                        f"{result_3d['atom_count']}"
                    )

                    try:
                        import py3Dmol
                        from streamlit.components.v1 import html

                        viewer = py3Dmol.view(
                            width=850,
                            height=550,
                        )

                        viewer.addModel(
                            result_3d["mol_block"],
                            "mol",
                        )

                        viewer.setStyle(
                            {"stick": {"radius": 0.16},
                             "sphere": {"scale": 0.22}},
                        )

                        viewer.setBackgroundColor("white")
                        viewer.zoomTo()

                        html(
                            viewer._make_html(),
                            height=570,
                            scrolling=False,
                        )

                    except ImportError:
                        st.info(
                            "Interactive 3D viewer is not installed. "
                            "Showing the generated 3D coordinates as a "
                            "static visualization instead."
                        )

                        mol_view = Chem.MolFromMolBlock(
                            result_3d["mol_block"],
                            removeHs=False,
                        )

                        conformer = mol_view.GetConformer()

                        coordinates = [
                            conformer.GetAtomPosition(i)
                            for i in range(mol_view.GetNumAtoms())
                        ]

                        fig = plt.figure(figsize=(9, 7))
                        ax = fig.add_subplot(111, projection="3d")

                        for bond in mol_view.GetBonds():
                            a = coordinates[bond.GetBeginAtomIdx()]
                            b = coordinates[bond.GetEndAtomIdx()]

                            ax.plot(
                                [a.x, b.x],
                                [a.y, b.y],
                                [a.z, b.z],
                            )

                        ax.scatter(
                            [p.x for p in coordinates],
                            [p.y for p in coordinates],
                            [p.z for p in coordinates],
                        )

                        ax.set_title(f"3D structure: {selected}")
                        st.pyplot(fig)
                        plt.close(fig)

                    except Exception as exc:
                        st.error(f"3D visualization failed: {exc}")

                    st.download_button(
                        "Download 3D structure (.mol)",
                        data=result_3d["mol_block"],
                        file_name=f"{selected}_3D.mol",
                        mime="chemical/x-mdl-molfile",
                    )

                else:
                    st.info(
                        "Click Generate 3D structure to create a conformer."
                    )

                st.caption(
                    "This is a computationally generated conformer, "
                    "not an experimentally determined protein-bound structure."
                )

            # ------------------------------------------------
            # 4. Molecular graph
            # ------------------------------------------------

            with tab_graph:
                st.markdown("### Molecular graph representation")

                st.write(
                    "Each atom is represented as a node and each chemical "
                    "bond as an edge."
                )

                graph = nx.Graph()

                for atom in mol.GetAtoms():
                    graph.add_node(
                        atom.GetIdx(),
                        element=atom.GetSymbol(),
                        charge=atom.GetFormalCharge(),
                        aromatic=atom.GetIsAromatic(),
                        degree=atom.GetDegree(),
                    )

                for bond in mol.GetBonds():
                    graph.add_edge(
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        bond_type=str(bond.GetBondType()),
                    )

                fig, ax = plt.subplots(figsize=(10, 7))

                positions = nx.spring_layout(
                    graph,
                    seed=42,
                    k=0.8,
                )

                node_labels = {
                    node: data["element"]
                    for node, data in graph.nodes(data=True)
                }

                nx.draw_networkx(
                    graph,
                    positions,
                    labels=node_labels,
                    with_labels=True,
                    node_size=650,
                    font_size=8,
                    ax=ax,
                )

                ax.set_title(f"Molecular graph: {selected}")
                ax.axis("off")

                st.pyplot(fig)
                plt.close(fig)

                st.markdown("#### Graph statistics")

                graph_cols = st.columns(3)

                graph_cols[0].metric(
                    "Nodes",
                    graph.number_of_nodes(),
                )

                graph_cols[1].metric(
                    "Edges",
                    graph.number_of_edges(),
                )

                graph_cols[2].metric(
                    "Connected components",
                    nx.number_connected_components(graph),
                )

                atom_rows = [
                    {
                        "Atom index": node,
                        "Element": data["element"],
                        "Formal charge": data["charge"],
                        "Aromatic": data["aromatic"],
                        "Degree": data["degree"],
                    }
                    for node, data in graph.nodes(data=True)
                ]

                st.markdown("#### Atom-level features")

                st.dataframe(
                    pd.DataFrame(atom_rows),
                    hide_index=True,
                    use_container_width=True,
                )

                bond_rows = [
                    {
                        "Start atom": a,
                        "End atom": b,
                        "Bond type": data["bond_type"],
                    }
                    for a, b, data in graph.edges(data=True)
                ]

                st.markdown("#### Bond-level features")

                st.dataframe(
                    pd.DataFrame(bond_rows),
                    hide_index=True,
                    use_container_width=True,
                )

            # ------------------------------------------------
            # 5. Morgan fingerprints
            # ------------------------------------------------

            with tab_fp:
                st.markdown("### Morgan molecular fingerprints")

                st.write(
                    "Morgan fingerprints encode local molecular "
                    "environments as a fixed-length binary representation."
                )

                fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=2,
                    nBits=2048,
                )

                active_bits = list(fingerprint.GetOnBits())

                fp_cols = st.columns(3)

                fp_cols[0].metric("Fingerprint length", 2048)
                fp_cols[1].metric("Radius", 2)
                fp_cols[2].metric("Active bits", len(active_bits))

                st.markdown("#### Fingerprint activity")

                fp_df = pd.DataFrame(
                    {
                        "Bit status": ["Active", "Inactive"],
                        "Count": [
                            len(active_bits),
                            2048 - len(active_bits),
                        ],
                    }
                )

                st.bar_chart(
                    fp_df.set_index("Bit status")
                )

                st.markdown("#### Active fingerprint bit positions")

                st.code(
                    ", ".join(map(str, active_bits))
                    if active_bits else "No active bits",
                    language=None,
                )

                st.download_button(
                    "Download active fingerprint bits",
                    data="\n".join(map(str, active_bits)),
                    file_name=f"{selected}_morgan_bits.txt",
                    mime="text/plain",
                )

                st.caption(
                    "This visualization calculates a radius-2, 2048-bit "
                    "Morgan fingerprint for the selected drug. It does not "
                    "by itself explain an interaction prediction."
                )

            st.markdown("---")
            st.caption(
                "Molecular descriptors, fingerprints, graph features and "
                "computed conformers are research representations. They "
                "do not independently establish clinical efficacy, "
                "interaction severity or medication safety."
            )


# ============================================================
# Data explorer
# ============================================================

with tab_data:
    st.markdown('<div class="section-eyebrow">Research data intelligence</div>', unsafe_allow_html=True)
    st.title("Data Explorer")
    st.markdown(
        "Explore DDIvolve's integrated drug interaction dataset, "
        "inspect interaction patterns, and examine research data coverage."
    )

    if interactions.empty:
        st.warning("No interaction records are available to explore.")
        display_dataset_status()

    else:
        # --------------------------------------------------
        # 1. Dataset overview
        # --------------------------------------------------
        total_rows = len(interactions)
        unique_drugs = len(
            set(interactions["Drug_A"].dropna().astype(str))
            | set(interactions["Drug_B"].dropna().astype(str))
        )
        unique_pairs = interactions[
            ["Drug_A", "Drug_B"]
        ].drop_duplicates().shape[0]
        unique_labels = interactions["Evidence_Label"].nunique()

        st.markdown("### Dataset overview")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Interaction records", f"{total_rows:,}")
        m2.metric("Unique drugs in records", f"{unique_drugs:,}")
        m3.metric("Unique drug pairs", f"{unique_pairs:,}")
        m4.metric("Distinct labels", f"{unique_labels:,}")

        st.caption(
            "Counts describe the currently loaded interaction dataset, "
            "not population-level clinical prevalence."
        )

        st.divider()

        # --------------------------------------------------
        # 2. Interaction distribution
        # --------------------------------------------------
        st.markdown("### Interaction distribution")

        label_counts = (
            interactions["Evidence_Label"]
            .fillna("Unspecified")
            .astype(str)
            .value_counts()
            .rename_axis("Interaction family")
            .reset_index(name="Records")
        )

        chart_col, summary_col = st.columns([1.5, 1])

        with chart_col:
            st.markdown("**Records by interaction label**")
            st.bar_chart(
                label_counts.set_index("Interaction family"),
                height=340,
            )

        with summary_col:
            st.markdown("**Dataset composition**")
            st.dataframe(
                label_counts,
                hide_index=True,
                use_container_width=True,
                height=340,
            )

        st.divider()

        # --------------------------------------------------
        # 3. Dataset search and filters
        # --------------------------------------------------
        st.markdown("### Research record explorer")
        st.markdown(
            "Search individual drug names or interaction descriptions "
            "and narrow the dataset using interaction labels."
        )

        f1, f2 = st.columns([1.2, 1])

        with f1:
            search_text = st.text_input(
                "Search research records",
                placeholder="Enter a drug name or interaction description...",
                key="data_search",
            )

        with f2:
            labels = sorted(
                interactions["Evidence_Label"]
                .dropna()
                .astype(str)
                .unique()
                .tolist(),
                key=str.casefold,
            )

            selected_labels = st.multiselect(
                "Interaction family",
                options=labels,
                default=[],
                help="Leave empty to include all interaction families.",
            )

        filtered = interactions.copy()

        if search_text.strip():
            term = search_text.strip().casefold()

            search_columns = [
                column for column in filtered.columns
                if filtered[column].dtype == "object"
                or pd.api.types.is_string_dtype(filtered[column])
            ]

            mask = pd.Series(False, index=filtered.index)

            for column in search_columns:
                mask |= (
                    filtered[column]
                    .fillna("")
                    .astype(str)
                    .str.casefold()
                    .str.contains(term, regex=False)
                )

            filtered = filtered[mask]

        if selected_labels:
            filtered = filtered[
                filtered["Evidence_Label"].astype(str).isin(selected_labels)
            ]

        st.write(
            f"**{len(filtered):,}** matching records "
            f"out of **{total_rows:,}** total records."
        )

        # --------------------------------------------------
        # 4. Filtered dataset metrics
        # --------------------------------------------------
        p1, p2, p3 = st.columns(3)

        p1.metric("Filtered records", f"{len(filtered):,}")

        if not filtered.empty:
            filtered_pairs = filtered[
                ["Drug_A", "Drug_B"]
            ].drop_duplicates().shape[0]
            filtered_labels = filtered["Evidence_Label"].nunique()
        else:
            filtered_pairs = 0
            filtered_labels = 0

        p2.metric("Unique pairs", f"{filtered_pairs:,}")
        p3.metric("Interaction labels", f"{filtered_labels:,}")

        # --------------------------------------------------
        # 5. Research records table
        # --------------------------------------------------
        st.markdown("**Interaction records**")

        if filtered.empty:
            st.info("No records match the selected search and filters.")
        else:
            st.dataframe(
                filtered.head(500),
                hide_index=True,
                use_container_width=True,
                height=420,
            )

            if len(filtered) > 500:
                st.caption(
                    "Showing the first 500 records for responsiveness. "
                    "The CSV export includes every matching record."
                )

            csv_bytes = filtered.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download filtered research data",
                data=csv_bytes,
                file_name="ddivolve_filtered_interactions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()

        # --------------------------------------------------
        # 6. Data quality and coverage
        # --------------------------------------------------
        with st.expander("Dataset quality and coverage", expanded=False):
            q1, q2, q3 = st.columns(3)

            q1.metric(
                "Missing drug A values",
                f"{interactions['Drug_A'].isna().sum():,}",
            )
            q2.metric(
                "Missing drug B values",
                f"{interactions['Drug_B'].isna().sum():,}",
            )
            q3.metric(
                "Missing interaction labels",
                f"{interactions['Evidence_Label'].isna().sum():,}",
            )

            st.markdown("**Loaded dataset structure**")

            schema = pd.DataFrame({
                "Column": interactions.columns,
                "Data type": interactions.dtypes.astype(str).values,
                "Missing values": interactions.isna().sum().values,
                "Available values": interactions.notna().sum().values,
            })

            st.dataframe(
                schema,
                hide_index=True,
                use_container_width=True,
            )

        st.caption(
            "Research note: An interaction record represents dataset evidence. "
            "It does not independently establish clinical severity, "
            "causality, or the safety of a drug combination."
        )

        # ============================================================
# AI EXPLANATION
# ============================================================
# ============================================================
# AI EXPLANATION
# ============================================================

with tab_explanation:
    import pandas as pd
    from src.inference.explain import explain_pair

    st.markdown(
        '<div class="section-eyebrow">Explainable artificial intelligence</div>',
        unsafe_allow_html=True,
    )
    st.title("AI Explanation")
    st.markdown(
        "Understand how the trained XGBoost model arrives at its "
        "predicted drug interaction family using pair-specific TreeSHAP."
    )

    drug_names = get_drug_names()

    if len(drug_names) < 2:
        st.warning("At least two drugs are required for explanation.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            drug1 = st.selectbox(
                "Select first medicine",
                drug_names,
                key="explain_drug1",
            )

        second_options = [name for name in drug_names if name != drug1]

        with col2:
            drug2 = st.selectbox(
                "Select second medicine",
                second_options,
                key="explain_drug2",
            )

        if st.button(
            "Generate AI Explanation",
            type="primary",
            use_container_width=True,
        ):
            try:
                record1 = get_drug_record(drug1)
                record2 = get_drug_record(drug2)

                def extract_smiles(record):
                    if record is None:
                        return None

                    for column in ("canonical_smiles", "smiles"):
                        value = record.get(column)

                        if pd.notna(value) and str(value).strip():
                            return str(value).strip()

                    return None

                smiles1 = extract_smiles(record1)
                smiles2 = extract_smiles(record2)

                if not smiles1 or not smiles2:
                    st.error(
                        "A valid molecular structure is unavailable "
                        "for one or both selected medicines."
                    )
                else:
                    with st.spinner(
                        "Generating XGBoost TreeSHAP explanation..."
                    ):
                        predictor = load_ddi_predictor()
                        result = explain_pair(
                            predictor,
                            smiles1,
                            smiles2,
                            top_k=20,
                        )

                    st.session_state["latest_explanation"] = result
                    st.session_state["latest_explanation_pair"] = (
                        drug1,
                        drug2,
                    )

            except Exception as error:
                st.error(f"Explanation could not be generated: {error}")

        result = st.session_state.get("latest_explanation")
        saved_pair = st.session_state.get("latest_explanation_pair")

        if result and saved_pair:
            st.divider()

            st.markdown("### Prediction summary")
            st.caption(f"{saved_pair[0]} + {saved_pair[1]}")

            a, b, c = st.columns(3)

            a.metric(
                "Predicted interaction family",
                result["predicted_family"],
            )

            b.metric(
                "Model confidence",
                f"{result['confidence'] * 100:.2f}%",
            )

            c.metric(
                "Explanation method",
                result["explanation_type"],
            )

            # Interaction family probabilities
            st.markdown("### Interaction family probabilities")

            probability_df = pd.DataFrame([
                {
                    "Interaction family": family,
                    "Probability (%)": probability * 100,
                }
                for family, probability in result["probabilities"].items()
            ]).sort_values("Probability (%)", ascending=False)

            st.table(probability_df.round(2))

            # Top SHAP features
            st.markdown("### Top contributing features")

            feature_df = pd.DataFrame(result["top_features"])

            if not feature_df.empty:
                st.table(
                    feature_df[
                        [
                            "feature",
                            "feature_value",
                            "shap_value",
                            "group",
                            "direction",
                        ]
                    ].round(4)
                )

            # Feature group analysis
            st.markdown("### Molecular feature group analysis")

            group_df = pd.DataFrame(result["group_importance"])

            if not group_df.empty:
                group_df = group_df.rename(
                    columns={
                        "group": "Feature group",
                        "importance": "Aggregated absolute SHAP",
                    }
                )

                st.table(group_df.round(4))

            st.info(
                "SHAP contributions explain how individual features "
                "influence the model's raw predicted-class score. "
                "Positive contributions support the predicted class, "
                "while negative contributions oppose it. These values "
                "are not probabilities and do not establish biological "
                "causality, clinical severity, or medication safety."
            )



            
with tab_reevaluation:
    import pandas as pd
    import plotly.express as px
    from severity_module.src.assess_severity import assess_severity

    st.markdown(
        '<div class="section-eyebrow">Computational validation</div>',
        unsafe_allow_html=True
    )
    st.subheader("Re-evaluation Analysis")

    st.markdown(
        "Compare an original drug combination with a proposed research "
        "candidate using the existing XGBoost interaction-family model."
    )

    drug_names = get_drug_names()

    col1, col2 = st.columns(2)

    with col1:
        original_a = st.selectbox(
            "Original Drug 1",
            drug_names,
            key="reeval_original_a"
        )

    with col2:
        original_b = st.selectbox(
            "Original Drug 2",
            [d for d in drug_names if d != original_a],
            key="reeval_original_b"
        )

    st.markdown("### Research candidate")

    replacement_position = st.radio(
        "Which drug should the candidate replace?",
        ["Drug 1", "Drug 2"],
        horizontal=True,
        key="reeval_replacement_position"
    )

    excluded = {original_a.casefold(), original_b.casefold()}
    candidate_options = [
        d for d in drug_names if d.casefold() not in excluded
    ]

    candidate = st.selectbox(
        "Select candidate medicine",
        candidate_options,
        key="reeval_candidate"
    )

    if st.button(
        "Run comparative re-evaluation",
        type="primary",
        key="run_reevaluation"
    ):
        try:
            with st.spinner("Evaluating both combinations..."):

                original_result = predict_ddi_family(
                    original_a, original_b
                )

                if replacement_position == "Drug 1":
                    candidate_a, candidate_b = candidate, original_b
                else:
                    candidate_a, candidate_b = original_a, candidate

                candidate_result = predict_ddi_family(
                    candidate_a, candidate_b
                )

                original_evidence = assess_severity(
                    original_a, original_b
                )

                candidate_evidence = assess_severity(
                    candidate_a, candidate_b
                )

            st.session_state["reevaluation_results"] = {
                "original_pair": (original_a, original_b),
                "candidate_pair": (candidate_a, candidate_b),
                "original_result": original_result,
                "candidate_result": candidate_result,
                "original_evidence": original_evidence,
                "candidate_evidence": candidate_evidence,
            }

        except Exception as e:
            st.error(f"Re-evaluation failed: {e}")

    results = st.session_state.get("reevaluation_results")

    if results:

        st.markdown("---")
        st.markdown("### 01 · Prediction comparison")

        original_result = results["original_result"]
        candidate_result = results["candidate_result"]

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Original combination")
            st.write(
                f"**{results['original_pair'][0]} + "
                f"{results['original_pair'][1]}**"
            )
            st.metric(
                "Predicted interaction family",
                original_result["interaction_family"]
            )
            st.metric(
                "Prediction confidence",
                f"{original_result['confidence']:.2%}"
            )

        with c2:
            st.markdown("#### Research candidate")
            st.write(
                f"**{results['candidate_pair'][0]} + "
                f"{results['candidate_pair'][1]}**"
            )
            st.metric(
                "Predicted interaction family",
                candidate_result["interaction_family"]
            )
            st.metric(
                "Prediction confidence",
                f"{candidate_result['confidence']:.2%}"
            )

        st.markdown("### 02 · Family probability comparison")

        probability_df = pd.DataFrame({
            "Interaction family": list(
                original_result["probabilities"].keys()
            ),
            "Original combination": list(
                original_result["probabilities"].values()
            ),
            "Research candidate": [
                candidate_result["probabilities"].get(
                    family, 0
                )
                for family in original_result["probabilities"].keys()
            ],
        })

        probability_df["Original combination"] *= 100
        probability_df["Research candidate"] *= 100

        st.dataframe(
            probability_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        chart_df = probability_df.melt(
            id_vars="Interaction family",
            var_name="Combination",
            value_name="Model probability (%)"
        )

        fig = px.bar(
            chart_df,
            x="Interaction family",
            y="Model probability (%)",
            color="Combination",
            barmode="group"
        )

        fig.update_layout(
            xaxis_tickangle=-35,
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 03 · Source evidence comparison")

        evidence_df = pd.DataFrame([
            {
                "Combination": "Original",
                "DDInter match": results["original_evidence"].get("matched"),
                "Source-reported severity": results["original_evidence"].get(
                    "source_reported_level", "Not available"
                ),
                "Evidence source": results["original_evidence"].get(
                    "evidence_source", "Not available"
                ),
            },
            {
                "Combination": "Research candidate",
                "DDInter match": results["candidate_evidence"].get("matched"),
                "Source-reported severity": results["candidate_evidence"].get(
                    "source_reported_level", "Not available"
                ),
                "Evidence source": results["candidate_evidence"].get(
                    "evidence_source", "Not available"
                ),
            },
        ])

        st.dataframe(
            evidence_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 04 · Research interpretation")

        if (
            original_result["interaction_family"]
            == candidate_result["interaction_family"]
        ):
            st.info(
                "The model predicted the same interaction family "
                "for both combinations."
            )
        else:
            st.info(
                "The model predicted different interaction families "
                "for the two combinations."
            )

        st.warning(
            "This is a computational comparison, not a therapeutic "
            "substitution recommendation. Model probabilities are not "
            "clinical risk scores. A lower probability, different family, "
            "or missing DDInter record does not establish that a candidate "
            "is safer or effective. Expert and experimental validation "
            "are required."
        )

# ============================================================
# Intervention research
# ============================================================

with tab_research:
    import pandas as pd
    from src.inference.intervention import analyze_intervention_research

    st.markdown(
        '<div class="section-eyebrow">Future-facing research</div>',
        unsafe_allow_html=True
    )
    st.subheader("Intervention research pathways")

    st.markdown(
        "Explore evidence-based research workflows and molecular intervention "
        "evaluation. This module does not prescribe treatments."
    )

    pathway = st.radio(
        "Choose a pathway",
        [
            "Clinical evidence review",
            "Molecular research and model evaluation",
        ],
        horizontal=True,
        key="research_pathway",
    )

    st.markdown("")

    if pathway == "Clinical evidence review":

        st.markdown("### Pair-specific evidence assessment")

        drug_names = get_drug_names()

        col1, col2 = st.columns(2, gap="large")

        with col1:
            drug_a = st.selectbox(
                "Select first medicine",
                drug_names,
                key="research_drug_a"
            )

        with col2:
            available_drugs = [d for d in drug_names if d != drug_a]
            drug_b = st.selectbox(
                "Select second medicine",
                available_drugs,
                key="research_drug_b"
            )

        if st.button(
            "Analyze Research Pathway",
            type="primary",
            key="analyze_intervention"
        ):
            try:
                result = analyze_intervention_research(drug_a, drug_b)

                st.markdown("---")
                st.markdown("### 01 · Research assessment")

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Source-reported severity",
                        result.get("severity", "Not assessed")
                    )

                with c2:
                    st.metric(
                        "Evidence match",
                        "Matched" if result.get("matched") else "No exact match"
                    )

                st.markdown("**Research pathway**")
                st.info(result.get("research_pathway", "Not available"))

                st.markdown("**Research focus**")
                st.write(result.get("research_focus", "Not available"))

                st.markdown("### 02 · Interaction evidence")

                description = result.get("interaction_description")
                st.write(description or "No matching interaction description found.")

                e1, e2 = st.columns(2)

                with e1:
                    st.write(
                        "**Evidence source:**",
                        result.get("evidence_source") or "Not available"
                    )
                    st.write(
                        "**Evidence quality:**",
                        result.get("evidence_quality") or "Not available"
                    )

                with e2:
                    st.write(
                        "**Assessment status:**",
                        result.get("assessment_status") or "Not assessed"
                    )

                st.markdown("### 03 · Management evidence")

                management = result.get("management")

                if management:
                    st.write(management)
                else:
                    st.info(
                        "No source-provided management recommendation is "
                        "available for this record."
                    )

                st.caption(
                    "Clinical effect: "
                    + str(result.get("clinical_effect") or "Not available")
                )

                st.caption(
                    "Clinical significance: "
                    + str(result.get("clinical_significance") or "Not available")
                )

                st.markdown("### 04 · Evidence limitations")

                st.warning(
                    result.get(
                        "evidence_scope",
                        "Independent clinical verification has not been performed."
                    )
                )

                st.warning(
                    "DDIvolve is a research prototype. This assessment does not "
                    "recommend starting, stopping, replacing, or changing medication."
                )

            except Exception as e:
                st.error(f"Research analysis failed: {e}")

        st.markdown("---")

        a, b = st.columns(2, gap="large")

        with a:
            with st.container(border=True):
                st.markdown("### Professional evidence review")
                st.markdown(
                    "- Confirm medicine identities and formulations.\n"
                    "- Review documented interaction evidence.\n"
                    "- Identify missing patient-specific information.\n"
                    "- Record evidence limitations."
                )

        with b:
            with st.container(border=True):
                st.markdown("### Professional assessment")
                st.markdown(
                    "- Consider indication, dose, and medical history.\n"
                    "- Review monitoring requirements.\n"
                    "- Require qualified clinician or pharmacist assessment."
                )

    else:

        st.markdown("### Molecular research workflow")

        a, b = st.columns(2, gap="large")

        with a:
            with st.container(border=True):
                st.markdown("### 01 · Molecular pipeline")
                st.markdown(
                    "- Validate SMILES and compound identity.\n"
                    "- Generate RDKit molecular descriptors.\n"
                    "- Generate Morgan fingerprints and molecular graphs.\n"
                    "- Preserve source IDs and missing-structure records."
                )

        with b:
            with st.container(border=True):
                st.markdown("### 02 · Model validation")
                st.markdown(
                    "- Verify pair construction and labels.\n"
                    "- Check pair leakage in data splitting.\n"
                    "- Evaluate precision, recall, F1 and PR-AUC.\n"
                    "- Track model version and prediction confidence."
                )

        st.markdown("### 03 · Future intervention research")

        st.markdown(
            "- Explore evidence-supported therapeutic alternatives.\n"
            "- Investigate computational molecular modifications.\n"
            "- Compare molecular descriptors and drug-likeness.\n"
            "- Re-evaluate candidate combinations using the trained model.\n"
            "- Record limitations and experimental validation requirements."
        )

        st.info(
            "A lower model prediction score does not establish that a "
            "molecular modification is safer or clinically effective. "
            "Experimental and expert validation are required."
        )


# ============================================================
# Footer
# ============================================================

st.divider()
footer_left, footer_right = st.columns([1, 1])
with footer_left:
    st.caption("DDIvolve · Drug–Drug Interaction Detection and Mitigation")
with footer_right:
    st.caption("Student research prototype · Evidence may be incomplete")

