from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# DDIvolve | Research dashboard
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

# Preferred app-ready files (for example, produced by your friend's
# PubChem pipeline or a future preprocessing script).
INTERACTIONS_FILE = DATA_DIR / "interactions.csv"
DRUGS_FILE = DATA_DIR / "drugs.csv"

# Existing research dataset files currently in your project.
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

tab_overview, tab_checker, tab_molecular, tab_data, tab_research = st.tabs(
    [
        "⌂ Overview",
        "⇄ Interaction checker",
        "⚛ Molecular explorer",
        "▤ Data explorer",
        "✧ Intervention research",
    ]
)


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

with tab_checker:
    st.markdown('<div class="section-eyebrow">Evidence lookup</div>', unsafe_allow_html=True)
    st.subheader("Drug interaction checker")
    st.markdown(
        '<div class="subtle">Search the currently loaded research dataset. '
        'A match is evidence of a dataset record, not a personalized clinical assessment.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    names = get_drug_names()
    if interactions.empty:
        st.error("No usable interaction data is loaded.")
        display_dataset_status()
    elif len(names) < 2:
        st.warning("At least two drug entries are needed to check a pair.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            drug_a = st.selectbox("First drug", names, key="checker_drug_a")
        with c2:
            candidates = [n for n in names if n.casefold() != drug_a.casefold()]
            drug_b = st.selectbox(
                "Second drug",
                candidates or names,
                key="checker_drug_b",
            )

        run_check = st.button("Search interaction evidence", type="primary", use_container_width=True)

        if run_check:
            if drug_a.casefold() == drug_b.casefold():
                st.warning("Select two different drugs.")
            else:
                matches = find_pair_records(drug_a, drug_b)
                if not matches.empty:
                    st.success(f"Found {len(matches):,} matching record(s) in the loaded dataset.")
                    st.markdown("#### Evidence records")
                    view_cols = ["Drug_A", "Drug_B", "Evidence_Label"]
                    if "Negative_Sample_ID" in matches.columns:
                        view_cols.append("Negative_Sample_ID")
                    st.dataframe(
                        matches[view_cols].reset_index(drop=True),
                        hide_index=True,
                        use_container_width=True,
                    )
                   
                else:
                    st.warning("No matching pair record was found in the loaded dataset.")
                    st.markdown(
                        '<div class="notice"><strong>Coverage limitation</strong><br>'
                        'No record found does not mean the combination is safe or has no interaction.</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("")
    st.caption("For real medication questions, consult a qualified doctor or pharmacist.")


# ============================================================
# Molecular explorer
# ============================================================

with tab_molecular:
    st.markdown('<div class="section-eyebrow">Chemical profile</div>', unsafe_allow_html=True)
    st.subheader("Molecular explorer")
    st.markdown(
        '<div class="subtle">Browse structure strings and descriptors supplied by the current molecular CSV. '
        'RDKit-derived features can be integrated in the next development step.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    if drugs.empty:
        st.info("Molecular data is not available yet. The app can use `data/drug_smiles.csv` now and `data/drugs.csv` when the PubChem enrichment file is ready.")
        display_dataset_status()
    else:
        drug_names = sorted(drugs["drug_name"].unique(), key=str.casefold)
        selected = st.selectbox("Choose a drug", drug_names, key="molecular_selected_drug")
        record = get_drug_record(selected)

        if record is not None:
            left, right = st.columns([0.85, 1.15], gap="large")
            with left:
                with st.container(border=True):
                    st.markdown("**Compound identity**")
                    st.markdown(f"### {safe_text(record.get('drug_name'))}")
                    for col_name, label in [
                        ("drug_id", "Source drug ID"),
                        ("pubchem_cid", "PubChem CID"),
                        ("molecular_formula", "Molecular formula"),
                        ("inchi_key", "InChIKey"),
                        ("lookup_status", "Lookup status"),
                    ]:
                        if col_name in record.index:
                            st.write(f"**{label}:** {safe_text(record.get(col_name))}")

            with right:
                with st.container(border=True):
                    st.markdown("**SMILES representation**")
                    smiles = safe_text(record.get("canonical_smiles"), "")
                    if not smiles:
                        smiles = safe_text(record.get("isomeric_smiles"), "")
                    if not smiles:
                        smiles = safe_text(record.get("smiles"), "")
                    if smiles:
                        st.code(smiles, language=None)
                    else:
                        st.info("No SMILES string is available for this entry.")

                    descriptor_map = [
                        ("molecular_weight", "Molecular weight", 2),
                        ("logp", "LogP", 2),
                        ("qed", "QED", 3),
                        ("h_bond_donors", "H-bond donors", 0),
                        ("h_bond_acceptors", "H-bond acceptors", 0),
                        ("tpsa", "TPSA", 2),
                        ("rotatable_bonds", "Rotatable bonds", 0),
                        ("atom_count", "Atom count", 0),
                        ("bond_count", "Bond count", 0),
                    ]
                    available = [
                        (key, label, decimals) for key, label, decimals in descriptor_map
                        if key in record.index and safe_text(record.get(key), "") != ""
                    ]
                    if available:
                        for start in range(0, len(available), 3):
                            cols = st.columns(3)
                            for col, (key, label, decimals) in zip(cols, available[start:start + 3]):
                                col.metric(label, safe_number(record.get(key), decimals))
                    else:
                        st.info("No calculated descriptors found yet. RDKit can generate these from valid SMILES.")

            st.caption(
                "Molecular descriptors describe chemical properties; they do not independently establish "
                "interaction risk, clinical severity, efficacy, or safety."
            )


# ============================================================
# Data explorer
# ============================================================

with tab_data:
    st.markdown('<div class="section-eyebrow">Dataset workspace</div>', unsafe_allow_html=True)
    st.subheader("Explore research records")
    st.markdown("Filter the currently loaded interactions and export the selected view.")

    if interactions.empty:
        st.warning("No interaction records are available to explore.")
        display_dataset_status()
    else:
        f1, f2 = st.columns([1.1, 0.9])
        with f1:
            search_text = st.text_input(
                "Search drug names or label",
                placeholder="Type a drug ID/name or label…",
                key="data_search",
            )
        with f2:
            labels = sorted(
                interactions["Evidence_Label"].dropna().astype(str).unique().tolist(),
                key=str.casefold,
            )
            selected_labels = st.multiselect(
                "Filter labels",
                options=labels,
                default=[],
                help="Leave blank to include all labels.",
            )

        filtered = interactions.copy()
        if search_text.strip():
            term = search_text.strip().casefold()
            mask = (
                filtered["Drug_A"].str.casefold().str.contains(term, regex=False)
                | filtered["Drug_B"].str.casefold().str.contains(term, regex=False)
                | filtered["Evidence_Label"].str.casefold().str.contains(term, regex=False)
            )
            filtered = filtered[mask]
        if selected_labels:
            filtered = filtered[filtered["Evidence_Label"].isin(selected_labels)]

        st.write(f"Showing **{len(filtered):,}** of **{len(interactions):,}** rows.")
        st.dataframe(filtered.head(500), hide_index=True, use_container_width=True)

        if len(filtered) > 500:
            st.caption("The preview is limited to the first 500 matching rows for responsiveness. The download includes all filtered rows.")

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name="ddivolve_filtered_interactions.csv",
            mime="text/csv",
            use_container_width=False,
        )

        with st.expander("Dataset profile"):
            profile1, profile2, profile3 = st.columns(3)
            profile1.metric("Rows after filters", f"{len(filtered):,}")
            profile2.metric("Unique pairs after filters", f"{filtered[['Drug_A', 'Drug_B']].drop_duplicates().shape[0]:,}")
            profile3.metric("Distinct labels after filters", f"{filtered['Evidence_Label'].nunique():,}")

            if not filtered.empty:
                label_counts = (
                    filtered["Evidence_Label"]
                    .value_counts()
                    .head(15)
                    .rename_axis("Label")
                    .reset_index(name="Rows")
                )
                st.markdown("**Most frequent labels in this view**")
                st.bar_chart(label_counts.set_index("Label"))


# ============================================================
# Intervention research
# ============================================================

with tab_research:
    st.markdown('<div class="section-eyebrow">Future-facing research</div>', unsafe_allow_html=True)
    st.subheader("Intervention research pathways")
    st.markdown(
        "This module records possible research workflows. It does not prescribe a treatment, "
        "choose replacement medicines, or redesign a molecule automatically."
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
        a, b = st.columns(2, gap="large")
        with a:
            with st.container(border=True):
                st.markdown("### 01 · Gather evidence")
                st.markdown(
                    "- Confirm the exact medicine identities and formulations.\n"
                    "- Review documented interaction evidence and its source.\n"
                    "- Identify uncertainty and missing patient-specific context."
                )
        with b:
            with st.container(border=True):
                st.markdown("### 02 · Professional assessment")
                st.markdown(
                    "- Have a licensed clinician or pharmacist assess relevance.\n"
                    "- Consider indication, dose, history, and monitoring needs.\n"
                    "- Do not recommend an alternative without qualified review."
                )
        st.warning(
            "DDIvolve is a student research prototype, not a clinical decision-support "
            "system. Do not start, stop, or change medication based on this app."
        )
    else:
        a, b = st.columns(2, gap="large")
        with a:
            with st.container(border=True):
                st.markdown("### 01 · Molecular pipeline")
                st.markdown(
                    "- Validate SMILES and compound identity.\n"
                    "- Generate RDKit descriptors and molecular fingerprints.\n"
                    "- Preserve source IDs and record invalid or missing structures."
                )
        with b:
            with st.container(border=True):
                st.markdown("### 02 · Model validation")
                st.markdown(
                    "- Verify positive and negative pair construction.\n"
                    "- Split data without pair leakage where appropriate.\n"
                    "- Report precision, recall, F1, PR-AUC, and calibration.\n"
                    "- Track model version, limitations, and prediction confidence."
                )
        st.info(
            "A change in descriptors or a model score alone does not prove that a "
            "molecular modification reduces interaction risk. Experimental and expert "
            "validation would be required."
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

