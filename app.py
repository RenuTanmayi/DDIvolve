
from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import pandas as pd
import streamlit as st


# ============================================================
# DDIvolve | Integrated Research Dashboard
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

# Original DDIvolve datasets
INTERACTIONS_FILE = DATA_DIR / "interactions.csv"
DRUGS_FILE = DATA_DIR / "drugs.csv"
RAW_DDIS_FILE = DATA_DIR / "ddis.csv"
RAW_SMILES_FILE = DATA_DIR / "drug_smiles.csv"

# New clinical evidence module
CLINICAL_EVIDENCE_FILE = (
    APP_DIR
    / "severity_module"
    / "data"
    / "processed"
    / "clinical_evidence.csv"
)

st.set_page_config(
    page_title="DDIvolve | Interaction Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
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

    .severity-card {
        border: 1px solid var(--ddi-line);
        background: rgba(148, 163, 184, 0.075);
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.7rem 0;
    }

    .severity-value {
        font-family: 'Manrope', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.2rem 0;
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
# GENERAL DATA UTILITIES
# ============================================================

def read_csv_safely(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), f"File not found: {path.name}"

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False
        ), None

    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                encoding="latin-1",
                low_memory=False
            ), None
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


def normalize(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(r"\s+", " ", str(value).strip().lower())


def safe_text(value: Any, fallback: str = "Not available") -> str:
    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value or value.casefold() in {
        "nan", "none", "<na>", "nat"
    }:
        return fallback

    return value


def safe_number(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value)
        if pd.isna(number):
            return "—"
        return f"{number:,.{decimals}f}"
    except (TypeError, ValueError, OverflowError):
        return "—"


def unordered_pair(a: Any, b: Any) -> tuple[str, str]:
    return tuple(sorted([normalize(a), normalize(b)]))


# ============================================================
# ORIGINAL INTERACTION DATA
# ============================================================

@st.cache_data(show_spinner="Loading interaction records...")
def load_interaction_data():

    path = first_existing(
        INTERACTIONS_FILE,
        RAW_DDIS_FILE
    )

    if path is None:
        return pd.DataFrame(), (
            "No interaction file found. Add data/ddis.csv or "
            "data/interactions.csv."
        ), "missing"

    df, error = read_csv_safely(path)

    if error:
        return pd.DataFrame(), error, path.name

    columns_lower = {
        str(col).strip().casefold(): col
        for col in df.columns
    }

    drug_a_col = next(
        (
            columns_lower[key]
            for key in ("drug_a", "drug1", "drug_1", "d1")
            if key in columns_lower
        ),
        None,
    )

    drug_b_col = next(
        (
            columns_lower[key]
            for key in ("drug_b", "drug2", "drug_2", "d2")
            if key in columns_lower
        ),
        None,
    )

    description_col = next(
        (
            columns_lower[key]
            for key in (
                "description",
                "interaction",
                "interaction_description",
                "type",
            )
            if key in columns_lower
        ),
        None,
    )

    if not drug_a_col or not drug_b_col:

        if len(df.columns) < 3:
            return pd.DataFrame(), (
                f"{path.name} needs at least three columns."
            ), path.name

        if path.name == INTERACTIONS_FILE.name:
            drug_a_col = df.columns[0]
            drug_b_col = df.columns[1]
            description_col = df.columns[2]

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
        result["Evidence_Label"] = clean_text_series(
            df[description_col]
        )
    else:
        result["Evidence_Label"] = "Label not supplied"

    if "Neg samples" in df.columns:
        result["Negative_Sample_ID"] = clean_text_series(
            df["Neg samples"]
        )

    result = result[
        (result["Drug_A"] != "")
        & (result["Drug_B"] != "")
        & (result["Drug_A"].str.casefold() != "nan")
        & (result["Drug_B"].str.casefold() != "nan")
    ].copy()

    return result.reset_index(drop=True), None, path.name


# ============================================================
# MOLECULAR DATA
# ============================================================

@st.cache_data(show_spinner="Loading molecular records...")
def load_drug_data():

    path = first_existing(
        DRUGS_FILE,
        RAW_SMILES_FILE
    )

    if path is None:
        return pd.DataFrame(), (
            "No molecular file found. Add data/drug_smiles.csv "
            "or data/drugs.csv."
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
    df = df.drop_duplicates(
        subset=["drug_name"],
        keep="first"
    )

    return df.reset_index(drop=True), None, path.name


# ============================================================
# CLINICAL EVIDENCE MODULE
# ============================================================

@st.cache_data(show_spinner="Loading clinical evidence...")
def load_clinical_evidence():

    df, error = read_csv_safely(CLINICAL_EVIDENCE_FILE)

    if error:
        return pd.DataFrame(), error

    required = [
        "drug1_name",
        "drug2_name",
        "drug1_key",
        "drug2_key",
        "severity",
        "source_reported_level",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        return pd.DataFrame(), (
            "Clinical evidence file is missing required columns: "
            + ", ".join(missing)
        )

    df = df.fillna("")

    return df, None


interactions, interactions_error, interactions_source = (
    load_interaction_data()
)

drugs, drugs_error, drugs_source = load_drug_data()

clinical_evidence, clinical_error = load_clinical_evidence()


# ============================================================
# CLINICAL EVIDENCE INDEX
# ============================================================

@st.cache_data
def build_clinical_index(df: pd.DataFrame):

    index = {}

    if df.empty:
        return index

    for _, row in df.iterrows():

        aliases = [
            (row.get("drug1_name", ""), row.get("drug2_name", "")),
            (row.get("drug1_key", ""), row.get("drug2_key", "")),
            (row.get("drug1_name", ""), row.get("drug2_key", "")),
            (row.get("drug1_key", ""), row.get("drug2_name", "")),
        ]

        for a, b in aliases:

            key = unordered_pair(a, b)

            if not all(key):
                continue

            # Preserve the first exact mapping for a pair.
            if key not in index:
                index[key] = row.to_dict()

    return index


clinical_index = build_clinical_index(clinical_evidence)


def assess_clinical_severity(drug1: str, drug2: str):

    key = unordered_pair(drug1, drug2)

    result = clinical_index.get(key)

    if result is None:

        return {
            "matched": False,
            "severity": "Not assessed",
            "source_reported_level": None,
            "assessment_status": "Not assessed",
            "evidence_source": None,
            "evidence_quality": None,
            "interaction_description": None,
            "clinical_effect": None,
            "clinical_significance": None,
            "management": None,
            "evidence_scope": None,
            "source_files": None,
            "message": (
                "No exact matching pair was found in the imported "
                "DDInter evidence. This does not mean that no "
                "interaction exists."
            ),
        }

    return {
        "matched": True,
        "severity": safe_text(
            result.get("severity"),
            "Unknown"
        ),
        "source_reported_level": safe_text(
            result.get("source_reported_level"),
            "Unknown"
        ),
        "assessment_status": safe_text(
            result.get("assessment_status")
        ),
        "evidence_source": safe_text(
            result.get("evidence_source")
        ),
        "evidence_quality": safe_text(
            result.get("evidence_quality")
        ),
        "interaction_description": safe_text(
            result.get("interaction_description"),
            ""
        ),
        "clinical_effect": safe_text(
            result.get("clinical_effect"),
            ""
        ),
        "clinical_significance": safe_text(
            result.get("clinical_significance"),
            ""
        ),
        "management": safe_text(
            result.get("management"),
            ""
        ),
        "evidence_scope": safe_text(
            result.get("evidence_scope"),
            ""
        ),
        "source_files": safe_text(
            result.get("source_files"),
            ""
        ),
        "canonical_pair_id": safe_text(
            result.get("canonical_pair_id"),
            ""
        ),
        "drug1_key": safe_text(
            result.get("drug1_key"),
            ""
        ),
        "drug2_key": safe_text(
            result.get("drug2_key"),
            ""
        ),
        "message": (
            "Severity level retrieved from DDInter 2.0. "
            "Independent clinical verification is pending."
        ),
    }


# ============================================================
# INTERACTION SEARCH HELPERS
# ============================================================

def find_pair_records(
    drug_a: str,
    drug_b: str
) -> pd.DataFrame:

    if interactions.empty:
        return pd.DataFrame()

    a = interactions["Drug_A"].astype(str).str.casefold()
    b = interactions["Drug_B"].astype(str).str.casefold()

    first = drug_a.casefold()
    second = drug_b.casefold()

    return interactions[
        ((a == first) & (b == second))
        | ((a == second) & (b == first))
    ].copy()


def get_drug_names() -> list[str]:

    names: set[str] = set()

    if not interactions.empty:
        names.update(interactions["Drug_A"].tolist())
        names.update(interactions["Drug_B"].tolist())

    if not drugs.empty:
        names.update(drugs["drug_name"].tolist())

    if not clinical_evidence.empty:
        names.update(
            clinical_evidence["drug1_name"].astype(str).tolist()
        )
        names.update(
            clinical_evidence["drug2_name"].astype(str).tolist()
        )

    return sorted(
        {
            name.strip()
            for name in names
            if name.strip()
            and name.strip().casefold() not in {"nan", "none"}
        },
        key=str.casefold,
    )


def get_drug_record(name: str) -> pd.Series | None:

    if drugs.empty:
        return None

    match = drugs[
        drugs["drug_name"].astype(str).str.casefold()
        == name.casefold()
    ]

    return None if match.empty else match.iloc[0]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def severity_color(level: str) -> str:

    colors = {
        "Minor": "#16A34A",
        "Moderate": "#D97706",
        "Major": "#DC2626",
        "Contraindicated": "#991B1B",
        "Unknown": "#64748B",
        "Not assessed": "#64748B",
    }

    return colors.get(level, "#64748B")


def display_dataset_status():

    st.markdown("#### Data sources")

    st.caption(
        f"Interaction source: `{interactions_source}`"
    )

    st.caption(
        f"Molecular source: `{drugs_source}`"
    )

    st.caption(
        f"Clinical evidence: `{CLINICAL_EVIDENCE_FILE.name}`"
    )

    if interactions_error:
        st.warning(interactions_error)

    if drugs_error:
        st.info(drugs_error)

    if clinical_error:
        st.warning(
            "Clinical evidence module unavailable: "
            + clinical_error
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧬 DDIvolve")
    st.caption("Interaction intelligence · Research workspace")

    st.divider()

    st.markdown("**Workspace status**")

    st.metric(
        "Interaction rows",
        f"{len(interactions):,}"
    )

    st.metric(
        "Drug records",
        f"{len(drugs):,}"
    )

    if not interactions.empty:

        pair_count = interactions[
            ["Drug_A", "Drug_B"]
        ].drop_duplicates().shape[0]

        st.metric(
            "Unique interaction pairs",
            f"{pair_count:,}"
        )

    st.metric(
        "Clinical evidence pairs",
        f"{len(clinical_evidence):,}"
    )

    st.divider()

    st.markdown("**Current modules**")

    st.markdown(
        "- Interaction evidence search\n"
        "- Clinical severity assessment\n"
        "- Molecular descriptor viewer\n"
        "- Dataset exploration\n"
        "- Intervention research"
    )

    st.divider()

    st.caption(
        "Research prototype only. It is not a medical device "
        "and must not be used to start, stop, or change medication."
    )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Drug safety research platform</div>
      <div class="hero-title">DDIvolve</div>
      <div class="hero-copy">
        An intelligent framework for exploring drug–drug interaction
        evidence, molecular features, clinical severity, and future
        risk-mitigation research.
      </div>
      <span class="hero-tag">
        Evidence-aware · Molecularly informed · Research-first
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW METRICS
# ============================================================

unique_drugs = len(get_drug_names())

unique_pairs = (
    interactions[
        ["Drug_A", "Drug_B"]
    ].drop_duplicates().shape[0]
    if not interactions.empty else 0
)

label_count = (
    interactions["Evidence_Label"]
    .replace("", pd.NA)
    .nunique(dropna=True)
    if not interactions.empty else 0
)

clinical_pair_count = len(clinical_evidence)

major_count = (
    int((clinical_evidence["severity"] == "Major").sum())
    if not clinical_evidence.empty else 0
)

moderate_count = (
    int((clinical_evidence["severity"] == "Moderate").sum())
    if not clinical_evidence.empty else 0
)

minor_count = (
    int((clinical_evidence["severity"] == "Minor").sum())
    if not clinical_evidence.empty else 0
)

unknown_count = (
    int((clinical_evidence["severity"] == "Unknown").sum())
    if not clinical_evidence.empty else 0
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Drug entities",
    f"{unique_drugs:,}"
)

m2.metric(
    "Interaction records",
    f"{len(interactions):,}"
)

m3.metric(
    "Unique interaction pairs",
    f"{unique_pairs:,}"
)

m4.metric(
    "Clinical evidence pairs",
    f"{clinical_pair_count:,}"
)

st.markdown("")


# ============================================================
# MAIN NAVIGATION
# ============================================================

(
    tab_overview,
    tab_checker,
    tab_severity,
    tab_molecular,
    tab_data,
    tab_research,
) = st.tabs(
    [
        "⌂ Overview",
        "⇄ Interaction checker",
        "⚕ Clinical severity",
        "⚛ Molecular explorer",
        "▤ Data explorer",
        "✧ Intervention research",
    ]
)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab_overview:

    left, right = st.columns(
        [1.25, 0.75],
        gap="large"
    )

    with left:

        st.markdown(
            '<div class="section-eyebrow">Platform overview</div>',
            unsafe_allow_html=True
        )

        st.subheader("From evidence to research insight")

        st.markdown(
            """
            DDIvolve is being developed as a modular research framework
            connecting interaction detection, molecular analysis,
            clinical evidence retrieval, and future mitigation research.
            """
        )

        with st.container(border=True):

            st.markdown("**Current workflow**")

            st.markdown(
                "1. **Find interactions** — inspect drug-pair records.\n"
                "2. **Assess severity evidence** — retrieve DDInter source-reported levels.\n"
                "3. **Inspect chemistry** — review available molecular descriptors.\n"
                "4. **Evaluate models** — integrate validated model predictions.\n"
                "5. **Explore mitigation** — organize future research pathways."
            )

    with right:

        with st.container(border=True):

            st.markdown("**Data readiness**")

            if not interactions.empty:

                st.markdown(
                    '<span class="status-pill">Interaction data loaded</span>',
                    unsafe_allow_html=True
                )

                st.write(
                    f"{len(interactions):,} interaction records available."
                )

            else:
                st.warning("Interaction data is not ready.")

            if not drugs.empty:

                st.markdown(
                    '<span class="status-pill">Molecular data loaded</span>',
                    unsafe_allow_html=True
                )

                st.write(
                    f"{len(drugs):,} molecular records available."
                )

            else:
                st.info("Molecular records are not available yet.")

            if not clinical_evidence.empty:

                st.markdown(
                    '<span class="status-pill">Clinical evidence loaded</span>',
                    unsafe_allow_html=True
                )

                st.write(
                    f"{len(clinical_evidence):,} evidence pairs available."
                )

            else:
                st.warning("Clinical evidence is not available.")

            st.caption(
                "A loaded dataset does not guarantee complete clinical "
                "coverage or independently validated predictions."
            )

    st.markdown("### Clinical evidence snapshot")

    if not clinical_evidence.empty:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Major", f"{major_count:,}")
        c2.metric("Moderate", f"{moderate_count:,}")
        c3.metric("Minor", f"{minor_count:,}")
        c4.metric("Unknown", f"{unknown_count:,}")

        st.caption(
            "These are DDInter source-reported levels among the "
            "matched pairs, not independently verified clinical outcomes."
        )

    else:
        st.info("Load the clinical evidence CSV to view this summary.")

    st.markdown("### Quick access")

    q1, q2, q3 = st.columns(3)

    with q1:
        with st.container(border=True):
            st.markdown("**⇄ Interaction checker**")
            st.caption("Find interaction records for a drug pair.")

    with q2:
        with st.container(border=True):
            st.markdown("**⚕ Clinical severity**")
            st.caption("Retrieve source-reported severity and evidence.")

    with q3:
        with st.container(border=True):
            st.markdown("**⚛ Molecular explorer**")
            st.caption("Inspect SMILES and chemical descriptors.")


# ============================================================
# TAB 2: INTERACTION CHECKER
# ============================================================

with tab_checker:

    st.markdown(
        '<div class="section-eyebrow">Interaction lookup</div>',
        unsafe_allow_html=True
    )

    st.subheader("Drug interaction checker")

    st.markdown(
        '<div class="subtle">Search the available DDIvolve interaction '
        'records and retrieve separate clinical evidence where available.</div>',
        unsafe_allow_html=True
    )

    names = get_drug_names()

    if len(names) < 2:

        st.warning(
            "At least two drug entries are needed. Check your loaded datasets."
        )

    else:

        c1, c2 = st.columns(2)

        with c1:

            drug_a = st.selectbox(
                "First drug",
                names,
                key="checker_drug_a"
            )

        with c2:

            candidates = [
                name for name in names
                if name.casefold() != drug_a.casefold()
            ]

            drug_b = st.selectbox(
                "Second drug",
                candidates,
                key="checker_drug_b"
            )

        run_check = st.button(
            "Assess drug combination",
            type="primary",
            use_container_width=True
        )

        if run_check:

            if drug_a.casefold() == drug_b.casefold():

                st.warning("Select two different drugs.")

            else:

                interaction_matches = find_pair_records(
                    drug_a,
                    drug_b
                )

                clinical_result = assess_clinical_severity(
                    drug_a,
                    drug_b
                )

                st.divider()

                st.markdown("### 1. Interaction dataset result")

                if not interaction_matches.empty:

                    st.success(
                        f"Found {len(interaction_matches):,} "
                        "matching interaction record(s)."
                    )

                    view_cols = [
                        "Drug_A",
                        "Drug_B",
                        "Evidence_Label"
                    ]

                    if "Negative_Sample_ID" in interaction_matches.columns:
                        view_cols.append("Negative_Sample_ID")

                    st.dataframe(
                        interaction_matches[view_cols],
                        hide_index=True,
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "No matching interaction record was found "
                        "in the loaded interaction dataset."
                    )

                st.divider()

                st.markdown("### 2. Clinical severity assessment")

                if clinical_result["matched"]:

                    severity = clinical_result["severity"]

                    color = severity_color(severity)

                    st.markdown(
                        f"""
                        <div class="severity-card">
                            <div class="section-eyebrow">
                                DDINTER SOURCE-REPORTED LEVEL
                            </div>
                            <div class="severity-value"
                                 style="color:{color}">
                                {severity}
                            </div>
                            <div>
                                {drug_a} + {drug_b}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.success(
                        "An exact matching pair was found in the "
                        "imported DDInter evidence."
                    )

                    st.markdown("#### Interaction description")

                    description = clinical_result[
                        "interaction_description"
                    ]

                    if description:
                        st.info(description)
                    else:
                        st.write("No description available.")

                    st.markdown("#### Evidence details")

                    e1, e2 = st.columns(2)

                    with e1:

                        st.write(
                            "**Evidence source:**",
                            clinical_result["evidence_source"]
                        )

                        st.write(
                            "**Assessment status:**",
                            clinical_result["assessment_status"]
                        )

                        st.write(
                            "**Evidence quality:**",
                            clinical_result["evidence_quality"]
                        )

                    with e2:

                        st.write(
                            "**Clinical effect:**",
                            clinical_result["clinical_effect"]
                            or "Not provided in imported evidence"
                        )

                        st.write(
                            "**Clinical significance:**",
                            clinical_result["clinical_significance"]
                            or "Not independently assessed"
                        )

                        st.write(
                            "**Management:**",
                            clinical_result["management"]
                            or "Not provided in imported evidence"
                        )

                    if severity == "Unknown":

                        st.warning(
                            "DDInter reports this pair's level as "
                            "Unknown. Do not interpret it as Minor "
                            "or as evidence of no interaction."
                        )

                    else:

                        st.warning(
                            "This is a source-reported DDInter 2.0 "
                            "severity level. Independent clinical "
                            "verification is pending."
                        )

                    with st.expander("Technical evidence details"):

                        st.json({
                            "canonical_pair_id": clinical_result.get(
                                "canonical_pair_id"
                            ),
                            "drug1_key": clinical_result.get(
                                "drug1_key"
                            ),
                            "drug2_key": clinical_result.get(
                                "drug2_key"
                            ),
                            "source_reported_level": severity,
                            "source_files": clinical_result.get(
                                "source_files"
                            ),
                            "evidence_scope": clinical_result.get(
                                "evidence_scope"
                            ),
                        })

                else:

                    st.markdown(
                        """
                        <div class="severity-card">
                            <div class="section-eyebrow">
                                CLINICAL EVIDENCE STATUS
                            </div>
                            <div class="severity-value"
                                 style="color:#64748B">
                                Not assessed
                            </div>
                            <div>
                                No exact matching DDInter pair found.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.warning(
                        clinical_result["message"]
                    )

                st.divider()

                st.caption(
                    "Interaction dataset matching and DDInter clinical "
                    "evidence matching are separate operations. Neither "
                    "result is a personalized medical recommendation."
                )


# ============================================================
# TAB 3: CLINICAL SEVERITY DASHBOARD
# ============================================================

with tab_severity:

    st.markdown(
        '<div class="section-eyebrow">Clinical evidence module</div>',
        unsafe_allow_html=True
    )

    st.subheader("Clinical severity and evidence overview")

    st.markdown(
        """
        This module connects DDIvolve interaction pairs to the
        source-reported severity levels available in DDInter 2.0.
        """
    )

    if clinical_evidence.empty:

        st.error(
            "Clinical evidence data is unavailable. "
            "Check severity_module/data/processed/clinical_evidence.csv."
        )

    else:

        total = len(clinical_evidence)

        counts = (
            clinical_evidence["severity"]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        major = int(counts.get("Major", 0))
        moderate = int(counts.get("Moderate", 0))
        minor = int(counts.get("Minor", 0))
        unknown = int(counts.get("Unknown", 0))

        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric("Matched pairs", f"{total:,}")
        k2.metric("Major", f"{major:,}")
        k3.metric("Moderate", f"{moderate:,}")
        k4.metric("Minor", f"{minor:,}")
        k5.metric("Unknown", f"{unknown:,}")

        st.markdown("### Severity distribution")

        chart_data = pd.DataFrame({
            "Severity": [
                "Major",
                "Moderate",
                "Minor",
                "Unknown"
            ],
            "Records": [
                major,
                moderate,
                minor,
                unknown
            ]
        }).set_index("Severity")

        st.bar_chart(chart_data)

        st.caption(
            "Counts represent the DDInter source-reported levels "
            "in the matched evidence dataset."
        )

        st.divider()

        st.markdown("### Search clinical evidence")

        search1, search2 = st.columns(2)

        with search1:

            search_drug = st.text_input(
                "Search by drug name",
                placeholder="Enter a drug name..."
            )

        with search2:

            severity_filter = st.multiselect(
                "Filter by source-reported severity",
                options=[
                    "Major",
                    "Moderate",
                    "Minor",
                    "Unknown"
                ],
                default=[]
            )

        filtered = clinical_evidence.copy()

        if search_drug.strip():

            term = normalize(search_drug)

            mask = (
                filtered["drug1_name"].astype(str).map(normalize).str.contains(
                    term, regex=False
                )
                | filtered["drug2_name"].astype(str).map(normalize).str.contains(
                    term, regex=False
                )
            )

            filtered = filtered[mask]

        if severity_filter:

            filtered = filtered[
                filtered["severity"].isin(severity_filter)
            ]

        st.write(
            f"Showing **{len(filtered):,}** of **{total:,}** evidence records."
        )

        display_columns = [
            "drug1_name",
            "drug2_name",
            "severity",
            "interaction_families",
            "evidence_source",
            "assessment_status"
        ]

        available_columns = [
            col for col in display_columns
            if col in filtered.columns
        ]

        st.dataframe(
            filtered[available_columns].head(500),
            use_container_width=True,
            hide_index=True
        )

        if len(filtered) > 500:

            st.caption(
                "The preview shows the first 500 records. "
                "The download contains all filtered records."
            )

        download_data = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download clinical evidence CSV",
            data=download_data,
            file_name="ddivolve_clinical_evidence.csv",
            mime="text/csv"
        )

        with st.expander("Clinical evidence limitations"):

            st.markdown(
                """
                - The current evidence source is DDInter 2.0.
                - Pair matching uses exact normalized names and keys.
                - Unmatched pairs are classified as Not assessed.
                - Unknown remains a separate source-reported category.
                - Clinical effects and management are not invented when
                  absent from the imported data.
                - Independent clinical verification is still pending.
                """
            )


# ============================================================
# TAB 4: MOLECULAR EXPLORER
# ============================================================

with tab_molecular:

    st.markdown(
        '<div class="section-eyebrow">Chemical profile</div>',
        unsafe_allow_html=True
    )

    st.subheader("Molecular explorer")

    st.markdown(
        '<div class="subtle">Browse SMILES strings and descriptors supplied '
        'by the current molecular dataset.</div>',
        unsafe_allow_html=True
    )

    if drugs.empty:

        st.info(
            "Molecular data is not available yet. Add "
            "data/drug_smiles.csv or data/drugs.csv."
        )

        display_dataset_status()

    else:

        drug_names = sorted(
            drugs["drug_name"].unique(),
            key=str.casefold
        )

        selected = st.selectbox(
            "Choose a drug",
            drug_names,
            key="molecular_selected_drug"
        )

        record = get_drug_record(selected)

        if record is not None:

            left, right = st.columns(
                [0.85, 1.15],
                gap="large"
            )

            with left:

                with st.container(border=True):

                    st.markdown("**Compound identity**")

                    st.markdown(f"### {safe_text(record.get('drug_name'))}")

                    identity_fields = [
                        ("drug_id", "Source drug ID"),
                        ("pubchem_cid", "PubChem CID"),
                        ("molecular_formula", "Molecular formula"),
                        ("inchi_key", "InChIKey"),
                        ("lookup_status", "Lookup status"),
                    ]

                    for field, label in identity_fields:

                        if field in record.index:

                            st.write(
                                f"**{label}:** "
                                f"{safe_text(record.get(field))}"
                            )

            with right:

                with st.container(border=True):

                    st.markdown("**SMILES representation**")

                    smiles = safe_text(
                        record.get("canonical_smiles"),
                        ""
                    )

                    if not smiles:
                        smiles = safe_text(
                            record.get("isomeric_smiles"),
                            ""
                        )

                    if not smiles:
                        smiles = safe_text(
                            record.get("smiles"),
                            ""
                        )

                    if smiles:
                        st.code(smiles, language=None)
                    else:
                        st.info("No SMILES string is available.")

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
                        (key, label, decimals)
                        for key, label, decimals in descriptor_map
                        if key in record.index
                        and safe_text(record.get(key), "") != ""
                    ]

                    if available:

                        for start in range(0, len(available), 3):

                            cols = st.columns(3)

                            for col, (key, label, decimals) in zip(
                                cols,
                                available[start:start + 3]
                            ):

                                col.metric(
                                    label,
                                    safe_number(
                                        record.get(key),
                                        decimals
                                    )
                                )

                    else:

                        st.info(
                            "No calculated descriptors found."
                        )

            st.caption(
                "Molecular descriptors do not independently establish "
                "interaction risk, clinical severity, efficacy, or safety."
            )


# ============================================================
# TAB 5: DATA EXPLORER
# ============================================================

with tab_data:

    st.markdown(
        '<div class="section-eyebrow">Dataset workspace</div>',
        unsafe_allow_html=True
    )

    st.subheader("Explore interaction records")

    if interactions.empty:

        st.warning(
            "No interaction records are available to explore."
        )

        display_dataset_status()

    else:

        f1, f2 = st.columns([1.1, 0.9])

        with f1:

            search_text = st.text_input(
                "Search drug names or labels",
                placeholder="Type a drug name or label...",
                key="data_search"
            )

        with f2:

            labels = sorted(
                interactions["Evidence_Label"]
                .dropna()
                .astype(str)
                .unique()
                .tolist(),
                key=str.casefold
            )

            selected_labels = st.multiselect(
                "Filter labels",
                options=labels,
                default=[]
            )

        filtered = interactions.copy()

        if search_text.strip():

            term = search_text.strip().casefold()

            mask = (
                filtered["Drug_A"].str.casefold().str.contains(
                    term, regex=False
                )
                | filtered["Drug_B"].str.casefold().str.contains(
                    term, regex=False
                )
                | filtered["Evidence_Label"].str.casefold().str.contains(
                    term, regex=False
                )
            )

            filtered = filtered[mask]

        if selected_labels:

            filtered = filtered[
                filtered["Evidence_Label"].isin(selected_labels)
            ]

        st.write(
            f"Showing **{len(filtered):,}** of "
            f"**{len(interactions):,}** rows."
        )

        st.dataframe(
            filtered.head(500),
            hide_index=True,
            use_container_width=True
        )

        if len(filtered) > 500:

            st.caption(
                "Preview limited to 500 rows. Download includes all filtered rows."
            )

        csv_bytes = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download filtered interaction CSV",
            data=csv_bytes,
            file_name="ddivolve_filtered_interactions.csv",
            mime="text/csv"
        )

        with st.expander("Dataset profile"):

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Filtered rows",
                f"{len(filtered):,}"
            )

            p2.metric(
                "Unique pairs",
                f"{filtered[['Drug_A', 'Drug_B']].drop_duplicates().shape[0]:,}"
            )

            p3.metric(
                "Distinct labels",
                f"{filtered['Evidence_Label'].nunique():,}"
            )

            if not filtered.empty:

                label_counts = (
                    filtered["Evidence_Label"]
                    .value_counts()
                    .head(15)
                    .rename_axis("Label")
                    .reset_index(name="Rows")
                )

                st.markdown("**Most frequent labels in this view**")

                st.bar_chart(
                    label_counts.set_index("Label")
                )


# ============================================================
# TAB 6: INTERVENTION RESEARCH
# ============================================================

with tab_research:

    st.markdown(
        '<div class="section-eyebrow">Future-facing research</div>',
        unsafe_allow_html=True
    )

    st.subheader("Intervention research pathways")

    st.markdown(
        """
        This module outlines future research workflows.
        It does not prescribe treatments, select replacement medicines,
        or automatically redesign molecules.
        """
    )

    pathway = st.radio(
        "Choose a pathway",
        [
            "Clinical evidence review",
            "Molecular research and model evaluation",
        ],
        horizontal=True,
        key="research_pathway"
    )

    st.markdown("")

    if pathway == "Clinical evidence review":

        a, b = st.columns(2, gap="large")

        with a:

            with st.container(border=True):

                st.markdown("### 01 · Gather evidence")

                st.markdown(
                    "- Confirm exact medicine identities and formulations.\n"
                    "- Review documented interaction evidence and sources.\n"
                    "- Identify uncertainty and missing clinical context."
                )

        with b:

            with st.container(border=True):

                st.markdown("### 02 · Professional assessment")

                st.markdown(
                    "- Have a qualified clinician or pharmacist assess relevance.\n"
                    "- Consider indication, dose, history, and monitoring needs.\n"
                    "- Do not recommend an alternative without qualified review."
                )

        st.warning(
            "DDIvolve is a research prototype, not a clinical decision-support "
            "system. Do not start, stop, or change medication based on this app."
        )

    else:

        a, b = st.columns(2, gap="large")

        with a:

            with st.container(border=True):

                st.markdown("### 01 · Molecular pipeline")

                st.markdown(
                    "- Validate SMILES and compound identity.\n"
                    "- Generate RDKit descriptors and fingerprints.\n"
                    "- Preserve source IDs and missing-structure information."
                )

        with b:

            with st.container(border=True):

                st.markdown("### 02 · Model validation")

                st.markdown(
                    "- Verify positive and negative pair construction.\n"
                    "- Prevent pair leakage during evaluation.\n"
                    "- Report precision, recall, F1, PR-AUC, and calibration.\n"
                    "- Track model versions and limitations."
                )

        st.info(
            "A model score or molecular descriptor alone does not prove "
            "that a molecular modification reduces interaction risk. "
            "Experimental and expert validation are required."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_left, footer_right = st.columns([1, 1])

with footer_left:

    st.caption(
        "DDIvolve · Drug–Drug Interaction Detection and Mitigation"
    )

with footer_right:

    st.caption(
        "Student research prototype · Evidence may be incomplete"
    )
