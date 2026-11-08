
from severity_module.src.assess_severity import assess_severity


def classify_research_pathway(description, interaction_family=None):
    """Classify an interaction description into a research pathway."""

    text = str(description or "").casefold()
    family = str(interaction_family or "").casefold()

    if any(term in text for term in ["serum concentration", "plasma concentration"]):
        pathway = "Pharmacokinetic research"
        focus = "Investigate changes in drug concentration and exposure."

    elif any(term in text for term in ["metabolism", "metabolized"]):
        pathway = "Drug metabolism research"
        focus = "Investigate possible changes in metabolic activity."

    elif any(term in text for term in ["adverse effect", "side effect", "toxicity"]):
        pathway = "Adverse-effect research"
        focus = "Investigate the reported adverse-effect interaction."

    elif any(term in text for term in ["therapeutic efficacy", "therapeutic effect"]):
        pathway = "Therapeutic efficacy research"
        focus = "Investigate possible changes in therapeutic effect."

    elif any(term in text for term in ["excretion", "elimination"]):
        pathway = "Drug disposition research"
        focus = "Investigate changes in drug elimination."

    elif any(term in text for term in ["absorption", "bioavailability"]):
        pathway = "Drug absorption research"
        focus = "Investigate changes in drug absorption or availability."

    else:
        pathway = "General evidence review"
        focus = "Review the interaction evidence and identify research gaps."

    return {
        "pathway": pathway,
        "research_focus": focus,
        "interaction_family": interaction_family,
    }


def analyze_intervention_research(drug1, drug2):
    """
    Build an evidence-based intervention research profile.

    This function does not prescribe medicines or recommend
    treatment changes.
    """

    evidence = assess_severity(drug1, drug2)

    if not evidence.get("matched"):
        return {
            "matched": False,
            "drug1": drug1,
            "drug2": drug2,
            "severity": "Not assessed",
            "research_pathway": "Insufficient pair-specific evidence",
            "research_focus": (
                "No exact matching DDInter record was found. "
                "Further evidence review is required."
            ),
            "interaction_description": None,
            "management": None,
            "management_status": (
                "No source-provided management recommendation available."
            ),
            "evidence_source": None,
            "evidence_quality": None,
            "assessment_status": "Not assessed",
        }

    description = evidence.get("interaction_description", "")

    pathway = classify_research_pathway(description)

    management = evidence.get("management")

    return {
        "matched": True,
        "drug1": drug1,
        "drug2": drug2,
        "severity": evidence.get("source_reported_level", "Unknown"),
        "research_pathway": pathway["pathway"],
        "research_focus": pathway["research_focus"],
        "interaction_description": description,
        "clinical_effect": evidence.get("clinical_effect"),
        "clinical_significance": evidence.get("clinical_significance"),
        "management": management,
        "management_status": (
            "Source-provided management information available."
            if management
            else "No source-provided management recommendation available."
        ),
        "evidence_source": evidence.get("evidence_source"),
        "evidence_quality": evidence.get("evidence_quality"),
        "assessment_status": evidence.get("assessment_status"),
        "evidence_scope": evidence.get("evidence_scope"),
        "notice": (
            "Research classification only. This output is not a "
            "treatment recommendation or clinical decision."
        ),
    }
