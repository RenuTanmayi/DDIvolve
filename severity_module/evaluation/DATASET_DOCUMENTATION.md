# DDIvolve — Severity Dataset Documentation

## 1. Purpose

This document describes the shared datasets used to develop
the DDIvolve Clinical Severity and Evidence Assessment Module.

## 2. Source files

| Dataset | Records | Purpose |
|---|---:|---|
| clean_drugs.csv | 1,701 | Drug identities and molecular information |
| interaction_template_catalog.csv | 86 | Interaction template and family mappings |
| validated_interaction_labels.csv | 190,238 | Validated interaction descriptions and labels |
| validated_pair_summary.csv | 189,832 | Pair-level interaction statistics |

## 3. Dataset statistics

- Unique drug records: 1,701
- Unique drug pairs in labels: 189,832
- Unique pairs in summary: 189,832
- Interaction families: 9
- Interaction descriptions: 190,238

## 4. Pair-level structure

- Pairs with multiple descriptions: 406
- Pairs with multiple families: 355
- Pairs with multiple affected-drug records: 135

## 5. Data consistency

- Label pairs missing from summary: 0
- Summary pairs missing from labels: 0
- Interaction count mismatches: 0
- Family count mismatches: 0
- Affected target count mismatches: 0

## 6. Clinical evidence limitations

The shared interaction labels describe interaction mechanisms,
families, directions and affected drugs.

They do not establish clinical severity.

An interaction family must not be treated as a Minor, Moderate,
Major or Contraindicated severity label.

Clinical severity requires additional evidence from reliable
clinical sources.

Absence of evidence must not be interpreted as evidence of
absence of an interaction.

## 7. Severity assessment categories

- Minor
- Moderate
- Major
- Contraindicated
- Unknown

Evidence quality must be recorded independently from severity.

## 8. Next development stage

Collect pair-specific clinical evidence, preserve source references,
document management information, and independently validate
severity assessments before building the severity engine.

## 9. Important limitations

This audit assesses only the four shared datasets.
It does not establish that no clinical evidence exists elsewhere.
All clinical severity assessments remain pending until evidence
collection and validation are completed.
