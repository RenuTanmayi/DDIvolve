
import pandas as pd
from severity_module.src.assess_severity import assess_severity

evidence = pd.read_csv(
    "severity_module/data/processed/clinical_evidence.csv",
    dtype=str
).fillna("")

# Test an existing pair
sample = evidence.iloc[0]

result = assess_severity(
    sample["drug1_key"],
    sample["drug2_key"]
)

print("EXISTING PAIR TEST")
print(result)

# Test reversed drug order
reverse_result = assess_severity(
    sample["drug2_key"],
    sample["drug1_key"]
)

print("\nREVERSED ORDER TEST")
print("Matched:", reverse_result["matched"])
print("Severity:", reverse_result["severity"])

# Test an unmatched pair
unmatched_result = assess_severity(
    "nonexistent_drug_1",
    "nonexistent_drug_2"
)

print("\nUNMATCHED PAIR TEST")
print(unmatched_result)

assert result["matched"] is True
assert reverse_result["matched"] is True
assert result["severity"] == reverse_result["severity"]
assert unmatched_result["matched"] is False

print("\nALL TESTS PASSED")
