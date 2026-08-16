"""Shared builders and the property tag used by the safety benchmark.

Every test is tagged with the safety property it exercises, so the benchmark can report
results per property rather than as one undifferentiated pass count. A property with three
tests passing tells you something; "47 tests passed" does not.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import schema as S  # noqa: E402

# --------------------------------------------------------------------------------------
# Safety properties. These are the rows of the benchmark table.
# --------------------------------------------------------------------------------------
MISSING_NOT_NORMAL = "Absent is not normal"
REFERENCE_RANGE = "Reference-range detection"
CONFLICT = "Conflict detection"
SCHEMA_REJECTION = "Malformed output rejection"
HALLUCINATION = "Hallucination rejection"
ESCALATION = "Escalation policy"
CASE_QUALITY = "Case-quality grading"
SCOPE = "Model-scope propagation"


def prop(name: str):
    """Tag a test with the safety property it exercises."""
    def deco(fn):
        fn.safety_property = name
        return fn
    return deco


# --------------------------------------------------------------------------------------
# Builders -- deliberately thin wrappers so a schema change breaks the tests loudly
# --------------------------------------------------------------------------------------
def lung_report(*, b_lines: float | None = 0.81, calibrated: bool = True,
                unreliable: list[str] | None = None,
                modelled: list[str] | None = None,
                status: str = "ok") -> dict:
    findings = [S.make_finding("b_lines", b_lines)] if b_lines is not None else []
    return S.make_report(
        "lung",
        findings,
        status=status,
        reliability={
            "confidence_calibrated": calibrated,
            "unreliable_findings": unreliable if unreliable is not None else ["pleural_effusion"],
            "has_normal_class": False,
            "modelled_findings": modelled if modelled is not None else
            ["b_lines", "consolidation", "pleural_effusion", "pleural_thickening"],
        },
        model="efficientnet_b0",
    )


def heart_report(*, label: str = "severe_dysfunction", confidence: float = 0.71,
                 calibrated: bool = True, group: str | None = None,
                 ceiling: float | None = None) -> dict:
    rel = {"confidence_calibrated": calibrated, "has_normal_class": True}
    if ceiling is not None:
        rel["confidence_ceiling"] = ceiling
    return S.make_report(
        "heart",
        [S.make_finding(label, confidence, group=group)],
        reliability=rel,
        measurements={"ejection_fraction": 28.0},
        model="unet_effb0",
    )


def bundle(*, triage: dict | None = None, ultrasound: dict | None = None,
           clinical: dict | None = None, encounter_id: str = "TEST-001") -> dict:
    return {
        "encounter_id": encounter_id,
        "triage": triage,
        "ultrasound": ultrasound or {},
        "clinical": clinical or {"age": 71, "sex": "F"},
    }


def llm_output(supporting: list[str], *, diagnosis: str = "Pulmonary oedema",
               likelihood: str = "moderate",
               contradicting: list[str] | None = None,
               missing_information: list[str] | None = None) -> dict:
    return {
        "differential": [{
            "diagnosis": diagnosis,
            "likelihood": likelihood,
            "supporting": supporting,
            "contradicting": contradicting or [],
        }],
        # Overridable: listing a test that WAS measured is now an error, so a fixture whose
        # case resulted troponin has to name a genuinely absent test instead.
        "missing_information": missing_information or ["troponin"],
        "uncertainty": "limited evidence",
        "recommended_next_step": "obtain troponin",
    }
