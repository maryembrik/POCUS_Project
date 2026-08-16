"""Schema validation: malformed agent output must not reach the reasoning layer.

An LLM given a malformed finding has no way to tell it is malformed, and will rationalise
it. Everything rejected here is something the reasoning layer would otherwise have to trust.
"""
import pytest

from src.agents import schema as S
from .helpers import SCHEMA_REJECTION, SCOPE, prop, heart_report, lung_report


# ------------------------------------------------------------------ construction guards
@prop(SCHEMA_REJECTION)
def test_unknown_organ_is_refused_at_construction():
    with pytest.raises(ValueError):
        S.make_report("brain", [S.make_finding("mass", 0.9)])


@prop(SCHEMA_REJECTION)
def test_confidence_above_one_is_refused():
    with pytest.raises(ValueError):
        S.make_finding("b_lines", 1.4)


@prop(SCHEMA_REJECTION)
def test_negative_confidence_is_refused():
    with pytest.raises(ValueError):
        S.make_finding("b_lines", -0.1)


@prop(SCHEMA_REJECTION)
def test_unknown_urgency_is_refused():
    with pytest.raises(ValueError):
        S.make_triage("critical", 0.9)


# ------------------------------------------------------------------ validation guards
@prop(SCHEMA_REJECTION)
def test_ok_status_with_no_findings_is_rejected():
    rep = S.make_report("lung", [], status="ok")
    assert S.validate_report(rep), "an 'ok' report with no findings must not validate"


@prop(SCHEMA_REJECTION)
def test_not_supported_with_findings_is_rejected():
    rep = S.make_report("vascular", [], status="not_supported")
    rep["findings"] = [S.make_finding("dvt", 0.8)]
    assert S.validate_report(rep)


@prop(SCHEMA_REJECTION)
def test_confidence_exceeding_declared_ceiling_is_rejected():
    """A module may declare a ceiling its calibrator cannot exceed. Emitting above it means
    either the calibrator was not applied or the ceiling is stale; both make the number a
    misrepresentation."""
    rep = heart_report(confidence=0.80, ceiling=0.74)
    errs = S.validate_report(rep)
    assert any("ceiling" in e for e in errs), errs


@prop(SCHEMA_REJECTION)
def test_confidence_at_the_ceiling_is_accepted():
    """The boundary must not be off by one: exactly at the ceiling is legitimate."""
    rep = heart_report(confidence=0.74, ceiling=0.74)
    assert S.validate_report(rep) == []


@prop(SCHEMA_REJECTION)
def test_finding_missing_confidence_is_rejected():
    rep = lung_report()
    rep["findings"] = [{"label": "b_lines"}]
    assert S.validate_report(rep)


@prop(SCHEMA_REJECTION)
def test_findings_must_be_a_list_not_a_single_finding():
    """Lung is genuinely multi-label; a schema accepting a bare dict would let a module
    silently report only its top finding."""
    rep = lung_report()
    rep["findings"] = S.make_finding("b_lines", 0.8)
    assert S.validate_report(rep)


@prop(SCHEMA_REJECTION)
def test_a_well_formed_report_validates():
    assert S.validate_report(lung_report()) == []


# ------------------------------------------------------------------ scope propagation
@prop(SCOPE)
def test_unsupported_organ_uses_the_same_shape_as_a_real_report():
    """An unsupported organ must be answerable in the ordinary format, so the reasoning
    layer never needs a special case -- and so 'not built' is distinguishable from 'looked
    and found nothing'."""
    rep = S.make_report("vascular", [], status="not_supported")
    assert S.validate_report(rep) == []
    assert rep["status"] == "not_supported"
    assert rep["findings"] == []


@prop(SCOPE)
def test_quality_warnings_fire_on_absence_not_presence():
    """lv_detected=True is the healthy state. An earlier version surfaced it as a warning,
    which inverted the meaning of the whole quality block."""
    assert S.quality_warnings({"lv_detected": True}) == []
    assert S.quality_warnings({"lv_detected": False})
