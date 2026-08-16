"""The clinical state: absent evidence, reference ranges, evidence grading, case quality.

The property under test throughout is that a gap in the record stays visible as a gap.
"""
from src.agents.clinical_state import build_clinical_state, render_state
from .helpers import (CASE_QUALITY, MISSING_NOT_NORMAL, REFERENCE_RANGE, SCOPE,
                      prop, bundle, heart_report, lung_report)
from src.agents import schema as S


# ------------------------------------------------------------------ absent is not normal
@prop(MISSING_NOT_NORMAL)
def test_unmeasured_lab_is_listed_as_missing():
    st = build_clinical_state(bundle(ultrasound={"lung": lung_report()}), labs={})
    assert "troponin" in st["missing"]["labs"]
    assert "troponin" not in st["labs"]


@prop(MISSING_NOT_NORMAL)
def test_unmeasured_lab_never_appears_as_normal_in_the_rendered_state():
    """The rendered state is what the language model actually reads. If an absent test can be
    read as normal there, every guarantee upstream is decorative."""
    st = build_clinical_state(bundle(ultrasound={"lung": lung_report()}), labs={})
    text = render_state(st)
    assert "NOT MEASURED" in text
    missing = st["missing"]["labs"]
    assert missing, "expected some labs to be absent in this case"
    for name in missing:
        for phrase in (f"{name} normal", f"{name}: normal", f"{name} within"):
            assert phrase.lower() not in text.lower(), \
                f"absent lab {name!r} rendered as normal"


@prop(MISSING_NOT_NORMAL)
def test_a_measured_lab_is_not_reported_as_missing():
    st = build_clinical_state(bundle(ultrasound={"lung": lung_report()}),
                              labs={"troponin": 340.0})
    assert "troponin" not in st["missing"]["labs"]
    assert "troponin" in st["labs"]


@prop(MISSING_NOT_NORMAL)
def test_organ_not_assessed_is_distinct_from_organ_negative():
    """'No gallbladder scan was performed' and 'the gallbladder scan was negative' are
    different clinical statements."""
    rep = S.make_report("gallbladder", [], status="not_supported")
    st = build_clinical_state(bundle(ultrasound={"gallbladder": rep}))
    assert "gallbladder" in st["imaging"]["organs_not_assessed"]
    assert not any(f["organ"] == "gallbladder" for f in st["imaging"]["findings"])


# ------------------------------------------------------------------ reference ranges
@prop(REFERENCE_RANGE)
def test_high_troponin_is_flagged_high():
    st = build_clinical_state(bundle(ultrasound={"lung": lung_report()}),
                              labs={"troponin": 340.0})
    entry = st["labs"]["troponin"]
    assert entry.get("flag") == "high", entry


@prop(REFERENCE_RANGE)
def test_normal_troponin_is_not_flagged_high():
    st = build_clinical_state(bundle(ultrasound={"lung": lung_report()}),
                              labs={"troponin": 5.0})
    assert st["labs"]["troponin"].get("flag") != "high"


@prop(REFERENCE_RANGE)
def test_low_oxygen_saturation_is_flagged_under_the_triage_feature_name():
    """The Triage Agent emits 'o2sat'; the state stores it canonically as 'spo2'. Input
    accepts the alias, output is canonical."""
    tri = S.make_triage("high", 0.8, features={"o2sat": 85})
    st = build_clinical_state(bundle(triage=tri, ultrasound={"lung": lung_report()}))
    assert "spo2" in st["vitals"], st["vitals"]
    assert st["vitals"]["spo2"]["flag"] == "low", st["vitals"]["spo2"]
    assert "spo2" not in st["missing"]["vitals"]


@prop(REFERENCE_RANGE)
def test_fahrenheit_temperature_is_converted_not_compared_directly():
    """98.6 F is normothermic. Compared against a 36--38 C range without conversion it reads
    as a high fever."""
    tri = S.make_triage("low", 0.8, features={"temp_f": 98.6})
    st = build_clinical_state(bundle(triage=tri, ultrasound={"lung": lung_report()}))
    assert "temp" in st["vitals"], st["vitals"]
    assert st["vitals"]["temp"]["flag"] == "normal", st["vitals"]["temp"]


@prop(REFERENCE_RANGE)
def test_all_triage_feature_names_are_recognised():
    """A regression guard for the whole alias table: every vital the triage model actually
    emits must be interpreted, not silently recorded as absent."""
    tri = S.make_triage("high", 0.8,
                        features={"o2sat": 91, "pulse": 118, "bpsys": 88, "respr": 24})
    st = build_clinical_state(bundle(triage=tri, ultrasound={"lung": lung_report()}))
    for name in ("spo2", "hr", "sbp", "rr"):
        assert name in st["vitals"], f"{name} not recognised: {st['vitals']}"
        assert name not in st["missing"]["vitals"]


# ------------------------------------------------------------------ evidence grading
@prop(CASE_QUALITY)
def test_uncalibrated_confidence_is_graded_experimental():
    """An uncalibrated score is not a probability, and must not be presented as one."""
    st = build_clinical_state(
        bundle(ultrasound={"lung": lung_report(calibrated=False)}))
    f = next(f for f in st["imaging"]["findings"] if f["label"] == "b_lines")
    assert f["evidence"] == "experimental", f


@prop(CASE_QUALITY)
def test_thin_evidence_declared_in_reliability_is_honoured():
    """A module can flag thin evidence per finding OR as a list in reliability. Checking only
    the first silently drops the flag for modules that use the second."""
    rep = lung_report(b_lines=0.7, unreliable=["b_lines"])
    st = build_clinical_state(bundle(ultrasound={"lung": rep}))
    f = next(f for f in st["imaging"]["findings"] if f["label"] == "b_lines")
    assert f["low_evidence"] is True
    assert f["evidence"] == "limited", f


@prop(CASE_QUALITY)
def test_low_confidence_is_graded_limited_even_when_calibrated():
    rep = lung_report(b_lines=0.20, unreliable=[])
    st = build_clinical_state(bundle(ultrasound={"lung": rep}))
    f = next(f for f in st["imaging"]["findings"] if f["label"] == "b_lines")
    assert f["evidence"] == "limited", f


@prop(CASE_QUALITY)
def test_a_sparse_case_is_graded_poor():
    """No triage, no labs, an uncalibrated low-confidence finding: less information must
    produce more caution, not a confident answer."""
    st = build_clinical_state(
        bundle(triage=None, ultrasound={"lung": lung_report(b_lines=0.3,
                                                            calibrated=False)}),
        labs={})
    assert st["case_quality"]["grade"] == "POOR", st["case_quality"]


@prop(CASE_QUALITY)
def test_case_quality_names_its_reasons():
    """A grade without reasons is not actionable."""
    st = build_clinical_state(bundle(ultrasound={"lung": lung_report()}), labs={})
    q = st["case_quality"]
    assert q["reasons"], "a non-GOOD grade must name why"


# ------------------------------------------------------------------ scope propagation
@prop(SCOPE)
def test_module_without_healthy_class_cannot_exclude_disease():
    st = build_clinical_state(bundle(ultrasound={"lung": lung_report()}))
    assert any("healthy class" in s for s in st["imaging"]["out_of_scope"]), \
        st["imaging"]["out_of_scope"]


@prop(SCOPE)
def test_unmodelled_finding_is_declared_inexcludable():
    """Pneumothorax has no positive examples in the data, so the module cannot exclude it.
    That limit has to reach the reasoning layer."""
    st = build_clinical_state(bundle(ultrasound={"lung": lung_report()}))
    assert any("pneumothorax" in s for s in st["imaging"]["out_of_scope"]), \
        st["imaging"]["out_of_scope"]


@prop(SCOPE)
def test_state_validates_structurally():
    from src.agents.clinical_state import validate_state
    st = build_clinical_state(
        bundle(triage=S.make_triage("high", 0.8, features={"o2sat": 91}),
               ultrasound={"lung": lung_report(), "heart": heart_report()}),
        labs={"troponin": 340.0})
    assert validate_state(st) == []
