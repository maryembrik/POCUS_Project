"""The Ultrasound Agent, tested by importing it rather than by running a notebook.

This file exists to close a gap the report named as a limitation: every other test builds a
module report by hand with `make_report`, so the suite proved the contract was *enforced* and
could not prove any perception module *honoured* it. The per-organ code lived in a notebook and
nothing could import it.

These tests run real inference on the checkpoints in the repository, on CPU, in about a second
per image after the model is cached. They are skipped rather than failed where the weights are
absent, because a checkout without them is a legitimate state and a red suite for a missing
4.9 GB artefact teaches nobody anything.
"""
import numpy as np

from src.agents import schema as S
from src.agents.ultrasound.agent import (
    LUNG_FINDINGS, available, predict_lung, ultrasound_agent)
from .helpers import MISSING_NOT_NORMAL, SCHEMA_REJECTION, prop

PERCEPTION = "Perception contract"

_HAVE_LUNG = available()["lung"]


def _image(seed: int = 0, shape=(300, 400)):
    return (np.random.RandomState(seed).rand(*shape) * 255).astype("uint8")


class _thresholds:
    """Force every operating point to a fixed value for the duration of a block.

    Patching the module's DEFAULT_THRESHOLD is not enough: it is only consulted for a finding
    the tuned results do not name, and they name all four.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def __enter__(self):
        import src.agents.ultrasound.agent as A
        self.A, self.saved = A, A._tuned
        A._tuned = lambda fold: {"thresholds": {n: self.value for n in LUNG_FINDINGS},
                                 "unreliable": [], "auroc": {}, "source": "test"}
        return self

    def __exit__(self, *exc):
        self.A._tuned = self.saved
        return False


@prop(PERCEPTION)
def test_the_agent_reports_what_it_can_actually_run():
    a = available()
    assert set(a) == {"lung", "heart", "gallbladder"}
    assert all(isinstance(v, bool) for v in a.values())


@prop(PERCEPTION)
def test_an_organ_without_weights_returns_not_supported_in_the_ordinary_format():
    """Not an exception and not an omission. The reasoning layer never branches on whether a
    key exists, so an unavailable module must differ from an available one in the value of a
    field rather than in the shape of the object."""
    for organ in ("heart", "gallbladder"):
        rep = ultrasound_agent(organ)
        assert rep["status"] == "not_supported"
        assert rep["findings"] == []
        assert S.validate_report(rep) == [], rep
        assert "not present" in rep["reliability"]["scope"]


@prop(SCHEMA_REJECTION)
def test_an_unknown_organ_is_rejected_rather_than_guessed():
    try:
        ultrasound_agent("spleen")
    except ValueError as e:
        assert "unknown organ" in str(e)
    else:
        raise AssertionError("an unknown organ must not silently produce a report")


@prop(PERCEPTION)
def test_real_inference_produces_a_schema_valid_report():
    """The check the suite could not make while this code lived in a notebook."""
    if not _HAVE_LUNG:
        return
    rep = ultrasound_agent("lung", image=_image())
    assert S.validate_report(rep) == [], S.validate_report(rep)
    assert rep["organ"] == "lung" and rep["status"] == "ok"


@prop(PERCEPTION)
def test_every_modelled_finding_is_accounted_for_exactly_once():
    """Four findings are screened, so four must be reported -- each either detected or
    screened-and-not-detected, never both and never dropped."""
    if not _HAVE_LUNG:
        return
    rep = predict_lung([_image()])
    labels = [f["label"] for f in rep["findings"]] + \
             [f["label"] for f in rep["not_detected"]]
    assert sorted(labels) == sorted(f.replace("_", " ") for f in LUNG_FINDINGS), labels
    assert len(labels) == len(set(labels))


@prop(MISSING_NOT_NORMAL)
def test_an_all_negative_clip_has_an_empty_findings_list_not_a_placeholder():
    """The defect this guards was real: the module once represented 'nothing above threshold'
    with a placeholder entry in `findings`. Everything there is marked detected downstream, so
    the ABSENCE of findings read as a positive one and suppressed the escalation that should
    have followed -- on exactly the scans where an unmodelled pathology is most likely to be
    the explanation.

    Forced by raising every threshold above 1.0, because random input does not reliably produce
    an all-negative clip and the case must be exercised deliberately.
    """
    if not _HAVE_LUNG:
        return
    with _thresholds(1.01):
        rep = predict_lung([_image()])

    assert rep["findings"] == [], rep["findings"]
    assert len(rep["not_detected"]) == len(LUNG_FINDINGS)
    assert rep["quality"]["no_finding_above_threshold"] is True
    assert S.validate_report(rep) == []


@prop(MISSING_NOT_NORMAL)
def test_an_all_negative_clip_still_escalates():
    """The consequence of the above, checked end to end rather than assumed. A scan where
    nothing fired must escalate, because the module has no healthy class and does not model
    pneumothorax."""
    if not _HAVE_LUNG:
        return
    from src.agents.clinical.clinical_state import build_clinical_state
    from src.agents.clinical.reasoning import escalation_decision

    with _thresholds(1.01):
        rep = predict_lung([_image()])

    st = build_clinical_state(
        {"encounter_id": "T", "triage": S.make_triage("high", 0.8),
         "ultrasound": {"lung": rep}, "clinical": {}}, labs={"troponin": 5.0, "lactate": 1.0})
    assert not any(f["detected"] for f in st["imaging"]["findings"])
    d = escalation_decision(st)
    assert d["escalate"] is True, d
    assert any("cannot exclude" in t for t in d["triggers"]), d["triggers"]


@prop(PERCEPTION)
def test_the_operating_point_is_the_one_the_training_run_chose():
    """The defect this guards cost a working model.

    `lung_calibration.json` was never exported, and both the notebook and the first version of
    this module fell back to a threshold of 0.5 for every finding. This model's outputs are
    compressed into roughly 0.2-0.8, so at 0.5 it fires on almost nothing: every scan came back
    all-negative, and the pipeline downstream read that as a screened-and-clear study.

    The tuned operating points were on disk the whole time, in the per-split results the
    training run wrote (0.30 / 0.20 / 0.35 / 0.45). A fallback to 0.5 here is a silent failure
    rather than a loud one, which is why it is asserted rather than trusted.
    """
    if not _HAVE_LUNG:
        return
    rep = predict_lung([_image()])
    q = rep["quality"]
    assert q["thresholds_source"].startswith("results_"), q["thresholds_source"]
    assert set(q["thresholds"]) == set(LUNG_FINDINGS)
    assert all(0.0 < t < 0.5 or t == 0.45 for t in q["thresholds"].values()), q["thresholds"]


@prop(PERCEPTION)
def test_a_finding_the_training_run_scored_badly_is_flagged_unreliable():
    """Pleural effusion had 24 positive clips and a tuned F1 of 0.44. It is still reported --
    suppressing it would be a different kind of lie -- but it carries the flag, so the reasoning
    layer can weigh it as the weak signal it is."""
    if not _HAVE_LUNG:
        return
    rep = predict_lung([_image()])
    assert "pleural_effusion" in rep["reliability"]["unreliable_findings"]
    for f in rep["findings"] + rep["not_detected"]:
        if f["label"] == "pleural effusion":
            assert f.get("unreliable") is True, f
            break
    else:
        raise AssertionError("pleural effusion was not reported at all")


@prop(PERCEPTION)
def test_an_uncalibrated_module_says_so_rather_than_implying_a_probability():
    """No calibrator was exported from the training notebook, so the confidence is a raw
    sigmoid. Reporting it as calibrated would let the reasoning layer weigh it against a
    genuinely calibrated module on a scale it does not share."""
    if not _HAVE_LUNG:
        return
    rep = predict_lung([_image()])
    rel = rep["reliability"]
    if not rel["confidence_calibrated"]:
        assert "RAW" in rel["scope"], rel["scope"]


@prop(PERCEPTION)
def test_the_declared_scope_travels_with_the_report():
    """Pneumothorax is not modelled, and the reasoning layer needs to know that from the
    report itself rather than from documentation."""
    if not _HAVE_LUNG:
        return
    rep = predict_lung([_image()])
    assert "pneumothorax" in rep["reliability"]["scope"].lower()
    assert rep["reliability"]["has_normal_class"] is False


@prop(PERCEPTION)
def test_inference_is_deterministic_for_the_same_input():
    if not _HAVE_LUNG:
        return
    img = _image(7)
    a = predict_lung([img])
    b = predict_lung([img])
    assert [f["confidence"] for f in a["findings"]] == \
           [f["confidence"] for f in b["findings"]]


@prop(PERCEPTION)
def test_a_clip_aggregates_by_median_not_by_one_frame():
    """Frames from a clip are repeated looks at the same anatomy, not independent evidence, so
    one blurred frame must not carry the result."""
    if not _HAVE_LUNG:
        return
    frames = [_image(i) for i in range(5)]
    clip = predict_lung(frames)
    assert clip["quality"]["frames"] == 5
    singles = [predict_lung([f]) for f in frames]

    def conf(rep, label):
        for f in rep["findings"] + rep["not_detected"]:
            if f["label"] == label:
                return f["confidence"]
        raise AssertionError(label)

    for label in (f.replace("_", " ") for f in LUNG_FINDINGS):
        values = sorted(conf(s, label) for s in singles)
        assert min(values) <= conf(clip, label) <= max(values)


@prop(PERCEPTION)
def test_no_frames_fails_rather_than_inventing_a_result():
    rep = predict_lung([])
    assert rep["status"] == "failed"
    assert rep["findings"] == []
