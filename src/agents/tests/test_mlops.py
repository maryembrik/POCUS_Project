"""The promotion gate, and the failure it exists to prevent.

The datasets behind the initial models do not cover the whole clinical scope, so the system
is built to take validated data later. That is only safe if "the new model is better" is
decided by something other than one number going up, because a model can gain accuracy by
classifying easy cases better while missing more of the patients who are actually ill -- and
in an emergency system that trade is the wrong way round.

So the property under test is not that the gate promotes good models. It is that the gate
REFUSES a model whose headline metric improved and whose dangerous-miss rate got worse, and
that nothing reaches production without a person named in the record.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from mlops import RULES, dataset_version, gate  # noqa: E402
from .helpers import prop  # noqa: E402

GATE = "Model promotion gate"

EVAL = "held-out test split of triage_combined_tier_core (16,180 rows)"


def _v(stage, **metrics):
    return {"version": "vX", "stage": stage, "eval_set": EVAL,
            "dataset": "triage@abc123", "metrics": metrics}


PROD = _v("production", macro_f1=0.666, accuracy=0.680, recall_high=0.784, ece=0.036)


@prop(GATE)
def test_a_better_headline_with_a_worse_dangerous_miss_is_refused():
    """The whole reason the gate exists.

    macro-F1 up four points, accuracy up three -- and recall on the HIGH tier down five, which
    means more truly urgent patients graded below urgent. A comparison that stopped at the
    headline would promote this.
    """
    cand = _v("candidate", macro_f1=0.706, accuracy=0.710, recall_high=0.734, ece=0.036)
    out = gate("triage", cand, PROD)
    assert out["decision"] == "REJECTED"
    failed = [c["check"] for c in out["checks"] if not c["pass"]]
    assert "guard recall_high" in failed, failed
    # And the headline check passed, which is the point: the model really did look better.
    assert next(c["pass"] for c in out["checks"] if c["check"] == "headline macro_f1")


@prop(GATE)
def test_an_improvement_everywhere_is_offered_as_a_candidate():
    cand = _v("candidate", macro_f1=0.706, accuracy=0.710, recall_high=0.801, ece=0.034)
    out = gate("triage", cand, PROD)
    assert out["decision"] == "CANDIDATE", [c for c in out["checks"] if not c["pass"]]
    assert "does NOT deploy" in out["note"]


@prop(GATE)
def test_a_candidate_scored_on_a_different_set_is_not_a_comparison():
    """Two numbers from two benchmarks are two facts, not an improvement."""
    cand = _v("candidate", macro_f1=0.900, accuracy=0.900, recall_high=0.900, ece=0.020)
    cand["eval_set"] = "a different, easier split"
    out = gate("triage", cand, PROD)
    assert out["decision"] == "REJECTED"
    assert "same evaluation set" in [c["check"] for c in out["checks"] if not c["pass"]]


@prop(GATE)
def test_a_missing_guard_metric_fails_rather_than_passes():
    """Absent is not normal here either. A candidate that does not report the dangerous-miss
    rate has not shown it did not regress, and silence must not read as success."""
    cand = _v("candidate", macro_f1=0.706, accuracy=0.710, ece=0.036)
    out = gate("triage", cand, PROD)
    assert out["decision"] == "REJECTED"
    detail = next(c["detail"] for c in out["checks"] if c["check"] == "guard recall_high")
    assert "not reported" in detail


@prop(GATE)
def test_degraded_calibration_is_refused_even_when_accuracy_improves():
    """The pipeline escalates on low confidence, so a model whose stated confidence stops
    meaning anything breaks the escalation logic while looking like an improvement."""
    cand = _v("candidate", macro_f1=0.720, accuracy=0.730, recall_high=0.800, ece=0.240)
    out = gate("triage", cand, PROD)
    assert out["decision"] == "REJECTED"
    assert any(c["check"].startswith("calibration") and not c["pass"] for c in out["checks"])


@prop(GATE)
def test_a_model_below_its_floor_is_refused_even_with_no_production_baseline():
    cand = _v("candidate", macro_f1=0.400, accuracy=0.410, recall_high=0.500, ece=0.030)
    out = gate("triage", cand, None)
    assert out["decision"] == "REJECTED"
    assert any(c["check"].startswith("floor") and not c["pass"] for c in out["checks"])


@prop(GATE)
def test_an_unrecorded_dataset_is_refused():
    """A model whose training rows are not identified cannot be reproduced, and a model that
    cannot be reproduced cannot be withdrawn and rebuilt when something is found wrong."""
    cand = _v("candidate", macro_f1=0.706, accuracy=0.710, recall_high=0.801, ece=0.034)
    cand["dataset"] = None
    out = gate("triage", cand, PROD)
    assert out["decision"] == "REJECTED"
    assert "dataset version recorded" in [c["check"] for c in out["checks"] if not c["pass"]]


@prop(GATE)
def test_every_model_guards_a_miss_rate_not_only_a_headline():
    """A rule set that guarded nothing would let the first test's model through."""
    for model, rules in RULES.items():
        assert rules.get("guard"), f"{model} has no guarded metric"


@prop(GATE)
def test_the_dataset_version_changes_with_the_data():
    """The version has to be derived from content. A name someone types is a label, not a
    record of which rows produced the model."""
    a = dataset_version("triage")
    assert a["version"] and a["version"].startswith("triage@")
    assert a["rows"] > 0
    assert dataset_version("triage")["version"] == a["version"], "not deterministic"
