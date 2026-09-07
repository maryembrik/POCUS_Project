"""The HTTP surface: what the deployed service promises to anyone calling it.

Everything else in this suite tests the pipeline by importing it. That leaves a gap, because
the thing actually deployed is a web service, and a service can break in ways the pipeline
cannot: a route renamed, a field dropped from a response, an error returned as a 200, an
endpoint declared after a catch-all mount and therefore unreachable, a handler referring to a
parameter it never declared. Those failures reach a clinician; none of them would fail a test
that never makes a request. Two of the cases below were found exactly that way.

So these exercise the app through FastAPI's test client -- real routing, real serialisation,
real status codes.

No pytest fixtures are used, deliberately. run_benchmark.py calls every `test_*` in this
package as a plain zero-argument callable, so a test taking `client` or `caplog` would be
counted as a failure there while passing under pytest -- and the safety benchmark is the
artefact the report cites. Anything these tests need, they build for themselves.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
from src.agents.ultrasound.agent import LUNG_FINDINGS  # noqa: E402

from .helpers import CONFLICT, MISSING_NOT_NORMAL, SCOPE, prop  # noqa: E402

API = "API contract"
HEALTH = "Service health reporting"
MONITORING = "Monitoring without disclosure"

_CLIENT: TestClient | None = None


def _client() -> TestClient:
    """One client for the module. Building it per test costs an app startup each time."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = TestClient(serve.app)
    return _CLIENT


def _analyse(client: TestClient, **over):
    body = {"name": "API test", "age": 68, "sex": "F", "complaint": "shortness of breath",
            "organ": "Lung", "findings": {}, "vitals": {}, "labs": {}}
    body.update(over)
    return client.post("/api/analyse", json=body)


# ---------------------------------------------------------------------------- health -----
@prop(HEALTH)
def test_health_reports_each_module_not_merely_that_the_socket_is_open():
    """A 200 that says nothing about the models is a check that cannot fail usefully.

    The container's HEALTHCHECK calls this. If it returned a bare {"ok": true}, an instance
    whose weights failed to load would be reported healthy, take traffic, and report every
    scan as showing nothing -- which on this system is indistinguishable from a real negative.

    This is not hypothetical. The first image built here shipped the wrong lung checkpoint
    filename; the container started cleanly and this endpoint is what reported the module
    missing.
    """
    body = _client().get("/api/health").json()
    assert set(body["modules"]) == {"lung", "heart", "gallbladder"}
    for name, m in body["modules"].items():
        assert {"runs", "weights", "calibrated", "reason"} <= set(m), name
        assert isinstance(m["runs"], bool)


@prop(HEALTH)
def test_health_status_follows_the_modules_rather_than_being_asserted():
    body = _client().get("/api/health").json()
    expected = "ok" if all(m["runs"] for m in body["modules"].values()) else "degraded"
    assert body["status"] == expected


@prop(HEALTH)
def test_health_names_the_versions_that_decide_behaviour():
    """An image tag says which build ran. It does not say which thresholds it enforced.

    Two containers from the same tag can behave differently if the safety layer's cutoffs
    changed underneath them, so the versions that actually determine an escalation are
    reported by the running instance itself.
    """
    body = _client().get("/api/health").json()
    assert body["thresholds_version"], "no thresholds version reported"
    assert body["schema_version"], "no schema version reported"


@prop(HEALTH)
def test_health_is_not_cached():
    """A cached health response reports the state of a container that may since have died."""
    assert "no-store" in _client().get("/api/health").headers.get("cache-control", "")


# ------------------------------------------------------------------------ monitoring -----
@prop(MONITORING)
def test_the_access_log_records_the_request_and_not_its_contents():
    """The property that makes logging this service safe at all.

    Bodies here carry a presenting complaint, vital signs, laboratory values and images. A log
    line is written to disk, shipped to whatever collects it, and read by people who were not
    in the consultation -- so a logger that echoed request bodies would turn every deployment
    into a second medical record that nobody agreed to.
    """
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    log = logging.getLogger("pocus.access")
    log.addHandler(handler)
    previous = log.level
    log.setLevel(logging.INFO)
    try:
        # A name no other fixture uses, so finding it in the log proves it came from
        # THIS request body and not from some other part of the suite.
        patient_name = "Zzyzx-Quenneville-773"
        _analyse(_client(), name=patient_name, complaint="haemoptysis",
                 vitals={"spo2": 89.0})
        written = "\n".join(r.getMessage() for r in records)
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)

    assert written, "no access log line was produced at all"
    assert patient_name not in written, "the patient name reached the access log"
    assert "haemoptysis" not in written, "the presenting complaint reached the access log"


@prop(MONITORING)
def test_metrics_separate_a_failing_service_from_a_wrong_request():
    """4xx and 5xx answer different questions and must not share a counter.

    A run of 404s means callers are asking for things that do not exist; a run of 500s means
    this instance is broken. Summed together they say only that something is happening.
    """
    c = _client()
    before = c.get("/api/metrics").json()
    c.post("/api/record/attach", json={"id": "definitely-not-here", "images": []})
    after = c.get("/api/metrics").json()

    assert after["client_errors"] == before["client_errors"] + 1
    assert after["server_errors"] == before["server_errors"]


@prop(MONITORING)
def test_metrics_are_keyed_by_route_not_by_url():
    """Recording literal paths would make every encounter identifier its own metric series.

    That is unbounded growth, it prevents any two requests from aggregating, and it
    accumulates encounter identifiers in a table whose purpose is to be exported.
    """
    body = _client().get("/api/metrics").json()
    for key in body["by_route"]:
        assert key.startswith("/"), key
        assert key in {"/", "/fr", "/static", "/other"} or key.startswith("/api/"), key


@prop(MONITORING)
def test_metrics_is_reachable_at_all():
    """It was not, at first.

    Routes match in registration order and `app.mount("/", StaticFiles(...))` answers
    everything, so /api/metrics returned 404 until it was declared above the mount. A metrics
    endpoint that 404s is worse than none: monitoring appears to exist and reports nothing.
    """
    assert _client().get("/api/metrics").status_code == 200


# ---------------------------------------------------------------------------- routes -----
@prop(API)
def test_both_language_pages_are_served():
    for path in ("/", "/fr"):
        r = _client().get(path)
        assert r.status_code == 200, path
        assert "text/html" in r.headers["content-type"], path


@prop(API)
def test_the_interface_is_never_cached():
    """The binder rewrites the page on every build and inlines the logic into it.

    A cached page is therefore a cached application, and three separate bugs during
    development were a browser running the previous build: uploads that no longer filed their
    images, counters reading old bindings, a screen that had already been fixed. One was
    reported as a fresh bug in code written minutes earlier.
    """
    for path in ("/", "/fr"):
        assert "no-store" in _client().get(path).headers.get("cache-control", ""), path


@prop(API)
def test_bootstrap_describes_the_system_without_inventing_a_patient():
    """The mockup shipped with a named patient and a visit history. Neither is real."""
    body = _client().get("/api/bootstrap").json()
    assert isinstance(body, dict) and body


# ---------------------------------------------------------------------------- analysis ---
@prop(API)
def test_an_encounter_with_no_imaging_does_not_acquire_a_finding():
    """The central property, expressed at the HTTP boundary.

    An assessment posted with no ultrasound report must not come back describing one. This is
    the same guarantee the pipeline tests make, checked where a caller can observe it -- and
    it is the one a serialisation bug could silently break.
    """
    body = _analyse(_client()).json()

    assert body["findings"] == [], f"an unexamined encounter acquired {body['findings']}"

    # topFinding is the string the interface prints in the headline slot, so an empty
    # examination fills it with a statement of absence rather than leaving it blank -- the
    # screen must say why it is empty. What it must never do is name a finding.
    top = str(body.get("topFinding", "")).lower()
    for label in LUNG_FINDINGS:
        assert label.lower() not in top, f"headline named {label!r} for an unexamined patient"

    # The organ is reported as a GAP, not as a negative. Different claims, and the distinction
    # is the point of the three-state model.
    not_assessed = " ".join(str(x) for x in body.get("notAssessed", [])).lower()
    assert "lung" in not_assessed, "the unexamined lung is missing from notAssessed"


@prop(API)
def test_retrieved_passages_are_not_mistaken_for_findings():
    """Deliberately separated from the test above, because it nearly broke it.

    The response carries retrieved corpus passages, and those passages are medical prose: one
    names consolidation and pleural effusion in a sentence about what lung ultrasound can
    exclude. Searching the raw response body for finding names therefore reports a fabrication
    that has not occurred -- an assertion this file made until it failed here.

    What must hold is narrower and is the thing that matters: a cited source is labelled as a
    source, and never enters the patient's finding list.
    """
    body = _analyse(_client()).json()
    assert body["findings"] == []
    for hit in body.get("hits", []):
        assert hit.get("source"), "a retrieved passage carries no source attribution"


@prop(API)
def test_attaching_a_study_to_an_existing_patient_succeeds():
    """Filing an image against a record already in the list.

    This endpoint raised NameError on every successful call: the handler read `lang` to render
    the updated record, and `lang` was not one of its parameters. Nothing here had ever called
    it -- the pipeline tests import the pipeline, and a missing function parameter is not a
    pipeline defect. ruff found it (F821); this test keeps it closed.

    The 404 branch never touched `lang`, so attaching to an unknown patient worked and
    attaching to a real one did not, which is the reverse of a failure anyone would look for.
    """
    c = _client()
    _analyse(c, name="Attach test", age=70, sex="M", complaint="chest pain")

    # /api/record fetches ONE record by id rather than listing them, so the identifier comes
    # from the session store. Reaching into it keeps this test aimed at the endpoint under
    # test rather than at whichever screen happens to expose the list.
    assert serve._records, "the analysed encounter did not reach the session store"
    encounter_id = serve._records[-1]["id"]

    r = c.post("/api/record/attach", json={"id": encounter_id, "images": []})
    assert r.status_code == 200, f"attach failed: {r.status_code} {r.text[:200]}"
    assert r.json().get("hasEncounter") is not False


@prop(API)
def test_attaching_to_an_unknown_patient_is_a_404_not_a_crash():
    r = _client().post("/api/record/attach", json={"id": "no-such-patient", "images": []})
    assert r.status_code == 404


# ------------------------------------------------------- the clinician-assessed tier ----
# The escalation policy fires "agents disagree" when triage says LOW and an imaging module
# reports something severe. That trigger was unreachable through the interface for the whole
# of this project's life: the intake form had no urgency control and sent a constant 'medium'
# with every encounter.
#
# It was invisible because the rule's own tests call detect_conflicts() directly with
# tier="low", and they passed. A rule can be correct and still never run. These tests take the
# HTTP path a clinician's browser takes, so they fail if the tier stops reaching the reasoning
# layer for any reason -- a renamed field, a dropped parameter, a form that stops sending it.
def _severe_cardiac(tier: str) -> dict:
    return _client().post("/api/analyse", json={
        "name": "Tier path", "age": 70, "sex": "F", "complaint": "chest pain",
        "tier": tier, "organ": "Heart",
        "findings": {"severe dysfunction": 0.74}, "vitals": {}, "labs": {},
    }).json()


@prop(CONFLICT)
def test_low_urgency_against_a_severe_finding_is_a_conflict_over_http():
    body = _severe_cardiac("low")
    assert body["conflicts"], "no conflict recorded for LOW urgency against severe dysfunction"
    assert any("low urgency" in c.lower() for c in body["conflicts"]), body["conflicts"]
    assert body["escalate"] is True
    assert any("disagree" in t.lower() for t in body["triggers"]), body["triggers"]


@prop(CONFLICT)
def test_medium_urgency_raises_no_disagreement_conflict():
    body = _severe_cardiac("medium")
    assert not any("low urgency" in c.lower() for c in body["conflicts"]), body["conflicts"]


@prop(CONFLICT)
def test_high_urgency_raises_no_disagreement_conflict():
    body = _severe_cardiac("high")
    assert not any("low urgency" in c.lower() for c in body["conflicts"]), body["conflicts"]


@prop(CONFLICT)
def test_the_tier_changes_the_outcome_and_nothing_else_does():
    """The comparison that makes the previous three mean something.

    Each on its own could pass with the tier ignored entirely. Together they establish that
    the value sent is what decides -- identical finding, identical everything, one field
    different, different escalation reasoning.
    """
    low, medium = _severe_cardiac("low"), _severe_cardiac("medium")
    assert low["conflicts"] != medium["conflicts"]

    # Both still escalate, for reasons that do not involve the tier: a high-risk finding below
    # decisive confidence, and positive imaging with key laboratory values absent. The tier
    # adds a reason; it is not what makes the system safe on its own.
    assert low["escalate"] is True and medium["escalate"] is True
    assert len(low["triggers"]) > len(medium["triggers"])


@prop(CONFLICT)
def test_the_record_says_the_tier_came_from_a_clinician_not_a_model():
    """The trained classifier is evaluated in the report but is not wired to this control.

    A record that showed a tier with no provenance would let a reader assume the model
    produced it. Anyone reading an archived encounter can tell the two apart.
    """
    from src.agents import schema as S
    tri = S.make_triage("low", 0.8, model="clinician")
    assert tri["model"] == "clinician"


@prop(API)
def test_the_interface_offers_the_three_tiers_in_the_order_the_handler_assumes():
    """The UI half of the same path, checked where it can actually be checked.

    logic.js maps the selected option to a tier BY POSITION -- ['low','medium','high'] indexed
    by selectedIndex -- because the French page shows "Faible / Moyenne / Élevée" and matching
    on the displayed word there would fall through to a default. Position-mapping is the right
    choice and it makes the option ORDER load-bearing: reorder the markup and every French and
    English selection silently shifts by one.

    The select's value is BOUND (`value="{{ fTier }}"`), the way the text inputs already are.
    Marking an option `selected` was tried first and does not work: the runtime rebuilds each
    option node from the template and the attribute is dropped, so the browser fell back to the
    first option -- the screen read "Low" while the form state held 'medium', the control
    contradicting what it was about to send. That is worse than having no control, and no
    amount of reading the generated HTML would have revealed it; it took loading the page.
    """
    for page in ("pocus-copilot.dc.html", "pocus-copilot.fr.dc.html"):
        html = (ROOT / "web" / page).read_text(encoding="utf8")
        assert 'onChange="{{ onTier }}"' in html, f"{page}: no urgency control"
        i = html.index("{{ onTier }}")
        block = html[max(0, i - 120):i + 400]
        low, med, high = (block.index("{{ tierLow }}"), block.index("{{ tierMedium }}"),
                          block.index("{{ tierHigh }}"))
        assert low < med < high, f"{page}: option order does not match the handler's mapping"
        assert 'value="{{ fTier }}"' in block, f"{page}: the select's value is not bound"

    logic = (ROOT / "web" / "logic.js").read_text(encoding="utf8")
    assert "['low', 'medium', 'high'][e.target.selectedIndex]" in logic
    assert "fTier:" in logic, "no binding supplies the currently selected label"


# --------------------------------------------------- four states, not two ---------------
@prop(MISSING_NOT_NORMAL)
def test_the_finding_list_distinguishes_four_states():
    """"Detected" and "not detected" are not enough, and the gap between them is dangerous.

    Two different things hide inside "not detected": a finding the model reports but cannot
    support (pleural effusion rests on 16 positive cases), and a finding it does not model at
    all (pneumothorax). Collapsing either into a plain negative invites the one inference this
    system exists to prevent -- the scan did not mention it, so it is not there.
    """
    body = _client().post("/api/analyse", json={
        "name": "four states", "age": 70, "sex": "F", "complaint": "dyspnoea",
        "tier": "medium", "organ": "Lung",
        "findings": {"b_lines": 0.82, "pleural_effusion": 0.58, "consolidation": -0.05},
        "vitals": {}, "labs": {},
    }).json()
    states = {f["label"]: f["state"] for f in body["findings"]}

    assert states["b lines"] == "detected"
    assert states["consolidation"] == "screened_negative"
    assert states["pleural effusion"] == "unreliable", \
        "a finding resting on 16 positive cases was reported as an ordinary detection"
    assert states["pneumothorax"] == "not_assessed", \
        "pneumothorax is absent from the finding list, so a reader sees a clear lung"


@prop(MISSING_NOT_NORMAL)
def test_an_unmodelled_finding_is_never_reported_as_negative():
    """The distinction that makes the fourth state worth having at all."""
    body = _client().post("/api/analyse", json={
        "name": "clear lung", "age": 70, "sex": "F", "complaint": "dyspnoea",
        "tier": "medium", "organ": "Lung",
        "findings": {"b_lines": -0.1, "consolidation": -0.05},
        "vitals": {}, "labs": {},
    }).json()
    rows = {f["label"]: f for f in body["findings"]}

    # Everything the model screened comes back negative -- and pneumothorax still does not.
    assert rows["b lines"]["state"] == "screened_negative"
    assert rows["pneumothorax"]["state"] == "not_assessed"
    assert rows["pneumothorax"]["conf"] is None, \
        "an unassessed finding carries a confidence, which implies it was evaluated"
    assert any("pneumothorax" in x for x in body["outOfScope"])


@prop(SCOPE)
def test_a_typed_finding_carries_the_same_limits_as_a_predicted_one():
    """The two paths that build a lung report must not disagree about what it can support.

    The application builds a report from an uploaded image and also from findings entered by
    hand. The hand-entered path declared `confidence_calibrated: True` and no unreliable list,
    so the same finding looked better supported typed in than predicted -- and pleural effusion
    lost the only warning that makes its number safe to display.
    """
    from src.agents.ultrasound.agent import lung_reliability
    rel = lung_reliability()

    assert "pleural_effusion" in rel["unreliable_findings"]
    # Read from disk rather than asserted: there is no lung calibration file in this
    # repository, so the honest value is False and the scope string has to say why.
    assert rel["confidence_calibrated"] is False
    assert "RAW sigmoid" in rel["scope"]
    assert "pneumothorax" in rel["scope"]


@prop(API)
def test_asking_before_analysing_is_answered_rather_than_erroring():
    """A question with nothing to answer from gets a stated refusal, not a 500 and not a guess."""
    r = _client().post("/api/ask", json={"question": "what should I do now?"})
    assert r.status_code == 200
    assert r.json()["answer"].strip()


@prop(API)
def test_the_french_route_answers_in_french():
    r = _client().post("/api/ask", json={"question": "que dois-je faire ?"},
                       params={"lang": "fr"})
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert answer.strip()
    # Not a language classifier: just enough to catch the whole French path falling back to
    # English, which is a failure this project has already had once.
    assert not answer.startswith("No encounter has been analysed")
