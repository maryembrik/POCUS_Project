"""Bind a Claude Design export to the pipeline.

    python tools/bind_design.py

Reads the pristine export from `web/design/` and writes the bound page to `web/`. Re-run it
after re-exporting the design; nothing here is hand-edited, so a new export costs one command
rather than an afternoon of surgery.

Two kinds of change are made, and only two:

  1. The mockup's logic block is REPLACED by `web/logic.js`, which fetches from `/api/*`.
     Everything clinical the mockup hard-coded goes with it -- the named patient, the roster,
     the visit history, the invented counters, the pre-written chat replies.

  2. Markup that displayed those constants is rewired to the bindings `logic.js` supplies.
     Repeated blocks become `sc-for` loops; single values become `{{ }}`.

Layout, palette, typography, motion and screen structure are never touched. If a replacement
below stops matching, the tool says so loudly rather than silently producing a page that still
shows a patient who does not exist.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fr_ui  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "web" / "design" / "pocus-copilot.dc.html"
LOGIC = ROOT / "web" / "logic.js"
OUT = ROOT / "web" / "pocus-copilot.dc.html"
OUT_FR = ROOT / "web" / "pocus-copilot.fr.dc.html"

def _visible_strings(html: str) -> list[str]:
    """Every distinct piece of text the page can display, in document order."""
    body = re.sub('<(script|style)[^>]*>.*?</\\1>', "", html, flags=re.S)
    seen: list[str] = []
    for t in re.findall(r">([^<>{}]+)<", body):
        t = t.strip()
        if t and re.search(r"[A-Za-z]{3}", t) and t not in seen:
            seen.append(t)
    return seen


applied: list[str] = []
missed: list[str] = []


def _match_close(s: str, start: int) -> int:
    """Index of the `</sc-if>` that closes the block opened before `start`.

    Nesting-aware. Tag counts balancing across the whole file says nothing about a close being
    in the right place, and a misplaced one silently nested every later screen inside this
    block -- they rendered empty while the totals still matched.
    """
    depth, k = 1, start
    while k < len(s):
        nxt_open = s.find("<sc-if", k)
        nxt_close = s.find("</sc-if>", k)
        if nxt_close < 0:
            return -1
        if 0 <= nxt_open < nxt_close:
            depth += 1
            k = nxt_open + 6
        else:
            depth -= 1
            if depth == 0:
                return nxt_close
            k = nxt_close + 8
    return -1


def sub(s: str, old: str, new: str, label: str) -> str:
    """Replace EVERY occurrence.

    Replacing only the first left the Assessment screen showing the mockup's sentence about a
    respiratory problem while the workup screen showed the real conclusion -- the same string
    appears on both, and the one that mattered was the second.
    """
    global applied, missed
    if old in s:
        applied.append(label)
        return s.replace(old, new)
    missed.append(label)
    return s


def cut(s: str, start: str, end: str, new: str, label: str) -> str:
    """Replace from `start` through the first `end` after it."""
    global applied, missed
    i = s.find(start)
    if i < 0:
        missed.append(label + " (start)")
        return s
    j = s.find(end, i + len(start))
    if j < 0:
        missed.append(label + " (end)")
        return s
    applied.append(label)
    return s[:i] + new + s[j + len(end):]


def main() -> int:
    s = io.open(SRC, encoding="utf8").read()
    logic = io.open(LOGIC, encoding="utf8").read()

    # ---- 1. swap the logic block -----------------------------------------------------
    OPEN = '<script type="text/x-dc" data-dc-script data-props="{}">'
    i = s.find(OPEN)
    if i < 0:
        print("FATAL: no dc script block found in the export", file=sys.stderr)
        return 1
    j = s.find("</script>", i)
    s = s[:i + len(OPEN)] + "\n" + logic + s[j:]
    applied.append("logic block")

    # ---- 2. identity, wherever the mockup wrote a name ------------------------------
    s = sub(s, ">Sarah Martin</div>\n    <div style=\"font-size:13px;color:#6A6785\">"
               "74 · Female · Acute breathlessness</div>",
            ">{{ pName }}</div>\n    <div style=\"font-size:13px;color:#6A6785\">"
            "{{ pAge }} · {{ pSex }} · {{ pComplaint }}</div>", "sidebar patient")
    # The sidebar priority chip was left hard-coded to "High priority", so an app with no
    # patient loaded still displayed a red high-priority badge under the words "No patient".
    s = re.sub(r'<div style="margin-top:12px;display:inline-flex;align-items:center;gap:7px;'
               r'background:#FDECEC;color:#C13238;border-radius:999px;padding:5px 11px;'
               r'font-size:12\.5px;font-weight:700">● High priority</div>',
               '<div style="margin-top:12px"><span style="{{ severityStyle }}">● '
               '{{ severity }}</span></div>', s)
    s = re.sub(r"Sarah Martin · 74 · Female · Acute breathlessness",
               "{{ pName }} · {{ pAge }} · {{ pSex }} · {{ pComplaint }}", s)
    s = re.sub(r"Sarah Martin · 74 · Female", "{{ pName }} · {{ pAge }} · {{ pSex }}", s)
    s = re.sub(r"Sarah Martin · 74F", "{{ pName }} · {{ pAge }}{{ pSex }}", s)
    s = re.sub(r"Sarah Martin · today", "{{ pName }} · encounter {{ pId }}", s)
    s = re.sub(r">Sarah Martin<", ">{{ pName }}<", s)
    s = re.sub(r"Continue the assessment for Sarah Martin\.",
               "Continue the current encounter.", s)
    s = re.sub(r"Attending: Dr\. A\. Reyes", "Encounter {{ pId }}", s)
    s = re.sub(r"Good morning, Dr\. Reyes", "{{ pName }}", s)
    s = re.sub(r"74 · Female · MRN [\d-]+ · Bed \d+",
               "{{ pAge }} · {{ pSex }} · {{ pComplaint }} · session-scoped, no database", s)
    applied.append("identity")

    # ---- 3. the clinician chip names a person who does not exist --------------------
    s = sub(s, '<div style="font-weight:600;font-size:13.5px">Dr. A. Reyes</div>'
               '<div style="font-size:12px;color:#8A87A8">Emergency Medicine</div>',
            '<div style="font-weight:600;font-size:13.5px">Perception modules</div>'
            '<div style="font-size:12px;color:#8A87A8">'
            '<sc-for list="{{ modules }}" as="m" hint-placeholder-count="3">'
            '<span>{{ m.dot }} {{ m.organ }} </span></sc-for></div>', "clinician chip")

    # ---- 4. hero counters ------------------------------------------------------------
    TILE = ('<div style="background:rgba(255,255,255,.14);border-radius:16px;padding:16px 18px">'
            '<div style="font-size:28px;font-weight:800">{n}</div>'
            '<div style="font-size:12.5px;opacity:.88">{t}</div></div>')
    for on, ot, nn, nt in (("12", "Assessments today", "{{ sessionCount }}",
                            "Patients this session"),
                           ("3", "Require review", "{{ studyCount }}",
                            "POCUS studies read"),
                           ("2", "Critical alerts", "{{ criticalCount }}", "High priority"),
                           ("7", "Completed", "{{ testCount }}", "Safety tests passing")):
        s = sub(s, TILE.format(n=on, t=ot), TILE.format(n=nn, t=nt), f"tile {ot}")

    # ---- 4b. the analyse button never said it was working ---------------------------
    # It now waits for a study still being read, and a button that looks idle while it waits
    # invites a second click on an encounter that is already being assessed.
    s = sub(s, 'font-size:14.5px;padding:13px 24px;border-radius:999px">✦ Analyze patient'
               '</button>',
            'font-size:14.5px;padding:13px 24px;border-radius:999px">{{ analyseLabel }}'
            '</button>', "analyse button label")

    # ---- 5. the workup form fields must actually write somewhere --------------------
    s = sub(s, 'Name<input value="Sarah Martin"',
            'Name<input value="{{ fName }}" onChange="{{ onName }}"', "workup name")
    s = sub(s, 'Age<input value="74"',
            'Age<input value="{{ fAge }}" onChange="{{ onAge }}"', "workup age")
    s = sub(s, 'Chief complaint<input value="Acute breathlessness"',
            'Chief complaint<input value="{{ fComplaint }}" onChange="{{ onComplaint }}"',
            "workup complaint")
    s = sub(s, '<textarea style', '<textarea onChange="{{ onHistory }}" style',
            "workup history")
    s = re.sub(r'(<label[^>]*>Sex<select )', r'\1onChange="{{ onSex }}" ', s)

    # ---- 5a. the urgency tier the clinician assesses --------------------------------
    # The mockup had no control for this, and the form therefore sent a constant: every
    # patient reached the reasoning layer as "medium". One escalation trigger reads the tier
    # -- it fires when triage says LOW and imaging reports something severe -- so with the
    # value pinned, that trigger could not fire for any patient entered through this screen.
    # It passed its own tests the whole time, because those call detect_conflicts() directly.
    #
    # This is the clinician's own assessment, not a model output. The trained triage
    # classifier is NOT wired to this control: it needs arrival mode and coded reason-for-visit
    # categories that this intake does not collect, and a tier inferred from half the features
    # would be worse than one a doctor stated.
    #
    # The select's value is BOUND, not marked with `selected` on an option. `selected` was
    # tried first and does not survive: the runtime rebuilds each option node from the template
    # and the attribute is dropped, so the DOM came back with defaultSelected false on all
    # three and the browser fell back to the first -- the screen read "Low" while the form
    # state held 'medium', the control disagreeing with what it was about to send. Binding
    # value the way the text inputs already do is what the runtime actually supports.
    # The suggestion sits directly above the control it fills in, so the reader sees what was
    # proposed and what they are about to confirm in one glance. It is hidden until there is a
    # suggestion to show: an empty "Suggested urgency —" line would read as a system that had
    # considered the patient and declined to say anything.
    # `sc-if value=`, not `cond=`: the runtime reads the `value` attribute (support.js walkIf),
    # and an sc-if whose condition it does not recognise renders nothing at all -- the block was
    # in the served page and invisible on screen, with no error anywhere.
    #
    # The badge's colours come from ONE bound object rather than a bound attribute beside a
    # static style, because that is the convention the rest of the page uses and there is no
    # sc-style in this runtime.
    _SUGGESTION = (
        '\n      <sc-if value="{{ hasSuggestion }}"><div style="display:flex;'
        'align-items:center;gap:10px;margin:14px 0 2px">'
        '<span style="font-size:12.5px;font-weight:700;color:#6A6785">'
        '{{ suggestedLabel }}</span>'
        '<span style="{{ suggestedStyle }}">{{ suggestedTier }}</span>'
        '</div>'
        '<div style="font-size:12.5px;color:#8A87A8;margin-bottom:6px">{{ suggestedBasis }}'
        '</div></sc-if>')

    # The demographics grid closes right after the Sex field. The two new SELECTS join it as
    # further cells; the suggestion goes AFTER the closing div so it spans the full width.
    # Inside the grid it was squeezed into one narrow column and "Based on 4 of 7 observations.
    # Not recorded: diastolic blood pressure, temperature, pain score." wrapped over six lines
    # beside the control it describes.
    _SEX_FIELD = ('<option>Female</option><option>Male</option></select></label>')
    _GRID_END = _SEX_FIELD + '\n      </div>'
    _URGENCY = (
        _SEX_FIELD +
        '\n        <label style="display:flex;flex-direction:column;gap:6px;font-size:12.5px;'
        'font-weight:700;color:#6A6785">{{ tierLabel }}'
        '<select value="{{ fTier }}" onChange="{{ onTier }}" '
        'style="border:1px solid #DEDCF4;border-radius:8px;padding:10px 12px;'
        'font-size:14.5px;background:#FAFAFE">'
        '<option>{{ tierLow }}</option>'
        '<option>{{ tierMedium }}</option>'
        '<option>{{ tierHigh }}</option>'
        '</select></label>'
        # Arrival mode sits beside the urgency because they are read together: how the patient
        # got here is part of the same first impression the tier records.
        '\n        <label style="display:flex;flex-direction:column;gap:6px;font-size:12.5px;'
        'font-weight:700;color:#6A6785">{{ arrivalLabel }}'
        '<select value="{{ fArrival }}" onChange="{{ onArrival }}" '
        'style="border:1px solid #DEDCF4;border-radius:8px;padding:10px 12px;'
        'font-size:14.5px;background:#FAFAFE">'
        '<option>{{ arrivalWalk }}</option>'
        '<option>{{ arrivalAmbulance }}</option>'
        '</select></label>'
        '\n      </div>' + _SUGGESTION)
    s = sub(s, _GRID_END, _URGENCY, "assessed urgency control")
    # the loop inputs the design lays out are display-only in the mockup
    s = re.sub(r'(<input value="\{\{ v\.value \}\}")', r'\1 onChange="{{ v.onChange }}"', s)
    s = re.sub(r'(<input value="\{\{ l\.value \}\}")', r'\1 onChange="{{ l.onChange }}"', s)
    applied.append("workup inputs")

    # ---- 5b. the organ chips must name the modules that exist ------------------------
    # The mockup offers Lung / Cardiac / FAST. No FAST module was ever built, and an
    # examination tab with nothing behind it is a claim that cannot be backed -- on the one
    # screen whose purpose is separating "not detected" from "never examined". The chips become
    # a loop over the organs the agent can actually run, and they select one.
    s = cut(s, '<div style="display:flex;gap:8px;flex-wrap:wrap">\n'
               '        <span style="background:#5B54D6;color:#fff;border-radius:999px;'
               'padding:7px 14px;font-size:12.5px;font-weight:700">Lung</span>',
            '>FAST</span>',
            '<div style="display:flex;gap:8px;flex-wrap:wrap">\n'
            '        <sc-for list="{{ organChips }}" as="o" hint-placeholder-count="3">\n'
            '        <button type="button" onClick="{{ o.onClick }}" style="{{ o.style }}">'
            '{{ o.label }}</button>\n        </sc-for>', "organ chips")

    # ---- 5b2. surface a rejected analysis instead of leaving the old encounter up -----
    s = sub(s, '{{ filledCount }} of 5 sections filled',
            '{{ filledCount }} of 5 sections filled'
            '<sc-if value="{{ hasError }}" hint-placeholder-val="{{ false }}">'
            '<span style="display:block;margin-top:6px;color:#C13238;font-weight:700;'
            'font-size:12.5px;max-width:46ch">{{ error }}</span></sc-if>',
            "analysis error")

    # ---- 5c. the clip slots must accept a real file ---------------------------------
    # `image-slot` is a Design Canvas drop target that persists to a sidecar; it never reaches
    # this application, so clicking it did nothing. The two slots become previews of what was
    # actually uploaded, and the "+ Add clip" tile becomes a real file input that posts to
    # /api/upload and runs the module. Cardiac takes two frames, so the input accepts multiple.
    s = cut(s, '<div style="display:grid;grid-template-columns:repeat(auto-fill,'
               'minmax(112px,1fr));gap:10px">\n'
               '        <div style="position:relative;aspect-ratio:4/3;background:#141824;'
               'border-radius:6px;overflow:hidden"><image-slot id="wk-clip-1"',
            'text-align:center;padding:8px">＋ Add clip</div>',
            '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));'
            'gap:10px">\n'
            '        <sc-for list="{{ clips }}" as="c" hint-placeholder-count="2">\n'
            '        <div style="position:relative;aspect-ratio:4/3;background:#141824;'
            'border-radius:6px;overflow:hidden">'
            '<img src="{{ c.src }}" style="width:100%;height:100%;object-fit:cover"></div>\n'
            '        </sc-for>\n'
            '        <label style="aspect-ratio:4/3;border:1px dashed #DEDCF4;border-radius:6px;'
            'display:flex;align-items:center;justify-content:center;color:#5B54D6;'
            'font-size:12.5px;text-align:center;padding:8px;cursor:pointer;font-weight:700">'
            '{{ uploadLabel }}'
            '<input type="file" accept="image/*" multiple onChange="{{ onUpload }}" '
            'style="display:none"></label>', "clip upload")

    # ---- 6. temperature note is a fact about this encounter -------------------------
    s = re.sub(r"Temperature is empty\. It will be recorded as not measured\.",
               "{{ missingCount }} value(s) left blank. Each is recorded as not measured, "
               "never as normal.", s)

    # ---- 6b. THE ASSESSMENT SCREEN ---------------------------------------------------
    # This screen was entirely hard-coded: a patient's symptoms, vitals, POCUS, laboratory
    # values and a three-entry differential, all fixed in the markup. It is the screen a
    # clinician actually reads, and it displayed the mockup's fabricated patient whatever had
    # been entered -- a gallbladder study on a 60-year-old with no vitals recorded still showed
    # B-lines, a heart rate of 122 and pulmonary oedema. Every part of it is bound below.
    s = sub(s, "The current findings suggest a significant respiratory problem requiring "
               "prompt clinical evaluation.",
            "{{ conclusion }}", "assessment conclusion")
    s = sub(s, ">High clinical priority</div>", ">{{ severity }} clinical priority</div>",
            "assessment priority kicker")

    s = sub(s, '<div style="margin:8px 0 18px;font-size:14.5px">Acute breathlessness, '
               'fatigue</div>',
            '<div style="margin:8px 0 18px;font-size:14.5px">{{ pComplaint }}</div>',
            "sees symptoms")

    s = cut(s, '<div style="display:flex;justify-content:space-between"><span>Heart rate '
               '122 bpm</span>',
            '<span>Blood pressure 104/68</span><span style="color:#5A7A0F;font-weight:700">'
            'Normal</span></div>',
            '<sc-for list="{{ vitals }}" as="v" hint-placeholder-count="5">'
            '<div style="display:flex;justify-content:space-between">'
            '<span>{{ v.label }} {{ v.value }} {{ v.unit }}</span>'
            '<span style="{{ v.flagStyle }}">{{ v.flag }}</span></div></sc-for>',
            "sees vitals")

    s = sub(s, '<div style="margin:8px 0 18px;font-size:14.5px">B-lines detected — bilateral '
               'anterior zones</div>',
            '<div style="margin:8px 0 18px;font-size:14.5px">'
            '<sc-for list="{{ findings }}" as="f" hint-placeholder-count="4">'
            '<div>{{ f.label }} — {{ f.status }} ({{ f.conf }})</div></sc-for>'
            '<sc-for list="{{ notAssessed }}" as="o" hint-placeholder-count="1">'
            '<div style="color:#C13238;font-weight:700">{{ o.organ }} — not assessed</div>'
            '</sc-for></div>', "sees pocus")

    s = cut(s, '<div style="display:flex;justify-content:space-between"><span>BNP 890</span>',
            '<span>D-dimer</span><span>Not available</span></div>',
            '<sc-for list="{{ labs }}" as="l" hint-placeholder-count="8">'
            '<div style="display:flex;justify-content:space-between">'
            '<span>{{ l.name }} {{ l.result }}</span>'
            '<span style="{{ l.statusStyle }}">{{ l.status }}</span></div></sc-for>',
            "sees labs")

    # The three fixed differential cards become one loop over the validated answer.
    s = cut(s, '<article style="border:1px solid #DEDCF4;border-radius:16px;'
               'padding:20px 22px;background:#FCFBFF"',
            'The current examination does not assess this condition.</p>\n        </article>',
            '<sc-for list="{{ ranked }}" as="r" hint-placeholder-count="3">\n'
            '        <article style="{{ r.cardStyle }}">\n'
            '          <div style="display:flex;justify-content:space-between;'
            'align-items:baseline;gap:16px;flex-wrap:wrap">\n'
            '            <div><span style="font-size:12.5px;color:#8A87A8;font-weight:700">'
            '{{ r.rank }}</span><div style="{{ r.titleStyle }}">{{ r.name }}</div></div>\n'
            '            <span style="{{ r.tagStyle }}">{{ r.tag }}</span>\n'
            '          </div>\n'
            '          <div style="margin-top:14px;height:8px;border-radius:6px;'
            'background:#E4E2F8;overflow:hidden"><div style="{{ r.fillStyle }}"></div></div>\n'
            '          <div style="margin-top:18px;display:grid;grid-template-columns:'
            'repeat(auto-fit,minmax(220px,1fr));gap:22px">\n'
            '            <div><div style="font-size:12px;letter-spacing:.09em;'
            'text-transform:uppercase;color:#5A7A0F;font-weight:800">Why it is considered</div>'
            '<p style="margin:10px 0 0;font-size:14px">{{ r.supports }}</p>'
            '<div style="margin-top:8px;font-size:12.5px;color:#8A87A8">cited as {{ r.now }}'
            '</div></div>\n'
            '            <div><div style="font-size:12px;letter-spacing:.09em;'
            'text-transform:uppercase;color:#9A6207;font-weight:800">What limits confidence'
            '</div><p style="margin:10px 0 0;font-size:14px;color:#6A6785">{{ r.limits }}</p>'
            '</div>\n          </div>\n'
            '          <div style="margin-top:18px;display:flex;gap:10px;flex-wrap:wrap">\n'
            '            <button type="button" onClick="{{ askWhy }}" style="appearance:none;'
            'border:0;cursor:pointer;background:#5B54D6;color:#fff;font-weight:700;'
            'font-size:13.5px;padding:10px 16px;border-radius:11px">Why?</button>\n'
            '            <button type="button" onClick="{{ askChallenge }}" '
            'style="appearance:none;border:1px solid #B9B5EC;background:#fff;cursor:pointer;'
            'color:#5B3CC4;font-weight:700;font-size:13.5px;padding:10px 16px;'
            'border-radius:11px">Challenge this assessment</button>\n'
            '          </div>\n        </article>\n        </sc-for>\n'
            '        <sc-if value="{{ noDifferential }}" hint-placeholder-val="{{ true }}">\n'
            '        <article style="border:1px solid #E4E2F8;border-radius:16px;'
            'padding:20px 22px">\n'
            '          <p style="margin:0;font-size:15px;color:#6A6785">'
            '{{ differentialNote }}</p>\n        </article>\n        </sc-if>',
            "differential cards")

    # The three fixed "what would help" rows become the computed recommendations.
    s = cut(s, '<div style="display:flex;gap:14px;padding:14px 16px;background:#FFF8EC;'
               'border-radius:14px"><span style="color:#E08A00;font-size:15px">●</span>'
               '<div><div style="font-weight:700;font-size:15px">D-dimer</div>',
            'Would help weigh an infective cause.</div></div></div>',
            '<sc-for list="{{ exams }}" as="e" hint-placeholder-count="3">'
            '<div style="display:flex;gap:14px;padding:14px 16px;background:#FFF8EC;'
            'border-radius:14px;margin-bottom:12px">'
            '<span style="color:#E08A00;font-size:15px">●</span>'
            '<div><div style="font-weight:700;font-size:15px">{{ e.exam }}</div>'
            '<div style="font-size:13.5px;color:#6A6785">{{ e.reason }}</div></div></div>'
            '</sc-for>', "clarify list")

    s = sub(s, "Consider obtaining the missing high-priority investigations and reassessing "
               "the patient's respiratory status.",
            "{{ nextStep }}", "next step")

    # ---- 6c. THE REPORT DOCUMENT ------------------------------------------------------
    # Hard-coded end to end: a chief complaint, a clinical summary, vitals, POCUS and
    # laboratory values, a three-line differential, limitations and a next step. It is the
    # document a clinician would print, so a fixed one is the worst of the lot.
    s = sub(s, "Emergency Department · 24 Aug 2026, 09:20",
            "Emergency Department · {{ generatedAt }}", "report header date")
    s = sub(s, "Acute breathlessness, onset 3 hours before arrival.",
            "{{ pComplaint }}", "report complaint")
    s = sub(s, "The patient presents with acute breathlessness associated with tachycardia, "
               "tachypnea and reduced oxygen saturation. Lung ultrasound demonstrates B-lines "
               "in the bilateral anterior zones. Laboratory results show elevated BNP, "
               "troponin and lactate.",
            "{{ conclusion }}", "report summary")

    s = cut(s, '<div>SpO₂ 88% — low</div>', '<div>Blood pressure 104/68 — normal</div>',
            '<sc-for list="{{ vitals }}" as="v" hint-placeholder-count="5">'
            '<div>{{ v.label }} {{ v.value }} {{ v.unit }} — {{ v.flag }}</div></sc-for>',
            "report vitals")
    s = cut(s, '<div>Lung ultrasound: B-lines detected</div>',
            '<div>Lactate 2.6 mmol/L — high</div>',
            '<sc-for list="{{ findings }}" as="f" hint-placeholder-count="4">'
            '<div>{{ f.label }} — {{ f.status }} ({{ f.conf }})</div></sc-for>'
            '<sc-for list="{{ labs }}" as="l" hint-placeholder-count="8">'
            '<div>{{ l.name }} {{ l.result }} — {{ l.status }}</div></sc-for>',
            "report findings")
    s = cut(s, '<li>Pulmonary edema — moderate likelihood</li>',
            '<li>Pneumothorax — cannot be excluded by the current examination</li>',
            '<sc-for list="{{ ranked }}" as="r" hint-placeholder-count="3">'
            '<li>{{ r.name }} — {{ r.tag }} likelihood</li></sc-for>', "report differential")
    s = cut(s, '<li>D-dimer, creatinine and temperature were not obtained at the time of '
               'assessment</li>',
            '<li>Not measured does not mean normal</li>',
            '<sc-for list="{{ missing }}" as="m" hint-placeholder-count="3">'
            '<li>{{ m.name }} was not obtained at the time of assessment</li></sc-for>'
            '<sc-for list="{{ outOfScope }}" as="o" hint-placeholder-count="2">'
            '<li>{{ o.text }}</li></sc-for>'
            '<li>Not measured does not mean normal</li>', "report limitations")
    s = sub(s, "Physician review of the current findings, with consideration of the most "
               "clinically relevant missing investigations.",
            "{{ nextStep }}", "report next step")

    # ---- 6d. THE ALERTS SCREEN -------------------------------------------------------
    # Three fixed sections: a compromised oxygenation, a set of abnormal vitals and three
    # missing tests. They appeared whatever the patient's values were -- a gallbladder study
    # with no oxygen saturation recorded still announced SpO2 88%.
    s = re.sub(r"\d+ alerts require physician attention", "{{ alertHeadline }}", s)
    s = cut(s, '<section style="background:#fff;border:1px solid #F6CFCF;'
               'border-left:5px solid #E5484D;border-radius:18px;padding:22px 26px">\n'
               '    <div style="font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;'
               'color:#C13238;font-weight:800">Immediate attention</div>',
            '<p style="margin:14px 0 0;font-size:15px;color:#6A6785">Together, these findings '
            'increase clinical urgency.</p>\n  </section>',
            '<sc-for list="{{ alerts }}" as="a" hint-placeholder-count="4">\n'
            '  <section style="{{ a.style }}">\n'
            '    <div style="{{ a.kickerStyle }}">{{ a.kicker }}</div>\n'
            '    <h2 style="margin:10px 0 6px;font-size:21px;font-weight:800">{{ a.type }}'
            '</h2>\n'
            '    <p style="margin:0;font-size:15px;color:#6A6785">{{ a.message }}</p>\n'
            '    <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">\n'
            '      <button type="button" onClick="{{ goAssessment }}" style="appearance:none;'
            'border:1px solid #DEDCF4;background:#fff;cursor:pointer;color:#2E2A78;'
            'font-weight:700;font-size:13.5px;padding:10px 18px;border-radius:11px">'
            'Review patient</button>\n    </div>\n  </section>\n  </sc-for>\n'
            '  <sc-if value="{{ noAlerts }}" hint-placeholder-val="{{ false }}">\n'
            '  <section style="background:#F0FADB;border:1px solid #D5E8A8;'
            'border-left:5px solid #5A7A0F;border-radius:18px;padding:22px 26px">\n'
            '    <div style="font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;'
            'color:#5A7A0F;font-weight:800">No alert</div>\n'
            '    <p style="margin:10px 0 0;font-size:15px;color:#6A6785">No measured value '
            'crossed a configured bound and the record shows no structural gap.</p>\n'
            '  </section>\n  </sc-if>\n'
            '  <sc-if value="{{ hasEncounter }}" hint-placeholder-val="{{ true }}">\n'
            '  <section style="background:#fff;border:1px solid #DEDCF4;'
            'border-left:5px solid #7C5CFC;border-radius:18px;padding:22px 26px">\n'
            '    <div style="font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;'
            'color:#5B3CC4;font-weight:800">Escalation triggers</div>\n'
            '    <h2 style="margin:10px 0 10px;font-size:21px;font-weight:800">Evaluated '
            'before the model runs</h2>\n'
            '    <sc-for list="{{ triggers }}" as="t" hint-placeholder-count="3">'
            '<div style="font-size:15px;color:#6A6785;padding:4px 0">{{ t.text }}</div>'
            '</sc-for>\n  </section>\n  </sc-if>', "alerts sections")

    # ---- 7. roster table -------------------------------------------------------------
    s = cut(s, '<tbody>\n        <tr style="border-top:1px solid #E9E8FB">'
               '<td style="padding:14px 0;font-weight:600"><span style="display:flex;'
               'align-items:center;gap:11px"><image-slot id="pt-sarah-sm"',
            "</tbody>",
            '<tbody>\n        <sc-for list="{{ roster }}" as="r" hint-placeholder-count="5">\n'
            '        <tr style="border-top:1px solid #E9E8FB">'
            '<td style="padding:14px 0;font-weight:600">'
            '<span style="display:flex;align-items:center;gap:11px">'
            '<span style="width:30px;height:30px;border-radius:50%;background:#E4E2F8;'
            'color:#5B3CC4;display:inline-flex;align-items:center;justify-content:center;'
            'font-weight:800;font-size:11px;flex:0 0 auto">{{ r.initials }}</span>'
            '{{ r.name }}</span></td><td style="padding:14px 0">{{ r.age }}</td>'
            '<td style="padding:14px 0">{{ r.complaint }}</td>'
            '<td style="padding:14px 0"><span style="{{ r.tagStyle }}">{{ r.tag }}</span></td>'
            '<td style="padding:14px 0;text-align:right;color:#6A6785">{{ r.alerts }}</td>'
            '</tr>\n        </sc-for>\n'
            '        <sc-if value="{{ noRoster }}" hint-placeholder-val="{{ false }}">\n'
            '        <tr style="border-top:1px solid #E9E8FB"><td colspan="5" '
            'style="padding:18px 0;color:#6A6785;font-size:14.5px">No patient has been '
            'assessed in this session. Open Patient workup to enter one.</td></tr>\n'
            '        </sc-if>\n      </tbody>', "roster table")

    # ---- 7b. history grid: four invented patients become the real roster -------------
    s = cut(s, '<button type="button" onClick="{{ goRecord }}" style="text-align:left;'
               'appearance:none;cursor:pointer;background:#fff;border:1px solid #E4E2F8;'
               'border-top:4px solid #E5484D',
            '22 Aug · 11:44</span>\n    </div>',
            '<sc-for list="{{ roster }}" as="r" hint-placeholder-count="5">\n'
            '    <button type="button" onClick="{{ r.onOpen }}" style="text-align:left;'
            'appearance:none;cursor:pointer;background:#fff;border:1px solid #E4E2F8;'
            'border-radius:18px;padding:22px 24px;display:flex;flex-direction:column;gap:8px;'
            'box-shadow:0 6px 20px rgba(91,60,196,.05)">\n'
            '      <span style="{{ r.tagStyle }}">{{ r.tag }}</span>\n'
            '      <span style="display:flex;align-items:center;gap:12px;margin-top:6px">'
            '<span style="width:38px;height:38px;border-radius:50%;background:#E4E2F8;'
            'color:#5B3CC4;display:inline-flex;align-items:center;justify-content:center;'
            'font-weight:800;font-size:13px;flex:0 0 auto">{{ r.initials }}</span>'
            '<span style="font-weight:800;font-size:17px">{{ r.name }} · {{ r.age }}'
            '{{ r.sex }}</span></span>\n'
            '      <span style="font-size:14px;color:#6A6785">{{ r.complaint }}</span>\n'
            '      <span style="font-size:13px;color:#8A87A8;margin-top:6px">'
            '{{ r.severity }} · {{ r.alerts }}</span>\n'
            '    </button>\n    </sc-for>\n'
            '    <sc-if value="{{ noRoster }}" hint-placeholder-val="{{ false }}">\n'
            '    <div style="background:#fff;border:1px solid #E4E2F8;border-radius:18px;'
            'padding:22px 24px;color:#6A6785;font-size:14.5px">Nothing has been assessed in '
            'this session yet. There is no database behind this screen: closing the app '
            'discards it.</div>\n    </sc-if>', "history grid")

    # ---- 7b2. the assistant strip announced cases that did not exist -----------------
    # "3 previous cases are waiting for your review" survived every earlier pass because the
    # fabrication check looks for the mockup's patient, not for its numbers. A fixed count of
    # waiting cases is the same fault as the roster was: it invents work for the clinician.
    s = sub(s, '<div style="font-size:13.5px;color:#6A6785">3 previous cases are waiting '
               'for your review.</div>',
            '<div style="font-size:13.5px;color:#6A6785">{{ assistantLine }}</div>',
            "assistant strip")

    # ---- 7b3. the Diagnosis screen invented a diagnosis AND a percentage -------------
    # "⌁ Top match / PULMONARY EDEMA / 62.4 %" was displayed for a live patient, beside that
    # patient's real heart rate and real alert count. It is the worst thing the mockup carried
    # and it survived every earlier pass: a diagnosis nobody derived, given a precision this
    # system deliberately never produces. The reasoning agent emits a LIKELIHOOD BAND -- high,
    # moderate, low -- and never a number, because a percentage implies a calibration no part
    # of this pipeline has. Worse, the real values around it lend it their credibility.
    s = sub(s, '<div style="margin-top:18px;border:3px solid #5B54D6;padding:20px 14px;'
               'text-align:center;font-size:12.5px;font-weight:800;letter-spacing:.1em;'
               'text-transform:uppercase">Pulmonary edema</div>\n'
               '      <div style="margin-top:18px;display:flex;align-items:baseline;gap:6px">'
               '<span style="font-size:32px;font-weight:800;letter-spacing:-.03em">62.4</span>'
               '<span style="font-size:15px;color:#6A6785">%</span></div>',
            '<div style="margin-top:18px;border:3px solid #5B54D6;padding:20px 14px;'
            'text-align:center;font-size:12.5px;font-weight:800;letter-spacing:.1em;'
            'text-transform:uppercase">{{ topDiagnosis }}</div>\n'
            '      <div style="margin-top:18px;font-size:19px;font-weight:800">'
            '{{ topLikelihood }}</div>\n'
            '      <div style="margin-top:8px;font-size:12.5px;color:#6A6785;line-height:1.5">'
            '{{ topBasis }}</div>', "top match")

    s = sub(s, "You have 12 incomplete clinical tasks today",
            "{{ diagnosisLine }}", "diagnosis header")
    s = sub(s, "✦ 23 suggestions", "✦ {{ examCount }}", "suggestion count")
    s = sub(s, "⚠ 8 anomalies", "⚠ {{ anomalyCount }}", "anomaly count")

    # ---- 7b4. a revision of an assessment that was never revised ---------------------
    # The Timeline carried "Assessment updated -- Previously: Pulmonary edema, Moderate ->
    # Now: Moderate-high -- Why: Troponin 62 ng/L now available". A narrated change of mind,
    # with a diagnosis, two likelihoods and a reason, none of it derived. Nothing in this
    # system revises an assessment in place: re-analysing produces a NEW encounter, which is
    # deliberate, because a record that rewrites itself cannot be compared with what was shown
    # at the time. The screen says that instead of dramatising a revision.
    s = cut(s, '<div style="font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;'
               'color:#2E2A78;font-weight:800">Assessment updated</div>',
            "added to the differential</li></ul>\n    </div>",
            '<div style="font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;'
            'color:#2E2A78;font-weight:800">One encounter, computed once</div>\n'
            '    <p style="margin:10px 0 0;font-size:15.5px;line-height:1.6">'
            '{{ timelineNote }}</p>', "timeline revision")

    # ---- 7b5. the assistant announced a patient who was not on screen ----------------
    # "● 74F — acute breathlessness" was fixed text in the chat header. With a 71-year-old
    # open it stated her age wrongly, in the one panel whose whole claim is that it answers
    # from this encounter and invents nothing.
    s = sub(s, '● 74F — acute breathlessness', '● {{ assistantPatient }}', "assistant patient")

    # ---- 7b5b. the panel claiming to hold the encounter held four fixed numbers ------
    # "Vitals 4 recorded · Laboratory 3 of 5 · POCUS Lung · 2 clips · Assessment generated
    # 09:15", under a paragraph promising the assistant keeps everything entered for this
    # encounter. The one panel asserting that nothing needs repeating was itself repeating a
    # mockup.
    for label, old, new in (("Vitals", "4 recorded", "{{ ctxVitals }}"),
                            ("Laboratory", "3 of 5", "{{ ctxLabs }}"),
                            ("POCUS", "Lung · 2 clips", "{{ ctxPocus }}"),
                            ("Assessment", "Generated 09:15", "{{ ctxGenerated }}")):
        s = sub(s, f'<span style="color:#6A6785">{label}</span>'
                   f'<span style="font-weight:700">{old}</span>',
                f'<span style="color:#6A6785">{label}</span>'
                f'<span style="font-weight:700">{new}</span>', f"context {label}")

    # ---- 7b5c. the history box came PRE-FILLED with somebody else's comorbidities ----
    # The textarea's default content was "Hypertension, prior heart failure. On furosemide,
    # ramipril." -- not a placeholder, actual content, submitted with the encounter unless the
    # clinician noticed and cleared it. Every patient entered on this screen carried invented
    # comorbidities and two invented drugs into their record. It is bound and empty, with the
    # example demoted to a placeholder, where text is visibly not a value.
    s = sub(s, 'min-height:64px;resize:vertical;background:#FAFAFE">'
               'Hypertension, prior heart failure. On furosemide, ramipril.</textarea>',
            'min-height:64px;resize:vertical;background:#FAFAFE" value="{{ fHistory }}" '
            'onChange="{{ onHistory }}" placeholder="{{ historyPlaceholder }}"></textarea>',
            "history textarea")

    # ---- 7b5d. three investigations chosen by the mockup ----------------------------
    # "D-dimer — may contribute to assessing pulmonary embolism", "Creatinine — could guide
    # treatment and renal contribution", "Temperature — would help weigh an infective cause",
    # then "Obtain the missing high-priority investigations and reassess the respiratory
    # status". Fixed text under the heading "What would clarify this case", which is a clinical
    # recommendation for whichever patient was open. The support module ranks the real ones
    # with the reason each was ranked; those are shown.
    s = cut(s, '<div style="display:flex;gap:12px;padding:13px 15px;background:#FFF8EC;'
               'border-radius:10px"><span style="color:#E08A00">●</span><div>'
               '<div style="font-weight:700;font-size:14.5px">D-dimer</div>',
            "Would help weigh an infective cause.</div></div></div>",
            '<sc-for list="{{ exams }}" as="e" hint-placeholder-count="3">\n'
            '          <div style="display:flex;gap:12px;padding:13px 15px;'
            'background:#FFF8EC;border-radius:10px"><span style="color:#E08A00">●</span>'
            '<div><div style="font-weight:700;font-size:14.5px">{{ e.exam }} '
            '({{ e.priority }})</div><div style="font-size:13px;color:#6A6785">'
            '{{ e.reason }}</div></div></div>\n          </sc-for>\n'
            '          <sc-if value="{{ noExams }}" hint-placeholder-val="{{ false }}">\n'
            '          <div style="padding:13px 15px;font-size:13.5px;color:#6A6785">'
            '{{ noExamsNote }}</div>\n          </sc-if>', "clarify list")

    s = sub(s, "Obtain the missing high-priority investigations and reassess the "
               "respiratory status.", "{{ nextStep }}", "clarify next step")

    # ---- 7b5e. three chips naming the tests the mockup thought were missing ----------
    # Under the heading "Important investigations have not been performed", on the ALERTS
    # screen, a fixed D-dimer / Creatinine / Temperature. For a patient who had all three and
    # was missing others, it named the wrong ones, in the place a clinician looks to see what
    # is absent -- the exact fact the system is built to keep straight.
    s = cut(s, '<span style="background:#EFEEFB;color:#2E2A78;border-radius:999px;'
               'padding:6px 14px;font-size:13.5px;font-weight:700">D-dimer</span>',
            '>Temperature</span>',
            '<sc-for list="{{ missingChips }}" as="m" hint-placeholder-count="3">'
            '<span style="background:#EFEEFB;color:#2E2A78;border-radius:999px;'
            'padding:6px 14px;font-size:13.5px;font-weight:700">{{ m.name }}</span>'
            '</sc-for>\n      <sc-if value="{{ noMissing }}" '
            'hint-placeholder-val="{{ false }}">'
            '<span style="font-size:13.5px;color:#6A6785">{{ noMissingNote }}</span>'
            '</sc-if>', "missing chips")

    # ---- 7b7. the language switch ---------------------------------------------------
    # A link, not a control with state: each language is its own page, built and checked at
    # bind time, so switching is a navigation rather than a re-render. It sits in the top bar
    # beside the assistant chip, which is the first place a clinician who cannot read the
    # screen will look.
    s = sub(s, '<span class="livedot">●</span> Assistant ready</div>',
            '<span class="livedot">●</span> {{ assistantReady }}</div>\n'
            '    <a href="{{ otherLangHref }}" style="display:inline-flex;align-items:center;'
            'gap:6px;background:rgba(255,255,255,.14);color:#fff;border-radius:999px;'
            'padding:7px 13px;font-size:13px;font-weight:700;text-decoration:none">'
            '{{ otherLangLabel }}</a>', "language switch")

    # ---- 7b6. a caseload nobody had ------------------------------------------------
    s = sub(s, "12 assessments · 3 require review", "{{ historyLine }}", "history count")

    # ---- 7c. the lime tile counted conversations this system never had ---------------
    s = sub(s, '<span style="font-weight:700;font-size:15.5px">▤ Clinical assistant</span>',
            '<span style="font-weight:700;font-size:15.5px">▤ Evidence cited</span>',
            "lime tile title")
    s = sub(s, '<span style="font-size:36px;font-weight:800;letter-spacing:-.03em">7,198</span>'
               '<span style="font-size:14px;opacity:.9">conversations</span>',
            '<span style="font-size:36px;font-weight:800;letter-spacing:-.03em">'
            '{{ alertCount }}</span>'
            '<span style="font-size:14px;opacity:.9">alerts raised</span>', "lime tile count")

    # ---- 7d. the patient record is a LIST you open ------------------------------------
    # It showed one patient -- whoever was last analysed -- so there was no way to see who else
    # had been assessed, and the screen's name promised something the screen did not do. The
    # list comes first; opening a patient restores the encounter exactly as it was assessed.
    i = s.find('<sc-if value="{{ onRecord }}"')
    if i > 0:
        j = s.find(">", i) + 1
        # Where does the onRecord screen actually END? Counting tags is not enough -- balanced
        # totals hid a close placed inside the wrong block, which nested every screen after it
        # and rendered them empty. Walk the nesting instead.
        close = _match_close(s, j)
        if close < 0:
            print("FATAL: onRecord screen has no closing tag", file=sys.stderr)
            return 1
        s = s[:close] + "</sc-if>\n" + s[close:]
        s = (s[:j]
             + '\n<sc-if value="{{ recordList }}" hint-placeholder-val="{{ true }}">\n'
               '<div style="display:flex;flex-direction:column;gap:20px">\n'
               '  <div>\n'
               '    <h1 style="margin:0;font-size:27px;font-weight:800;'
               'letter-spacing:-.02em">Patients</h1>\n'
               '    <p style="margin:6px 0 0;color:#6A6785;font-size:15px">'
               '{{ patientCount }} assessed this session. Open one to see its case, findings, '
               'differential, timeline and report. There is no database behind this screen: '
               'closing the app discards it.</p>\n'
               '  </div>\n'
               '  <sc-if value="{{ noPatients }}" hint-placeholder-val="{{ false }}">\n'
               '  <div style="background:#fff;border:1px solid #E4E2F8;border-radius:10px;'
               'padding:26px 30px;color:#6A6785;font-size:15px">No patient has been assessed '
               'yet. Open Patient workup, enter what you have and analyse the case.</div>\n'
               '  </sc-if>\n'
               '  <div style="display:grid;grid-template-columns:repeat(auto-fit,'
               'minmax(300px,1fr));gap:18px">\n'
               '    <sc-for list="{{ patients }}" as="p" hint-placeholder-count="4">\n'
               '    <button type="button" onClick="{{ p.onOpen }}" style="text-align:left;'
               'appearance:none;cursor:pointer;background:#fff;border:1px solid #E4E2F8;'
               'border-radius:14px;padding:22px 24px;display:flex;flex-direction:column;'
               'gap:10px;box-shadow:0 6px 20px rgba(91,60,196,.05)">\n'
               '      <span style="display:flex;align-items:center;gap:12px">'
               '<span style="width:40px;height:40px;border-radius:50%;background:#E4E2F8;'
               'color:#5B3CC4;display:inline-flex;align-items:center;justify-content:center;'
               'font-weight:800;font-size:13px;flex:0 0 auto">{{ p.initials }}</span>'
               '<span><span style="font-weight:800;font-size:17px;display:block">{{ p.name }}'
               '</span><span style="font-size:13px;color:#8A87A8">{{ p.age }} · {{ p.sex }} · '
               '{{ p.at }}</span></span></span>\n'
               '      <span style="font-size:14px;color:#6A6785">{{ p.complaint }}</span>\n'
               '      <span style="font-size:13px;color:#8A87A8">{{ p.organ }} — '
               '{{ p.findings }}</span>\n'
               '      <span style="display:flex;gap:8px;align-items:center">'
               '<span style="{{ p.tagStyle }}">{{ p.severity }}</span>'
               '<span style="font-size:12.5px;color:#8A87A8">{{ p.alerts }}</span></span>\n'
               '    </button>\n    </sc-for>\n  </div>\n</div>\n</sc-if>\n'
               '<sc-if value="{{ recordOpen }}" hint-placeholder-val="{{ true }}">\n'
               '<button type="button" onClick="{{ backToList }}" style="appearance:none;'
               'border:1px solid #DEDCF4;background:#fff;cursor:pointer;color:#5B3CC4;'
               'font-weight:700;font-size:13.5px;padding:9px 15px;border-radius:9px;'
               'margin-bottom:16px">← All patients</button>\n'
             + s[j:])
        applied.append("patient list")
    else:
        missed.append("patient list")

    # ---- 8. patient-record stats imply a database ------------------------------------
    ST = ('<div><div style="color:#6A6785">{k}</div><div style="font-weight:800;'
          'font-size:18px;margin-top:3px">{v}</div></div>')
    for k, val, nk, nv in (("Visits", "4", "Encounters", "{{ recEncounters }}"),
                           ("POCUS studies", "7", "POCUS studies", "{{ recStudies }}"),
                           ("Reports", "3", "Alerts", "{{ alertCount }}"),
                           ("Last seen", "Today", "Severity", "{{ severity }}")):
        s = sub(s, ST.format(k=k, v=val), ST.format(k=nk, v=nv), f"record stat {k}")
    s = re.sub(r"\d+ studies · grouped by visit",
               "session-scoped · closing the app discards it", s)

    # ---- 8b. the record's "add images" tile was decorative too ------------------------
    s = sub(s, '<div style="font-weight:700;font-size:15px">＋ Add images to this patient</div>\n'
               '      <div style="margin-top:5px;font-size:13.5px;color:#6A6785">Uploads are '
               'stored against the record and stay available at the next visit.</div>',
            '<label style="font-weight:700;font-size:15px;cursor:pointer;color:#5B54D6">'
            '＋ Add images to this patient'
            '<input type="file" accept="image/*" multiple onChange="{{ onRecordUpload }}" '
            'style="display:none"></label>\n'
            '      <div style="margin-top:5px;font-size:13.5px;color:#6A6785">Read by the '
            'module now and filed against this patient, for this session only — there is no '
            'database behind this screen. The assessment above is not re-run: it was reached '
            'without this study.</div>', "record upload")

    # ---- 8b2. three prior visits this patient never had ------------------------------
    # The Reports tab listed "12 Jun 2026 · Exertional breathlessness · Discharged with
    # follow-up" and "3 Mar 2026 · Ankle swelling · Routine review" under whichever real
    # patient was open: a fabricated medical history attached to a named person, which is the
    # most consequential kind of invention in the whole interface. It lists this session's
    # encounters for that patient, and says plainly when there is only the one.
    s = cut(s, '<button type="button" onClick="{{ goReport }}" style="text-align:left;'
               'appearance:none;cursor:pointer;background:#fff;border:1px solid #E4E2F8;'
               'border-radius:8px;padding:20px 22px;display:flex;flex-direction:column;'
               'gap:7px">\n        <span style="font-size:12.5px;color:#6A6785;'
               'font-weight:700">24 Aug 2026 · 09:20</span>',
            '<span style="font-size:13.5px;color:#6A6785">Routine review</span>\n'
            '      </div>\n    </div>\n  </section>',
            '<sc-for list="{{ patientReports }}" as="r" hint-placeholder-count="3">\n'
            '      <button type="button" onClick="{{ r.onOpen }}" style="text-align:left;'
            'appearance:none;cursor:pointer;background:#fff;border:1px solid #E4E2F8;'
            'border-radius:8px;padding:20px 22px;display:flex;flex-direction:column;gap:7px">\n'
            '        <span style="font-size:12.5px;color:#6A6785;font-weight:700">{{ r.at }}'
            '</span>\n'
            '        <span style="font-weight:800;font-size:16px">{{ r.complaint }}</span>\n'
            '        <span style="font-size:13.5px;color:#6A6785">{{ r.summary }}</span>\n'
            '        <span style="margin-top:6px;color:#2E2A78;font-weight:700;'
            'font-size:13.5px">Open →</span>\n      </button>\n      </sc-for>\n'
            '      <sc-if value="{{ oneReportOnly }}" hint-placeholder-val="{{ false }}">\n'
            '      <div style="background:#fff;border:1px solid #E4E2F8;border-radius:8px;'
            'padding:20px 22px;color:#6A6785;font-size:13.5px">This session holds no earlier '
            'encounter for this patient. There is no database behind this screen: it shows '
            'what was assessed here, not a medical history.</div>\n      </sc-if>\n'
            '    </div>\n  </section>', "stored reports")

    # ---- 8b3. four columns of visits that never happened -----------------------------
    # The measurements table was headed Today / 12 Jun / 3 Mar / 18 Jan. The three older
    # columns held only dashes, which reads as "measured then, not now" rather than "these
    # visits are invented" -- a dash under a date is a claim that the date existed.
    for old, new in (("12 Jun", "{{ visitCol2 }}"), ("3 Mar", "{{ visitCol3 }}"),
                     ("18 Jan", "{{ visitCol4 }}")):
        s = sub(s, f'>{old}</th>', f'>{new}</th>', f"visit column {old}")

    # ---- 8c. the stored studies were drop targets, never the patient's images ---------
    # The grid looped the mockup's visits and drew an `image-slot` in each cell: a slot the
    # designer drops a picture into, which lives in the design file and never reaches the app.
    # So a study the doctor had just uploaded and had read to them was not in the record of the
    # patient it was read for. It loops the studies the module actually read instead.
    s = sub(s, '<sc-for list="{{ visits }}" as="v" hint-placeholder-count="3">\n'
               '      <div style="margin-top:26px">',
            '<sc-for list="{{ gallery }}" as="v" hint-placeholder-count="3">\n'
            '      <div style="margin-top:26px">', "image gallery list")
    s = sub(s, '<image-slot id="{{ img.slotId }}" shape="rect" '
               'placeholder="{{ img.zone }}"></image-slot>',
            '<img src="{{ img.src }}" alt="{{ img.zone }}" style="width:100%;height:100%;'
            'object-fit:contain;display:block;background:#141824">', "stored image")
    s = sub(s, '<sc-for list="{{ gallery }}" as="v" hint-placeholder-count="3">',
            '<sc-if value="{{ noImages }}" hint-placeholder-val="{{ false }}">\n'
            '    <div style="margin-top:22px;border:1px solid #E4E2F8;border-radius:8px;'
            'padding:20px;color:#6A6785;font-size:14.5px">No study has been read for this '
            'patient. Uploading one on Patient workup files it here with what the module '
            'made of it.</div>\n    </sc-if>\n'
            '    <sc-for list="{{ gallery }}" as="v" hint-placeholder-count="3">',
            "no stored images")

    io.open(OUT, "w", encoding="utf8", newline="\n").write(s)

    # ---- 9. the same page in French --------------------------------------------------
    # A second page rather than a runtime translation: it is checked by the same balance and
    # fabrication checks as the English one, costs nothing to serve, and the language is
    # settled before any script runs instead of being swapped afterwards.
    #
    # The coverage check is the point. A sentence added to the design and forgotten here would
    # reach a French-speaking clinician in English, which is the failure this exists to
    # prevent, so an uncovered string fails the build rather than shipping quietly.
    visible = _visible_strings(s)
    uncovered = fr_ui.assert_covered(visible)
    # The constant itself, not an injected <script>: the export has no plain script tag to
    # inject into, so the first attempt silently left the French page running in English --
    # every clinical string translated by the server, every nav label still English.
    fr = fr_ui.translate(s)
    old_lang = "const LANG = (typeof window !== 'undefined' && window.LANG) || 'en';"
    if old_lang not in fr:
        print("FATAL: the LANG constant moved; the French page would run in English",
              file=sys.stderr)
        return 1
    fr = fr.replace(old_lang, "const LANG = 'fr';")
    io.open(OUT_FR, "w", encoding="utf8", newline="\n").write(fr)
    print(f"wrote {OUT_FR.relative_to(ROOT)}  ({len(fr):,} chars)")
    print(f"french coverage: {len(visible) - len(uncovered)}/{len(visible)} visible strings")
    if uncovered:
        print(f"\n{len(uncovered)} STRING(S) WITH NO FRENCH FORM:")
        for u in uncovered:
            print("   -", u[:100])
        return 1

    # ---- report ---------------------------------------------------------------------
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(s):,} chars)")
    print(f"applied {len(applied)} rewrites")
    if missed:
        print(f"\n{len(missed)} DID NOT MATCH — the export has moved on and these need "
              f"re-anchoring:")
        for m in missed:
            print("   -", m)

    left = [m for m in ("Sarah Martin", "Dr. Reyes", "Patient B", "7,198", "MRN 4471")
            if m in s]
    print("\nfabricated content remaining:", ", ".join(left) if left else "none")

    for tag in ("sc-for", "sc-if"):
        o = len(re.findall(rf"<{tag}\b", s))
        c = len(re.findall(rf"</{tag}>", s))
        print(f"{tag}: {o} open / {c} close" + ("" if o == c else "   *** UNBALANCED ***"))
        if o != c:
            return 1
    return 0 if not left else 1


if __name__ == "__main__":
    raise SystemExit(main())
