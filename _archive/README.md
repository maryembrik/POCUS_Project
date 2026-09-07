# Archive

Superseded code, kept as a record of how the project got where it is. Nothing here runs, and
nothing outside this directory imports from it.

## `app.py` — the Streamlit prototype

The first clinician-facing interface. It was the working application until late August 2026,
when the interface was rebuilt on the design export and the deployment moved to
[`serve.py`](../serve.py): a FastAPI service run by uvicorn.

**Do not run this file.** Two reasons, and the second is the serious one:

1. It is stale. Its last change was 25 August 2026; the deployment has moved on.

2. **It still contains screens that display invented content.** The rebuilt interface had six
   such faults removed — among them a differential-diagnosis panel that showed a confident
   "top match" with a percentage for patients whose differential had never been generated, a
   fabricated medical history attached to real patient names, and a history field pre-filled
   with comorbidities nobody had entered. Those fixes were made in `serve.py`, `web/logic.js`
   and `tools/bind_design.py`. They were **not** back-ported here, because this file was
   already superseded when they were found.

   A screen that invents a diagnosis is worse than a screen that shows nothing, and this one
   invents them under real patients' names. That is why the file is archived rather than left
   at the repository root where a reader might reasonably try `streamlit run app.py`.

It is kept rather than deleted because it is the honest record of an iteration — the report
describes the move from it to the current interface — and because the git history of the
current work runs through it.

The port is the one thing that survived: 8501 is Streamlit's default, and `serve.py` kept the
number so existing links continued to work. That is the whole reason the current FastAPI
deployment sits on a port people associate with Streamlit.
