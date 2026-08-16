# Ultrasound Agent

The router and the per-organ inference live in `notebooks/ultrasound_agent.ipynb`, not in
this package. That is deliberate for now: the modules load PyTorch checkpoints from Google
Drive and run on a GPU runtime, so the notebook is where they are actually executed.

What the agent does:

| organ | model | output |
|---|---|---|
| lung | EfficientNet-B0, multi-label | four findings, each detected or `not_detected`, with calibrated confidence |
| heart | U-Net + EfficientNet-B0 encoder | segmentation → ejection fraction → severity |
| gallbladder | EfficientNet-B0 | one pathology class of five |
| vascular, FAST | — | `status: "not_supported"`, in the ordinary format |

Everything it emits goes through `src/agents/schema.py`, which is why this directory is a
package rather than a note: the contract is shared, and the Clinical Reasoning Agent reads
what is written through it without knowing which module produced it.

**The distinction this agent has to preserve.** A finding it screened for and did not see is
recorded in `not_detected`. An organ it never assessed is absent from the report entirely.
Those are different clinical statements, and collapsing them was a real defect: the lung
module once represented "nothing above threshold" with a placeholder entry in `findings`,
which the state read as a positive finding and which suppressed the escalation that should
have followed.

**To extract this into Python** — worth doing once the checkpoints are in a fixed location —
port `predict_lung`, `predict_heart`, `predict_gallbladder` and the router from the notebook
into `agent.py` here, keeping `make_report` as the only way a result leaves the module.
