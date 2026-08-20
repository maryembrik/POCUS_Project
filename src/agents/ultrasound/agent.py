"""Ultrasound Agent -- per-organ inference, extracted from the notebooks.

Until now the per-organ prediction code lived only in `notebooks/ultrasound_agent.ipynb`, so
nothing outside a Colab session could run it and no test could import it. The suite proved the
*contract* was enforced; it could not prove this agent honoured it. That gap is stated as a
limitation in the report, and this module closes it for the lung module, whose checkpoints are
in the repository.

What is here and what is not:

    lung          real inference. Three fold checkpoints ship with the repository, the
                  architecture is loaded strict=True against them, and predictions are made on
                  CPU in about a second per image.
    heart         not available. The CAMUS-trained weights are not in the repository, so the
                  router returns `not_supported` rather than a guess.
    gallbladder   the same.

Calibration is the honest wrinkle, and it has two halves that were NOT lost together.

    thresholds    RECOVERED. The per-split operating points chosen by F1 tuning on the
                  validation fold are recorded in `results_efficientnet_b0_split{n}.json`, and
                  this module reads them: 0.30 / 0.20 / 0.35 / 0.45, not 0.5. The distinction
                  is not cosmetic -- this model's outputs are compressed into roughly 0.2-0.8,
                  so at 0.5 it fires on almost nothing and every scan reads as all-negative.
                  The notebook has the same fallback and the same defect.

    Platt scaling LOST. The calibrators were fitted inside the training notebook and never
                  written to `lung_calibration.json`, so confidence is a **raw sigmoid output**
                  and `confidence_calibrated` is false. That is not a workaround -- it is the
                  behaviour the schema was designed for, and the reasoning layer discounts an
                  uncalibrated confidence rather than reading it as a probability.

A tuned threshold is not calibration, and reporting one as the other would be the same mistake
in the other direction: it says where the decision boundary sits, not what the number means.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .. import schema as S

LUNG_FINDINGS = ["b_lines", "consolidation", "pleural_effusion", "pleural_thickening"]
IMG_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226

ROOT = Path(__file__).resolve().parents[3]
LUNG_WEIGHTS = sorted((ROOT / "Pulmonary").glob(
    "lung_finding_classifier_efficientnet_b0_split*_best.pth"))
LUNG_CALIBRATION = ROOT / "Pulmonary" / "lung_calibration.json"
LUNG_RESULTS = ROOT / "Pulmonary" / "results_efficientnet_b0_split{fold}.json"

# Only reached when neither the calibration file nor the per-split results are on disk. 0.5 is
# the neutral choice and is wrong for this model -- its outputs sit in roughly 0.2-0.8, so at
# 0.5 almost nothing fires. It is a last resort, and `thresholds_source` says when it was used.
DEFAULT_THRESHOLD = 0.5

# A finding whose tuned F1 falls below this is reported but flagged `unreliable`, so the
# reasoning layer can weigh it accordingly. Pleural effusion is the one that fails: 24 positive
# clips in the training data, F1 0.44, and it is named in the report's limitations.
UNRELIABLE_F1 = 0.5

_MODELS: dict[str, Any] = {}


def _torch():
    import torch  # noqa: PLC0415
    return torch


def available() -> dict[str, bool]:
    """Which organs this deployment can actually run."""
    return {"lung": bool(LUNG_WEIGHTS), "heart": False, "gallbladder": False}


def _build_lung_model():
    """The architecture the shipped checkpoints were saved from.

    Recovered from the checkpoint rather than from the notebook, because they disagree: the
    notebook's head is a two-layer MLP (`head.1`, `head.4`), while every shipped checkpoint has
    a single `head.weight` of shape (4, 1280). Loading is `strict=True` so that a future
    mismatch fails loudly instead of silently leaving layers at their initialisation.
    """
    import torch.nn as nn
    import torchvision

    e = torchvision.models.efficientnet_b0(weights=None)

    class LungFindingClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(e.features, e.avgpool)
            self.head = nn.Linear(e.classifier[1].in_features, len(LUNG_FINDINGS))

        def forward(self, x):
            return self.head(self.features(x).flatten(1))

    return LungFindingClassifier()


def load_lung(fold: int = 0):
    """Load one fold's model, cached."""
    key = f"lung{fold}"
    if key in _MODELS:
        return _MODELS[key]
    if not LUNG_WEIGHTS:
        raise FileNotFoundError(f"no lung checkpoint under {ROOT / 'Pulmonary'}")
    torch = _torch()
    model = _build_lung_model()
    model.load_state_dict(torch.load(LUNG_WEIGHTS[fold % len(LUNG_WEIGHTS)],
                                     map_location="cpu"))
    model.eval()
    _MODELS[key] = model
    return model


def _prep(image) -> Any:
    """One image to a normalised 3-channel tensor.

    Mirrors the notebook's preprocessing: resize to 224 with INTER_AREA, convert to a single
    channel, scale to 0-1 only when the input is 0-255, normalise with the ImageNet grey
    statistics, and repeat to three channels because the backbone expects them.
    """
    import cv2
    import numpy as np
    torch = _torch()

    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    a = cv2.resize(a, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    a = a.astype("float32")
    if a.max() > 1.5:
        a = a / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(a).float().unsqueeze(0).repeat(3, 1, 1)


def _calibration() -> dict[str, Any] | None:
    if LUNG_CALIBRATION.exists():
        return json.loads(LUNG_CALIBRATION.read_text(encoding="utf8"))
    return None


def _tuned(fold: int) -> dict[str, Any]:
    """Operating points and reliability flags recovered from the recorded per-split results.

    The training run wrote its F1-tuned thresholds and per-finding metrics to disk even though
    it never exported the calibrators. Reading them back is not a substitute for calibration --
    it only puts the decision boundary where the training run put it, instead of at an
    arbitrary 0.5 that would silence the model on almost every scan.
    """
    path = Path(str(LUNG_RESULTS).format(fold=fold % max(len(LUNG_WEIGHTS), 1)))
    if not path.exists():
        return {"thresholds": {}, "unreliable": ["pleural_effusion"], "source": "default 0.5"}
    r = json.loads(path.read_text(encoding="utf8"))
    strip = lambda d: {k.replace("finding_", ""): v for k, v in (d or {}).items()}  # noqa: E731
    f1 = strip(r.get("f1_tuned"))
    return {"thresholds": strip(r.get("thresholds")),
            "unreliable": sorted(k for k, v in f1.items() if v < UNRELIABLE_F1),
            "auroc": strip(r.get("auroc")),
            "source": path.name}


def predict_lung(frames: Iterable, fold: int = 0) -> dict[str, Any]:
    """Findings for one clip, or for a single still passed as a one-item list.

    Frame probabilities reduce to one score per clip by the MEDIAN, matching how the module was
    evaluated. Frames from a clip are repeated looks at the same anatomy rather than independent
    evidence, so a mean would let one blurred frame carry the result.
    """
    import numpy as np
    torch = _torch()

    frames = list(frames)
    if not frames:
        return S.make_report("lung", [], status="failed",
                             reliability={"scope": "no frame supplied"})

    model = load_lung(fold)
    xs = torch.stack([_prep(f) for f in frames])
    with torch.no_grad():
        p_frame = torch.sigmoid(model(xs)).cpu().numpy()
    p_clip = np.median(p_frame, axis=0)

    cal = _calibration()
    tuned = _tuned(fold)
    thresholds = (cal or {}).get("thresholds") or tuned["thresholds"]
    unreliable = (cal or {}).get("unreliable_findings") or tuned["unreliable"]

    detected, not_detected = [], []
    for i, name in enumerate(LUNG_FINDINGS):
        conf = float(p_clip[i])
        entry = S.make_finding(name.replace("_", " "), conf)
        if name in unreliable:
            entry["unreliable"] = True
        thr = thresholds.get(name, DEFAULT_THRESHOLD)
        (detected if conf > thr else not_detected).append(entry)

    # An all-negative clip is an informative result, represented as an empty `findings` beside
    # a populated `not_detected`. It must never be a placeholder entry in `findings`: everything
    # there is marked detected downstream, so a sentinel would make the ABSENCE of findings read
    # as a positive one and suppress the escalation that should follow.
    return S.make_report(
        "lung",
        sorted(detected, key=lambda f: -f["confidence"]),
        not_detected=sorted(not_detected, key=lambda f: -f["confidence"]),
        status="ok",
        quality={"no_finding_above_threshold": not detected,
                 "frames": len(frames),
                 "thresholds_source": "lung_calibration.json" if cal else tuned["source"],
                 "thresholds": {k: thresholds.get(k, DEFAULT_THRESHOLD)
                                for k in LUNG_FINDINGS}},
        reliability={
            "confidence_calibrated": bool(cal),
            "has_normal_class": False,
            "modelled_findings": LUNG_FINDINGS,
            "unreliable_findings": unreliable,
            "auroc": tuned.get("auroc"),
            "scope": "187 clips / 165 cases; pneumothorax is NOT modelled and cannot be "
                     "excluded" + ("" if cal else "; confidence is a RAW sigmoid output, not a "
                                                  "calibrated probability -- the decision "
                                                  "boundary is tuned, the number is not"),
        },
        model=f"effnetb0_multilabel_fold{fold}")


def ultrasound_agent(organ: str, **kwargs) -> dict[str, Any]:
    """One entry point. An organ without a usable model returns `not_supported` in the ordinary
    format, so the reasoning layer never branches on whether a key exists."""
    organ = organ.lower()
    if organ not in S.ORGANS:
        raise ValueError(f"unknown organ {organ!r}; expected one of {sorted(S.ORGANS)}")

    if organ == "lung" and LUNG_WEIGHTS:
        return predict_lung(kwargs.get("frames") or [kwargs["image"]],
                            fold=kwargs.get("fold", 0))

    reason = ("weights are not present in this deployment"
              if organ in ("heart", "gallbladder") else "module not implemented")
    return S.make_report(organ, [], status="not_supported",
                         reliability={"scope": reason})
