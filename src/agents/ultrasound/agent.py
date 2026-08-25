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

# The gallbladder module is single-label over five classes, unlike the lung module's four
# independent findings. It predicts eight fine-grained classes and MARGINALISES to these five
# rather than taking the argmax and mapping it -- merging the predictions scored 63.1% against
# 57.2% for merging the labels, because a case that splits its mass across cholecystitis,
# gangrene and perforation is one confident inflammation, not three uncertain diagnoses.
#
# Declared here rather than in the notebook so the contract has one importable home. The
# weights are not in this deployment, so nothing here runs inference yet; the names and groups
# are what a report must carry when it does.
GB_CLASSES = ["Cholelithiasis", "Acute cholecystitis (any severity)",
              "Polyps / adenomyomatosis", "Carcinoma", "Wall thickening"]
GB_GROUP = {"Cholelithiasis": "Cholelithiasis",
            "Acute cholecystitis (any severity)": "Acute inflammation",
            "Polyps / adenomyomatosis": "Wall thickening / mass",
            "Carcinoma": "Wall thickening / mass",
            "Wall thickening": "Wall thickening / mass"}
GB_SCOPE = "teaching-atlas stills; no healthy class exists in the training data"
# Below this the module's own notebook marks the read low-confidence rather than presenting it
# as a call. Five classes on 199 cases at 59.7% balanced accuracy does not support more.
GB_LOW_CONFIDENCE = 0.4

IMG_SIZE = 224
IMG_SIZE_SEG = 256
IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226

# The gallbladder model predicts eight fine-grained classes and MARGINALISES to the five above
# rather than taking the argmax and mapping it. Merging the predictions scored 63.1% against
# 57.2% for merging the labels: a case that splits its mass across cholecystitis, gangrene and
# perforation is one confident inflammation, not three uncertain diagnoses.
GB_TRAIN_NAMES = ["Gallstones", "Cholecystitis", "Membranous / gangrenous", "Perforation",
                  "Polyps", "Adenomyomatosis", "Carcinoma", "Wall thickening"]
GB_TRAIN_TO_EVAL = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 4}

# Ejection fraction bands, and the label each becomes in a report.
EF_FINDING = {"normal": "Normal ventricular function",
              "reduced": "Mild-to-moderate LV dysfunction",
              "severe": "Severe LV dysfunction"}

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


# Where the training notebooks write the other two modules' weights. Both are saved to Google
# Drive from Colab and are not in this repository, so these paths normally do not exist. They
# are probed rather than assumed so that a checkout which DOES have them reports the true
# reason it still cannot run them.
HEART_WEIGHTS = ROOT / "Cardiac" / "cardiac_effnet_unet_best.pth"
HEART_CALIBRATION = ROOT / "Cardiac" / "cardiac_calibration.json"
GB_WEIGHTS = ROOT / "Abdominal_Gallbladder" / "gallbladder_effnetb0_best.pth"
GB_CALIBRATION = ROOT / "Abdominal_Gallbladder" / "gallbladder_calibration.json"


def _json(path: Path) -> dict[str, Any] | None:
    """A calibration artefact, or None if it was never exported."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    return None


def module_status() -> dict[str, dict[str, Any]]:
    """Whether each organ can run, and — when it cannot — which of the two reasons applies.

    The distinction matters and was previously collapsed. `available()` hard-coded heart and
    gallbladder to False, so a checkout that had downloaded their weights would still be told
    "weights absent", which is a false explanation of a true refusal. There are two separate
    obstacles and they have different fixes:

        weights absent      the checkpoint is not on disk. Download it from the training run.
        not implemented     the checkpoint is on disk, but inference for that organ has not
                            been extracted from the notebook yet, so nothing here can call it.

    Only the lung module has cleared both.
    """
    def state(weights: bool, calib: Path) -> dict[str, Any]:
        if not weights:
            return {"runs": False, "weights": False, "calibrated": False,
                    "reason": "weights absent"}
        cal = calib.exists()
        return {"runs": True, "weights": True, "calibrated": cal,
                "reason": "ready" if cal else "ready, uncalibrated"}

    return {
        "lung": state(bool(LUNG_WEIGHTS), LUNG_CALIBRATION),
        "heart": state(HEART_WEIGHTS.exists(), HEART_CALIBRATION),
        "gallbladder": state(GB_WEIGHTS.exists(), GB_CALIBRATION),
    }


def available() -> dict[str, bool]:
    """Which organs this deployment can actually run. See `module_status` for the reason."""
    return {k: v["runs"] for k, v in module_status().items()}


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


# ═══════════════════════════════════════════════════════════════════ gallbladder
def load_gallbladder():
    if "gallbladder" in _MODELS:
        return _MODELS["gallbladder"]
    if not GB_WEIGHTS.exists():
        raise FileNotFoundError(f"no gallbladder checkpoint at {GB_WEIGHTS}")
    import torch.nn as nn
    import torchvision
    torch = _torch()
    m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(GB_TRAIN_NAMES))
    m.load_state_dict(torch.load(GB_WEIGHTS, map_location="cpu"))
    m.eval()
    _MODELS["gallbladder"] = m
    return m


def predict_gallbladder(image) -> dict[str, Any]:
    """One still to one of five classes.

    Single-label, unlike the lung module's four independent findings. Exactly one class is
    reported and `not_detected` stays empty: the four that lost the argmax were not screened
    out, they simply were not the winner, and listing them as negatives would be a stronger
    claim than the model made.
    """
    import numpy as np
    torch = _torch()

    model = load_gallbladder()
    with torch.no_grad():
        logits = model(_prep(image).unsqueeze(0)).cpu().numpy()[0]

    cal = _json(GB_CALIBRATION)
    if cal and cal.get("temperature"):
        p8 = torch.softmax(torch.from_numpy(logits) / float(cal["temperature"]),
                           dim=-1).numpy()
    else:
        p8 = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    p5 = np.zeros(len(GB_CLASSES), dtype=np.float32)
    for t, e in GB_TRAIN_TO_EVAL.items():          # marginalise, do not argmax-then-map
        p5[e] += p8[t]
    c = int(p5.argmax())
    conf = float(p5[c])

    return S.make_report(
        "gallbladder",
        [S.make_finding(GB_CLASSES[c], conf, group=GB_GROUP[GB_CLASSES[c]])],
        quality={"low_confidence": conf < GB_LOW_CONFIDENCE,
                 "fine_grained": GB_TRAIN_NAMES[int(p8.argmax())]},
        reliability={"confidence_calibrated": bool(cal),
                     "has_normal_class": False,
                     "modelled_findings": GB_CLASSES,
                     "ece": (cal or {}).get("ece"),
                     "scope": GB_SCOPE},
        model="effnetb0_8class_marginalised")


# ══════════════════════════════════════════════════════════════════════ cardiac
def load_heart():
    if "heart" in _MODELS:
        return _MODELS["heart"]
    if not HEART_WEIGHTS.exists():
        raise FileNotFoundError(f"no cardiac checkpoint at {HEART_WEIGHTS}")
    import segmentation_models_pytorch as smp
    torch = _torch()
    m = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=1,
                 classes=4)
    m.load_state_dict(torch.load(HEART_WEIGHTS, map_location="cpu"))
    m.eval()
    _MODELS["heart"] = m
    return m


def _prep_seg(image):
    import cv2
    import numpy as np
    a = cv2.resize(np.asarray(image, dtype=np.float32), (IMG_SIZE_SEG, IMG_SIZE_SEG),
                   interpolation=cv2.INTER_LINEAR)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if a.max() > 1.5:
        a = a / 255.0
    return a.astype("float32")


def _tta(img) -> list:
    """Eight fixed augmentations, deliberately fixed rather than random.

    Confidence is the fraction of them that agree on the reported band, so a random set would
    make the same clip yield a different number on every run.
    """
    import cv2
    import numpy as np
    h, w = img.shape
    out = [img.copy()]
    for ang in (3.0, -3.0):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out.append(cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    for b in (1.05, 0.95):
        out.append(np.clip(img * b, 0, 1))
    for c in (1.05, 0.95):
        m = img.mean()
        out.append(np.clip((img - m) * c + m, 0, 1))
    out.append(np.fliplr(img).copy())
    return [a.astype("float32") for a in out]


def _ef_band(ef: float) -> str:
    return "normal" if ef >= 55 else ("reduced" if ef >= 30 else "severe")


def predict_heart(ed_image, es_image) -> dict[str, Any]:
    """Ejection-fraction band from two frames: end-diastole and end-systole.

    EF is a comparison between the two, so one still cannot produce it -- the module segments
    the left ventricle in each and derives the fractional change.

    The area-to-volume correction matters and is not cosmetic. Raw area understates the
    fractional change because volume scales as area^(3/2); correcting it moved MAE from 12.7
    to 7.1 percentage points in the training run.
    """
    import numpy as np
    torch = _torch()

    model = load_heart()

    def lv_area(a) -> float:
        x = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return float((model(x).argmax(1).squeeze(0).cpu().numpy() == 1).sum())

    ed, es = _prep_seg(ed_image), _prep_seg(es_image)
    views = list(zip(_tta(ed), _tta(es)))
    efs = []
    for a, b in views:
        a_ed, a_es = lv_area(a), lv_area(b)
        # BOTH areas must be non-zero. Guarding only the diastolic one lets a systolic frame
        # where the segmentation found nothing through as an area of zero, which the formula
        # reads as a ventricle that ejected all of its volume: EF 100%, reported as "normal
        # ventricular function" at the calibrator's maximum confidence. A frame where the
        # ventricle cannot be found is a failed measurement, not a measurement of zero, and
        # this is the direction that matters -- the invented value was reassuring.
        if a_ed > 0 and a_es > 0:
            r = min(max(1.0 - (a_ed - a_es) / a_ed, 1e-6), 1.0)
            efs.append((1.0 - r ** 1.5) * 100.0)

    if len(efs) < len(views) / 2:
        # Fewer than half the augmented views yielded a usable pair. Reporting a band from the
        # remainder would rest the reading on whichever views happened to segment.
        return S.make_report(
            "heart", [], status="failed",
            quality={"lv_detected": bool(efs), "usable_views": len(efs),
                     "views": len(views)},
            reliability={"scope": f"left ventricle segmented in only {len(efs)} of "
                                  f"{len(views)} views; too few for a reliable band"})

    median_ef = float(np.median(efs))
    if median_ef > 85.0:
        # No ventricle ejects this fraction. A value here means the systolic segmentation
        # collapsed rather than that the heart is hyperdynamic, and the honest report is that
        # the measurement failed.
        return S.make_report(
            "heart", [], status="failed",
            quality={"lv_detected": True, "implausible_ef": round(median_ef, 1)},
            reliability={"scope": f"derived ejection fraction of {median_ef:.0f}% is "
                                  f"physiologically implausible and indicates a failed "
                                  f"systolic segmentation, not a hyperdynamic ventricle"})

    bands = [_ef_band(e) for e in efs]
    band = max(set(bands), key=bands.count)
    raw = bands.count(band) / len(bands)

    cal = _json(HEART_CALIBRATION)
    conf = (float(np.interp(raw, cal["isotonic_x"], cal["isotonic_y"])) if cal else raw)

    return S.make_report(
        "heart",
        [S.make_finding(EF_FINDING[band], conf)],
        measurements={"ejection_fraction": round(median_ef, 1),
                      "ef_spread_pp": round(float(np.std(efs)), 1)},
        quality={"lv_detected": True, "tta_agreement": round(raw, 3),
                 "usable_views": len(efs), "views": len(views)},
        reliability={"confidence_calibrated": bool(cal),
                     "has_normal_class": True,
                     "confidence_ceiling": (cal or {}).get("ceiling", 1.0),
                     "ece": (cal or {}).get("ece"),
                     "scope": "CAMUS-like 4CH stills; EF is an area proxy, not volumetric"},
        model="effnetb0_unet")


def ultrasound_agent(organ: str, **kwargs) -> dict[str, Any]:
    """One entry point. An organ without a usable model returns `not_supported` in the ordinary
    format, so the reasoning layer never branches on whether a key exists."""
    organ = organ.lower()
    if organ not in S.ORGANS:
        raise ValueError(f"unknown organ {organ!r}; expected one of {sorted(S.ORGANS)}")

    # A missing image is a caller error, but it must not reach the caller as a KeyError. Every
    # other degradation in this system arrives as a report with `status: failed`, and this one
    # does too, so the reasoning layer never has to catch an exception to stay correct.
    def _no_image(what: str) -> dict[str, Any]:
        return S.make_report(organ, [], status="failed",
                             reliability={"scope": f"no {what} supplied"})

    if organ == "lung" and LUNG_WEIGHTS:
        frames = kwargs.get("frames")
        if frames is None:
            frames = [kwargs["image"]] if kwargs.get("image") is not None else []
        return predict_lung(frames, fold=kwargs.get("fold", 0))

    if organ == "gallbladder" and GB_WEIGHTS.exists():
        if kwargs.get("image") is None:
            return _no_image("image")
        return predict_gallbladder(kwargs["image"])

    if organ == "heart" and HEART_WEIGHTS.exists():
        # EF is a comparison between end-diastole and end-systole, so one still cannot produce
        # it. Passing a single frame as both would compute a fractional change of zero and
        # report a normal ventricle -- an invented measurement dressed as a reading -- so the
        # caller must supply both and is told so rather than quietly given a wrong number.
        ed, es = kwargs.get("ed"), kwargs.get("es")
        if ed is None or es is None:
            frames = kwargs.get("frames") or []
            if len(frames) >= 2:
                ed, es = frames[0], frames[-1]
        if ed is None or es is None:
            return S.make_report(
                organ, [], status="failed",
                reliability={"scope": "ejection fraction needs two frames, end-diastole and "
                                      "end-systole; one still cannot produce it"})
        return predict_heart(ed, es)

    reason = ("weights are not present in this deployment"
              if organ in ("heart", "gallbladder") else "module not implemented")
    return S.make_report(organ, [], status="not_supported",
                         reliability={"scope": reason})
