"""Reproducible lung training, extracted from notebooks/lung_cv_training.ipynb.

    python src/lung/train.py --epochs 8 --folds 3 --seed 42

The model, the grouping and the threshold tuning are the notebook's, unchanged. What is new is
that a run is a COMMAND with recorded arguments rather than a session with remembered state, and
that it writes the metrics the promotion gate asks for instead of the ones that happened to be
interesting while iterating.

Three things this file is careful about, all of them ways a lung run can look better than it is:

GROUPING. Frames from one clip, and clips from one patient, must never span the train/test
line. The notebook derives a case key from the filename and splits on it with GroupKFold;
that is reused verbatim, because a leak here inflates every number downstream and the
resulting model would pass the gate on an illusion.

RECALL IS REPORTED PER FINDING. The gate guards recall on b-lines, consolidation and pleural
effusion, so this writes them. A run that reported only macro-F1 would be refused by the gate
for not having shown its dangerous-miss rate -- which is the correct outcome, and the reason
this script exists in the form it does.

THRESHOLDS ARE TUNED ON TRAINING FOLDS, NEVER ON THE TEST FOLD. The operating point is part of
the model; choosing it on the data you then score against is how a system reports numbers it
cannot reproduce in use.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests" / "pulmonary_manifest.csv"
DATA_ROOT = ROOT / "Pulmonary" / "POCUS_extracted"
FINDING_COLS = ["finding_b_lines", "finding_consolidation",
                "finding_pleural_effusion", "finding_pleural_thickening"]
SHORT = [c.replace("finding_", "") for c in FINDING_COLS]

# The notebook's tokens: these name a provider, not a patient, so they must never form a group.
GENERIC_TOKENS = {"avi", "clarius", "youtube", "grep", "video", "image", "butterfly", "pocus"}


def case_key(filename: str, disease_class: str) -> str:
    """Patient/case identifier derived from the filename. The notebook's rule, unchanged.

    Groups ONLY on explicit within-series markers: over-grouping is not free either, and an
    earlier rule that stripped trailing digits merged 21 unrelated clips into one 'case'.
    """
    import re

    stem = re.sub(r"\.[a-z0-9]+$", "", str(filename), flags=re.I)
    core = re.sub(r"^(cov|pneu|reg|vir)[-_]", "", stem, flags=re.I).lower()
    core = core.replace("-", "_").replace("+", "_")
    m = re.search(r"case(\d+)", core)
    if m:
        return f"case{m.group(1)}"
    m = re.match(r"(.*?)_?day_?\d+$", core)
    if m:
        return f"{m.group(1)}_dayseries"
    m = re.match(r"(.*?)_?(?:vid|clip)_?\d+$", core)
    if m:
        base = m.group(1).strip("_")
        if len(base) > 6 and base.split("_")[0] not in GENERIC_TOKENS:
            return f"{base}_series"
    return f"{disease_class}::{core}"


def extract_frames(filepath: Path, media_type: str, n_frames: int) -> list:
    """The notebook's extractor: evenly spaced frames from a clip, or the single image."""
    import cv2
    import imageio.v2 as imageio

    if media_type == "image":
        return [np.array(Image.open(filepath).convert("RGB"))]
    if filepath.suffix.lower() == ".gif":
        frames = [np.array(Image.fromarray(f).convert("RGB"))
                  for f in imageio.mimread(str(filepath))]
    else:
        cap = cv2.VideoCapture(str(filepath))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    return [frames[i] for i in idx]


class FrameSet(Dataset):
    def __init__(self, frames, labels, groups, img_size, train: bool):
        self.frames, self.labels, self.groups = frames, labels, groups
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tf = (T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)),
                              T.RandomHorizontalFlip(), T.RandomAffine(8, (0.05, 0.05)),
                              T.ColorJitter(0.2, 0.2), T.ToTensor(), norm])
                   if train else
                   T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)),
                              T.ToTensor(), norm]))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.tf(self.frames[i]), torch.tensor(self.labels[i], dtype=torch.float32)


class LungFindingClassifier(nn.Module):
    """Backbone -> dropout -> 512 -> ReLU -> dropout -> one logit per finding. The notebook's."""

    def __init__(self, feature_extractor, num_ftrs, num_labels, dropout=0.4, hidden=512):
        super().__init__()
        self.features = feature_extractor
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(num_ftrs, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, num_labels))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def build_model(unfreeze_last_n: int, device) -> nn.Module:
    effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model = LungFindingClassifier(nn.Sequential(effnet.features, effnet.avgpool),
                                  effnet.classifier[1].in_features,
                                  len(FINDING_COLS)).to(device)
    # Partial fine-tuning: the early blocks keep generic edge/texture filters that ~165 cases
    # cannot re-learn; the last few adapt to ultrasound speckle, which ImageNet does not encode.
    for p in model.features.parameters():
        p.requires_grad = False
    blocks = list(model.features[0])
    for blk in blocks[-unfreeze_last_n:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return model


def load_cache(frames_per_clip: int, verbose: bool = True):
    """Decode every usable study once. 198 files, so this is cheap and worth not repeating."""
    df = pd.read_csv(MANIFEST)
    usable = df[~df.flag_do_not_use & ~df.flag_off_target_organ & ~df.excluded_by_curators]
    frames, labels, groups = [], [], []
    for _, row in usable.iterrows():
        fp = DATA_ROOT / Path(str(row.filepath).replace("\\", "/"))
        if not fp.exists():
            continue
        got = extract_frames(fp, row.media_type, frames_per_clip)
        y = [int(bool(row[c])) for c in FINDING_COLS]
        g = case_key(row.filename, row["class"])
        for f in got:
            frames.append(f)
            labels.append(y)
            groups.append(g)
    if verbose:
        print(f"      {len(frames)} frames from {len(set(groups))} case groups")
    return frames, np.array(labels), np.array(groups)


def tune_thresholds(y_true, prob):
    """Per-finding operating point, chosen on the data given here -- which the caller must
    ensure is NOT the fold being scored."""
    out = {}
    for i, name in enumerate(SHORT):
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.05):
            f1 = f1_score(y_true[:, i], (prob[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        out[name] = round(best_t, 2)
    return out


def run_fold(tr_idx, te_idx, frames, labels, groups, args, device, log):
    tr = FrameSet([frames[i] for i in tr_idx], labels[tr_idx], groups[tr_idx],
                  args.img_size, train=True)
    te = FrameSet([frames[i] for i in te_idx], labels[te_idx], groups[te_idx],
                  args.img_size, train=False)
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_te = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.unfreeze, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # Positive weighting: every finding is a minority class, and an unweighted loss on this
    # data is minimised by predicting "absent" for all four.
    pos = labels[tr_idx].sum(axis=0).clip(min=1)
    neg = len(tr_idx) - pos
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(neg / pos, dtype=torch.float32, device=device))

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        log(f"      epoch {ep + 1}/{args.epochs}  train loss {total / len(tr):.4f}")

    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
            trues.append(yb.numpy())
    # Thresholds from the TRAINING fold only.
    tr_probs = []
    with torch.no_grad():
        for xb, _ in DataLoader(FrameSet([frames[i] for i in tr_idx], labels[tr_idx],
                                         groups[tr_idx], args.img_size, train=False),
                                batch_size=args.batch_size):
            tr_probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    thresholds = tune_thresholds(labels[tr_idx], np.vstack(tr_probs))
    return np.vstack(probs), np.vstack(trues), thresholds, model


def main() -> int:
    p = argparse.ArgumentParser(description="Train the lung finding classifier.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--frames", type=int, default=8, help="frames sampled per clip")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--unfreeze", type=int, default=3, help="backbone blocks to fine-tune")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-version", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    def log(msg):
        if not args.quiet:
            print(msg, flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "outputs" / "lung" / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    log(f"[1/4] Loading dataset ({device})")
    t0 = time.time()
    frames, labels, groups = load_cache(args.frames, verbose=not args.quiet)
    if not frames:
        print("FATAL: no frames decoded. The media is not on disk; this run would have "
              "produced metrics from nothing.", file=sys.stderr)
        return 1

    n_folds = min(args.folds, len(set(groups)))
    log(f"[2/4] Training {n_folds} fold(s), {args.epochs} epoch(s) each")
    gkf = GroupKFold(n_splits=n_folds)
    oof_prob = np.zeros_like(labels, dtype=float)
    fold_thresholds, last_model = [], None
    for k, (tr_idx, te_idx) in enumerate(gkf.split(frames, labels, groups), 1):
        log(f"   fold {k}/{n_folds}")
        prob, _, thr, model = run_fold(tr_idx, te_idx, frames, labels, groups,
                                       args, device, log)
        oof_prob[te_idx] = prob
        fold_thresholds.append(thr)
        last_model = model

    log("[3/4] Evaluating (out-of-fold, thresholds tuned on training folds only)")
    thresholds = {n: float(np.mean([t[n] for t in fold_thresholds])) for n in SHORT}
    metrics: dict[str, float] = {}
    for i, name in enumerate(SHORT):
        yt, pr = labels[:, i], oof_prob[:, i]
        pred = (pr >= thresholds[name]).astype(int)
        metrics[f"auroc_{name}"] = float(roc_auc_score(yt, pr)) if len(set(yt)) > 1 else 0.0
        metrics[f"f1_{name}"] = float(f1_score(yt, pred, zero_division=0))
        # The gate guards these. A run that did not report them would be refused, correctly.
        metrics[f"recall_{name}"] = float(recall_score(yt, pred, zero_division=0))
    metrics["macro_auroc"] = float(np.mean([metrics[f"auroc_{n}"] for n in SHORT]))
    metrics["macro_f1"] = float(np.mean([metrics[f"f1_{n}"] for n in SHORT]))
    metrics["macro_recall"] = float(np.mean([metrics[f"recall_{n}"] for n in SHORT]))
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    for n in SHORT:
        log(f"      {n:22} AUROC {metrics[f'auroc_{n}']:.3f}  "
            f"F1 {metrics[f'f1_{n}']:.3f}  recall {metrics[f'recall_{n}']:.3f}")
    log(f"      {'macro':22} AUROC {metrics['macro_auroc']:.3f}  "
        f"F1 {metrics['macro_f1']:.3f}  recall {metrics['macro_recall']:.3f}")

    log(f"[4/4] Writing {out}")
    torch.save(last_model.state_dict(), out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    (out / "config.json").write_text(json.dumps({
        "model": "lung",
        "dataset_version": args.dataset_version,
        "eval_set": f"pulmonary manifest, GroupKFold {n_folds}-fold out-of-fold, "
                    f"thresholds tuned on training folds",
        "frames": len(frames), "case_groups": int(len(set(groups))),
        "epochs": args.epochs, "folds": n_folds, "batch_size": args.batch_size,
        "img_size": args.img_size, "unfreeze_last_n": args.unfreeze, "lr": args.lr,
        "seed": args.seed, "device": str(device),
        "thresholds": thresholds,
        "seconds": round(time.time() - t0, 1),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf8")
    log(f"      model.pt, metrics.json, config.json  ({time.time() - t0:.0f}s)")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
