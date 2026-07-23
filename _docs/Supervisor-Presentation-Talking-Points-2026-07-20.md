# Supervisor Presentation — Talking Points (2026-07-20)

Covers: how the dataset is organized, the model choice per organ, and the lung hyperparameters/design decisions.

## 1. Dataset organization

- **Manifest-driven, not folder-driven.** Each organ has its own CSV manifest (`pulmonary_manifest.csv`, `cardiac_manifest.csv`, etc.) that maps every file to its labels/metadata — keeps data loading decoupled from raw file layout.
- **Cross-cutting rule: split by patient/clip ID, never by frame or row.** Frames from the same ultrasound clip are near-duplicates; splitting after frame extraction would leak the same clip into both train and val. This rule is applied consistently across every organ (also called out for gallbladder, where only 14–32 unique patients exist per class despite 10k+ images).
- **Curator flags filtered at the manifest level** before anything reaches training (`flag_do_not_use`, `flag_off_target_organ` for lung).
- **Frame sampling, not every frame.** For lung, 8 evenly-spaced frames per clip — adjacent frames in an ultrasound clip are near-identical for a texture-based task, so more buys little.
- **Lung specifically is multi-label**, not single-class: co-occurring pathologies (e.g. consolidation + effusion together) mean a sigmoid-per-class design was used instead of softmax, which would wrongly force findings to compete for probability mass.

## 2. Model choice per organ — and why

Framed as: one organ (pulmonary) built deep this internship, the rest architected as extensible stubs.

- **Lung — 3-candidate "bake-off"**, same fine-tuning recipe applied to all three for a fair comparison:
  - *USCL ResNet-18* — ultrasound-domain pretrained (Butterfly + COVID19-LUSMS), the paper's own number was 94% vs ~84% for ImageNet pretraining.
  - *ImageNet ResNet-18* — included specifically as an honest baseline, to measure the actual domain-pretraining gap on *our* data rather than trusting the paper's claimed number.
  - *USFM ViT-B* — a multi-organ ultrasound foundation model (2M+ images pretrained), the more modern/ambitious candidate.
  - **Pneumothorax deliberately excluded from the CNN** — it's a motion sign (lung sliding) invisible in a single static frame, so no amount of labeled data fixes it; handled instead via a classical motion-variance/optical-flow feature. (Also explicitly rejected an autoencoder-anomaly-detection approach — it solves "not enough labeled examples," not the actual problem of no motion signal being present at all.)
- **Cardiac** — U-Net/nnU-Net trained from scratch on CAMUS (500 patients, segmentation task). No pretrained ultrasound backbone was needed/available at this scale; benchmarked against nnU-Net's published Dice scores (0.93/0.86/0.89 for LV/myocardium/left atrium).
- **Gallbladder (planned, not yet built)** — ImageNet-pretrained EfficientNet-B0/ResNet-50, because no specialized abdominal-ultrasound pretrained backbone exists publicly.
- **Triage agent** — deliberately *not* deep learning: gradient-boosted trees (XGBoost/LightGBM) on tabular vitals/complaint data (NHAMCS), which outperforms neural approaches at this data volume.
- **Vascular/DVT** — stub only, no model yet: no public dataset exists (the one project collecting one, ThrombUS+, is unreleased).

## 3. Lung — hyperparameters and design decisions

- 224×224 input, batch size 16, 25 epochs, LR 1e-3, 8 frames/clip.
- **Class-weighted loss** (`pos_weight` in BCE) — effusion/thickening are much rarer positives than b-lines/consolidation in this dataset, so an unweighted loss would underfit them.
- **Identical freeze strategy across all 3 backbones** — only the last residual block/transformer block + the new classifier head are trainable, everything else frozen. Keeping this identical is what makes the 3-way comparison fair rather than confounded by different amounts of fine-tuning.
- **Backbone-correct normalization** — USCL expects mean/std=0.5/0.25 (not ImageNet stats), which the pipeline accounts for per-backbone.
- **Per-class threshold tuning** instead of a blanket 0.5 cutoff, since positive rates differ hugely by class.
- **Regularization added after diagnosing overfitting from the loss curves** (train loss → ~0 while val loss climbed): Dropout(0.3) before the head, weight_decay=1e-4 in Adam, checkpoint selection by validation AUROC instead of validation loss, and a 15/85 val/train split (slightly less validation data, slightly more training data, since no additional data is available).
- **Fixed random seeding** added for reproducibility across the three backbone runs.

### Honest caveat worth stating directly

The real ceiling right now is dataset size — 198 total clips, ~24 positive effusion clips overall. Regularization and threshold tuning improved things but effusion/thickening AUROC still sits around 0.70 vs ~0.85 for the more frequent classes. That's a data problem, not a tuning problem — the actual next lever is external datasets (ICLUS-DB, COVIDx-US, BEDLUS), which are currently gated/unreachable, not further hyperparameter search.
