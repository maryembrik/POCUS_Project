"""Recompute calibration artifacts WITHOUT retraining.

The calibrators live only in each training notebook's memory, so a lost runtime loses them. But
the model weights are on Drive, and calibration only needs predictions -- which is inference.
These cells rebuild the artifacts the unified Ultrasound Agent expects.

Gallbladder is not here: its weights were never saved at all, so it genuinely must be re-run.
"""

# =====================================================================================
# CARDIAC  --  no new code needed, just run the right cells
# =====================================================================================
# Sections 10 and 11 already load the checkpoint themselves, so they are inference-only.
# In a fresh runtime run, in order:
#
#     Section 1  Setup
#     Section 2  Cache            (rebuilds the local CAMUS cache, few minutes)
#     Section 3  Split
#     Section 4  Dataset
#     Section 5  Model
#     Section 6  Loss
#     Section 7  FIRST cell only  (defines run_epoch / dice_per_class -- SKIP the second,
#                                  which is the training loop)
#     Section 8  CAMUS results
#     Section 10 Ejection fraction
#     Section 11 Agent contract + calibration
#
# then this export cell:

CARDIAC_EXPORT = r"""
import json
xs = np.linspace(0, 1, 21)
json.dump({'isotonic_x': xs.tolist(),
           'isotonic_y': CALIBRATOR.predict(xs).tolist(),
           'ceiling': float(CALIBRATOR.predict([1.0])[0]),
           'ece': float(ece)},
          open(DRIVE_ROOT / 'Cardiac' / 'cardiac_calibration.json', 'w'), indent=2)
print('saved cardiac_calibration.json')
"""


# =====================================================================================
# LUNG  --  replace Section 9 (the training loop) with this
# =====================================================================================
# Section 9 is the only part that trains. Every fold's best checkpoint was saved to Drive, so the
# out-of-fold predictions can be rebuilt by loading them and running inference. Run Sections 1-8
# normally (they only define things), then THIS cell instead of Section 9, then 10 and 11.

LUNG_RECOMPUTE = r"""
# Rebuild pooled out-of-fold predictions from the checkpoints Section 9 already saved.
# Inference only -- no training. Requires DEV_MODE = False so cv_splits is the full partition.
assert not DEV_MODE, 'set DEV_MODE = False in Section 1: pooling needs the full partition'

backbone = 'efficientnet_b0'
BACKBONES_TO_RUN = [backbone]
pooled_labels = df[FINDING_COLS].to_numpy(dtype=int)
pooled_probs = {backbone: np.full((len(df), len(FINDING_COLS)), np.nan)}

for k, (tr, va) in enumerate(cv_splits):
    ckpt = DRIVE_ROOT / 'Pulmonary' / f'lung_finding_classifier_{backbone}_split{k}_best.pth'
    if not ckpt.exists():
        raise FileNotFoundError(
            f'{ckpt.name} not found. Section 9 must have completed at least once for this fold; '
            'there is no way to recover its predictions otherwise.')

    _tr_ds, _tr_ld, va_ds, va_ld = build_dataloaders(backbone, tr, va)
    model = build_model(backbone)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    logits, labels, clip_ids = predict(model, va_ld)
    frame_probs = 1 / (1 + np.exp(-logits))
    clip_probs, _ = aggregate_by_clip(frame_probs, labels, clip_ids)

    # global_positions maps this fold's local clip ids back to rows of df, so folds write into
    # disjoint slices of one full-dataset array.
    pooled_probs[backbone][va_ds.global_positions] = clip_probs
    print(f'fold {k}: restored {len(va_ds.global_positions)} clip predictions from {ckpt.name}')

missing = int(np.isnan(pooled_probs[backbone][:, 0]).sum())
print(f'\ncoverage: {len(df) - missing}/{len(df)} clips')
assert missing == 0, 'some clips still unpredicted -- a fold checkpoint is missing'
print('Now run Section 10 (pooled results) and Section 11 (calibration), then the export below.')
"""

LUNG_EXPORT = r"""
import json
json.dump({'platt': {n: {'coef': float(CALIBRATORS[n].coef_[0][0]),
                         'intercept': float(CALIBRATORS[n].intercept_[0])}
                     for n in FINDING_COLS},
           'thresholds': {n.replace('finding_', ''): float(np.mean(chosen_t[n]))
                          for n in FINDING_COLS},
           'unreliable_findings': UNRELIABLE},
          open(DRIVE_ROOT / 'Pulmonary' / 'lung_calibration.json', 'w'), indent=2)
print('saved lung_calibration.json')
"""


if __name__ == '__main__':
    for name, snippet in [('CARDIAC export', CARDIAC_EXPORT),
                          ('LUNG recompute (replaces Section 9)', LUNG_RECOMPUTE),
                          ('LUNG export', LUNG_EXPORT)]:
        print('=' * 78)
        print(name)
        print('=' * 78)
        print(snippet)
