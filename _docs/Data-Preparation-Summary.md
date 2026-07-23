# Data Preparation Phase — Summary (2026-07-11)

CRISP-DM Phase 3 (Data Preparation). Code lives in `src/data_prep/`, all outputs in `manifests/`.

## Pulmonary — `src/data_prep/pulmonary_manifest.py` → `manifests/pulmonary_manifest.csv`

- 198/198 files classified (71 covid19, 80 healthy, 40 bacterial_pneumonia, 7 viral_pneumonia).
- Bonus: the dataset ships a `dataset_metadata.csv` with real free-text findings from reviewing doctors for 174/198 files. Extracted keyword flags from it — `finding_b_lines` (65), `finding_consolidation` (65), `finding_pleural_effusion` (26), `finding_pleural_thickening` (47) — giving multi-label pathology signal beyond the coarse 4-class label, directly feeding the Tier-1 multi-label classifier design.
- Caught 1 file explicitly flagged "do not use" by the original curators, and 11 files that mention liver in view (10 of these are normal lung-liver interface views, clinically fine; only 1 overlaps with the do-not-use flag).
- Confirmed 0 pneumothorax mentions anywhere in the findings text — consistent with the known real-world gap.

## Cardiac — `src/data_prep/cardiac_manifest.py` → `manifests/cardiac_manifest.csv`

- 500/500 CAMUS patients read cleanly (0 bad zips — confirms the integrity-repair work from the download phase held).
- Per-patient: sex, age, image quality (Good/Medium/Poor) for both 2CH/4CH views, ejection fraction, frame rate, ED/ES frame indices.
- Ejection fraction distribution: mean 44%, range 5-81% — directly usable for an "impaired ventricular function" signal from the original spec, not just segmentation.
- Image quality: 69/500 "Poor" on 2CH, 47/500 "Poor" on 4CH — worth excluding or down-weighting for initial training.

## Abdominal/Gallbladder — `src/data_prep/gallbladder_manifest.py` → `manifests/gallbladder_manifest.csv`

- 10,694 images across the 9 disease classes.
- **Important catch**: only 14-32 unique patients per class despite 1,000+ images per class (~40-90 images/patient). A naive random image-level split would leak the same patient across train/val/test and inflate validation metrics — **must split by `patient_id`**, flagged clearly for the modeling phase.

## Triage — `src/data_prep/triage_nhamcs_prep.py` → `manifests/nhamcs_triage_core_combined.csv`

- Parsed the previously-unparseable raw ASCII NHAMCS files (2021+2022) into 32,232 real ED visits using the Stata dictionaries pulled from the CDC.
- **Two data-quality issues caught and fixed by checking the CDC's own PDF documentation rather than guessing:**
  - Sentinel missing-value codes (`-9`=Blank, `-8`=Unknown, `998`=Doppler reading) were initially polluting vitals statistics — e.g. pulse "998" is a Doppler-method flag, not a real 998 bpm reading. Fixed via an explicit sentinel map per field.
  - `IMMEDR` (the real triage-nurse-assigned acuity — the actual label an ESI-style triage model should train on) — initially mis-labeled with guessed generic time buckets; corrected to the official categories: Immediate / Emergent / Urgent / Semi-urgent / Nonurgent / no-triage-reported / ESA-doesn't-triage.
  - `TEMPF` has an implied decimal (raw `982` = 98.2°F) — applied the /10 correction; range now sanity-checks at 84-105.8°F.
- Final vitals completeness after cleanup: 90-95% for temp/pulse/resp/BP/SpO2, 57% for pain scale (matches the CDC's own documented collection rate).

## Biological — `src/data_prep/vitaldb_prep.py` → `manifests/vitaldb_biological_manifest.csv`

- 6,388 cases, pivoted from 928,448 long-format lab readings (34 test types) into a wide per-case table (median value per lab).
- Confirms the registry's caveat: lactate present (54% coverage), but troponin and D-dimer are absent from VitalDB entirely — still need the missing-datasets request to the clinical team for those, or PhysioNet Challenge 2019 (still blocked by this environment's network) as a fallback.
- Reminder: VitalDB is 100% surgical/anesthesia context, not ED — code-practice data for the fusion logic, not a stand-in for real ED cases.

## Not yet processed

- **MIMIC-IV-ED Demo + MIMIC-IV Clinical Demo** — structured, already close to analysis-ready; lower priority since NHAMCS now covers the core triage-modeling need.
- **MTSamples** — free-text clinical notes; useful later for the Clinical Reasoning Agent's history/exam-text extraction, not urgent right now.

## Cross-cutting notes for the modeling phase

1. **Split by patient/case ID everywhere**, not by file/image/row — confirmed necessary for Gallbladder (severe leakage risk) and worth double-checking for Pulmonary (some POCUS patients may have both video and image entries) and CAMUS (each patient has 2CH+4CH — keep both views together in the same split).
2. Sentinel/missing-value codes are dataset-specific and undocumented in the raw files themselves — always check the source's own data dictionary before trusting a numeric column's summary statistics (NHAMCS was the clear example here).
