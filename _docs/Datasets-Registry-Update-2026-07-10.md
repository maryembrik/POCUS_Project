# Datasets & Resources Registry — Update (2026-07-10)

Supersedes/extends Section 6 of `POCUS-Emergency-Project-Context.md`. Reflects a full re-audit of the project folder plus a fresh search pass for anything not already listed there, focused especially on closing the Triage Agent and Cardiac gaps.

## Folder reorganization

Everything in `POCUS-Project/` is now sorted into subfolders by what it's for, instead of sitting flat in the root:

| Folder | Contents |
|---|---|
| `_docs/` | Context doc, original spec (`POCUS_EchoetAI-en.docx` + FR version), pipeline diagram |
| `Pulmonary/` | POCUS (Born et al.), COVID-US repo (scraper + masks), USCL backbone, B-line detection repo |
| `Cardiac/` | CAMUS (new) |
| `Abdominal_Gallbladder/` | UIdataGB gallbladder dataset (new) |
| `Triage_NHAMCS/` | NHAMCS raw ASCII (2021 + 2022) + Stata/SAS format dictionaries (new) |
| `Triage_MIMIC/` | MIMIC-IV-ED Demo + MIMIC-IV Clinical Demo |
| `Clinical_Notes_MTSamples/` | MTSamples (was mislabeled `archive.zip` at root) |
| `Biological_VitalDB/` | VitalDB case data + parameter/format docs |
| `Reference_ClinicalReasoning/` | Auto-US (prompt template reference) |
| `Reference_OtherOrgan_BreastCVANet/` | CVA-Net (breast, reference architecture only) |

Two corrections found during re-audit:
- The file that was sitting at the root as `archive.zip` is actually **MTSamples** (`mtsamples.csv`), not the Mendeley gallbladder set as previously guessed.
- `clinical information.csv` + `lab results.csv` at the root **are** the VitalDB case data — despite the registry's earlier "reported downloaded, unverified" status, this is real, valid VitalDB data (confirmed by column structure: `caseid, subjectid, preop_hb, preop_pt...` etc.). Status upgraded to **verified in hand**.
- A stray 844MB `Non confirmé 356006.crdownload` (an interrupted Chrome download, unusable in that state) was removed — almost certainly an earlier failed attempt at the gallbladder dataset, now superseded by a clean download.

## Newly found and downloaded this pass

| Dataset | Fills | Size | License | Status |
|---|---|---|---|---|
| **CAMUS** (cardiac echo, 500 patients, 2D 4-chamber/2-chamber segmentation) | Cardiac — was a total gap | 3.5GB, `Cardiac/CAMUS_nifti/` | CC BY-NC-SA 4.0 (citation required, non-commercial) | **In hand, verified** — 500/500 patients, every file byte-matched against the source manifest, spot-checked with `unzip -t` |
| **UIdataGB** (gallbladder ultrasound, 1,782 patients, 10,692 images, 9 classes) | Abdominal — closes the previously-unverified gap | 2.04GB, `Abdominal_Gallbladder/` | CC BY 4.0 | **In hand, verified** — 9/9 category files, every file byte-matched against the source manifest, spot-checked with `unzip -t` |
| **NHAMCS format dictionaries + 2022 raw data** (Stata `.dct`, SAS format/input programs, 2021+2022 raw ASCII) | Unblocks Triage — the already-in-hand NHAMCS raw file was previously unparseable | ~6MB | Public domain (CDC) | Downloaded |

## Searched but still blocked (need your action, not mine)

These need an account, signed agreement, or personal-identity credentialing — I can't create accounts or sign data use agreements on your behalf, so these remain exactly where the original registry left them:

| Dataset | Fills | Blocker |
|---|---|---|
| **EchoNet-Dynamic** (cardiac, 10,030 echo videos + ejection fraction) | Cardiac (bigger, video-based complement to CAMUS) | Requires a Stanford AIMI account + signed Data Use Agreement |
| **MC-MED** (118K ED visits, vitals + waveforms + ESI + chief complaint) | Would be a major Triage Agent upgrade | Hosted on PhysioNet — **this environment cannot reach physionet.org at all** (connection refused at the network level, confirmed). Also likely credentialed given the physiologic waveform content. You'll need to download this yourself. |
| **MIETIC** (MIMIC-IV-Ext Triage Instruction Corpus, 9,629 ESI-aligned triage cases) | Triage | Also PhysioNet — same network block |
| **PhysioNet Challenge 2019 (Sepsis)** | Biological/labs | Same PhysioNet network block (the file itself is small and license-free — just physically unreachable from here) |
| **FedMML ED Triage Dataset** (Hugging Face, 87K synthetic ED encounters, vitals+notes+labs+ESI) | Triage | Gated on Hugging Face — requires login + accepting access conditions |
| **ICLUS-DB** | Pulmonary severity scoring | Site (`iclus-db.imedlab.org`) doesn't resolve at all right now — may be down, or an access-request portal that's changed |
| **Full MIMIC-IV-ED + MIMIC-IV-Note** | Triage + clinical notes at scale | CITI training + signed DUA tied to your identity |
| **BEDLUS** | Pulmonary (most clinically relevant lung set) | Harvard Dataverse gated access request |
| **COVIDx-US actual video data** | Pulmonary | Repo only has a scraper notebook (`create_COVIDxUS.ipynb`) that pulls live from external sites — running it is a build task, not a download, and some 2022-era sources may be stale/dead now |
| **Kaggle "Hospital Triage and Patient History Data"** (Yale New Haven, 972 variables/visit) | Triage (very rich) | Kaggle requires an API token tied to your account |

## Confirmed real gaps — searched specifically, nothing usable found

- **FAST/E-FAST abdominal free-fluid ultrasound** — no public dataset exists; only single-institution studies not released publicly.
- **DVT/vascular ultrasound** — the ThrombUS+ project is collecting one now but hasn't released it yet. Nothing else found.
- **IVC-specific ultrasound dataset** — nothing found at all, not even gated.
- **Pneumothorax/pleural effusion real-patient ultrasound** — still essentially nonexistent publicly (unchanged from original registry finding); scattered small cohorts (9-31 patients) exist only "available on request" from individual papers, not real datasets.

## Bottom line for the Triage Agent specifically

You can now build and test the full Triage Agent pipeline on real, parseable data: **MIMIC-IV-ED Demo** (structured vitals/ESI) + **NHAMCS 2021/2022** (now parseable with the new format dictionaries) + **MTSamples** (free-text). The bigger upgrades (MC-MED, MIETIC, full MIMIC-IV-ED) all sit behind PhysioNet, which this environment cannot reach — that's a "you download it, I pick it up" step whenever you're ready.
