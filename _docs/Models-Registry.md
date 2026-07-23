# Pretrained Models Registry (2026-07-11)

Model choices per pipeline component, picked to save training time and get the best results given the assumed Colab-tier/single-GPU compute budget (see Decision Log — still an open question to confirm actual compute).

## Pulmonary CV model (Ultrasound Agent — lung branch)

**Pick: USCL ResNet-18 backbone** — already the plan in the context doc; now fully in hand.

- Pretrained on ultrasound-domain data (Butterfly + COVID19-LUSMS lung sources), not ImageNet — scored 94.19% fine-tuning accuracy on POCUS vs ~84% from ImageNet pretraining (10-point gap, confirmed by the original paper and independent domain-adaptation literature: in-domain ultrasound pretraining consistently beats ImageNet transfer for ultrasound tasks).
- **Weights now in hand**: `Pulmonary/USCL_weights/checkpoint/best_model.pth` (47MB) + `config.yaml`, pulled from the paper's Google Drive link.

### Architecture decision (2026-07-11): split by signal type, not one model for all 5 pathologies

The 5 target pathologies split into two genuinely different problems, so one architecture for all of them is the wrong call:

- **Texture-based** (interstitial syndrome/B-lines, pulmonary oedema, pneumonia, pleural effusion) — visible in a single still frame. Structural/textural pattern recognition, exactly what a CNN is good at.
- **Motion-based** (pneumothorax) — "lung sliding" is a *motion* pattern (the M-mode "seashore vs. barcode" sign), not visible in a static frame at all. A per-frame CNN structurally cannot see this regardless of training data.

**Tier 1 (build first, fits the 1.5-week modeling phase):**
- USCL ResNet-18 fine-tuned as a **multi-label** classifier (sigmoid heads, not softmax) for the 4 texture-based pathologies — multi-label because they co-occur clinically (e.g. pneumonia + effusion), softmax would wrongly force them to compete for probability mass.
- **Pneumothorax handled outside the neural net entirely**: a classical, training-free motion-variance feature (frame-to-frame pixel variance / optical flow at the pleural line, replicating M-mode interpretation computationally) feeds into the final decision alongside the CNN output. This directly targets the actual diagnostic sign instead of asking a classifier to learn "motion" from the near-zero real pneumothorax examples available — and it costs no training data or time to build.
- Bonus: gives a motion-heatmap visualization for pneumothorax, which reads more intuitively to a clinician than a Grad-CAM saliency map would for a signal that's fundamentally about movement, not texture.

**Tier 2 (stretch — only pursue if Tier 1 lands with time to spare):**
- USCL's own pretraining used *video-level* contrastive pairs (nearby frames pulled together in embedding space), so it already gives temporally-aware embeddings for free. Natural upgrade path: **freeze the USCL backbone, train only a small GRU/attention head on top of its frame embeddings** across a short clip window. Few parameters, cheap to train even on modest compute/data — this can absorb or beat the classical motion feature once there's bandwidth to compare.

**Considered and explicitly not pursued:** an autoencoder-based anomaly branch (trained on "normal" lung patterns, flagging high-reconstruction-error clips). Rejected because it addresses the vaguer "not enough labeled examples" problem rather than pneumothorax's actual problem (no motion signal in a static frame) — the classical motion feature is a more targeted, cheaper fix for the same gap. Worth one line in the internship report as a considered alternative.

- Attach Grad-CAM (for the Tier-1 CNN branch) after a working baseline, not before.

## Cardiac segmentation (Cardiac stub — now has real data via CAMUS)

**Pick: train a standard U-Net (or nnU-Net) directly on CAMUS** — no separate pretrained weights to source; CAMUS is small (500 patients) and 2D, so training from scratch is fast even on modest compute (hours, not days).

- Benchmark to aim for: nnU-Net gets Dice 0.93 (LV) / 0.86 (myocardium) / 0.89 (left atrium) on CAMUS — that's the target to beat or match.
- Efficiency reference worth replicating: a 2025 IEEE IUS paper built a 2M-parameter lightweight U-Net (16× smaller than nnU-Net's 33M) hitting statistically equivalent Dice scores (0.93/0.85/0.89) at 4× the inference speed (1.35ms/frame vs 5.40ms). **No public code/weights released** for that specific paper — it's an architecture/ablation reference only (their finding: simple affine augmentation + deep supervision matter most; bigger model capacity gives diminishing returns), not something to download.
- This stays a segmentation task (chamber/structure), not the functional emergency signs (pericardial effusion, RV strain) the original spec wants — that's exactly why the missing-datasets request to the clinical team asks for real ED cardiac cases.

## Abdominal / Gallbladder (now has real data via UIdataGB)

**Pick: standard ImageNet-pretrained CNN (EfficientNet-B0 or ResNet-50) fine-tuned on UIdataGB's 9 classes.** No specialized abdominal-ultrasound pretrained backbone exists publicly (unlike lung/cardiac) — this is normal, standard transfer learning is the right call here, nothing further to source.

## Triage Agent

**Pick: rules-based / calibrated classifier (e.g. gradient-boosted trees on structured vitals), not deep learning.** Matches the original MVP plan — no pretrained model needed or expected here; this is a tabular-data problem on MIMIC-IV-ED Demo + NHAMCS, where gradient boosting (XGBoost/LightGBM) with calibration will comfortably outperform anything else at this data volume.

## Clinical Reasoning Agent (LLM + RAG)

**Pick: HuatuoGPT-o1-8B** (recommended over the Auto-US precedent of DeepSeek-R1-7B).

- Apache 2.0 license, built on Llama-3.1-8B, continued-pretrained + fine-tuned specifically for step-by-step medical reasoning (o1-style: generates a reasoning trace, reflects, refines, then answers) — a strong match for "ranked differential diagnosis with grounded justification," which is exactly the output shape this project needs.
- Downloaded the **Q4_K_M GGUF quantization** (~4.9GB) — runs comfortably on a single 8-16GB GPU (Colab-tier), leaving headroom for the RAG retrieval step alongside it.
- In hand: `Reference_ClinicalReasoning/HuatuoGPT-o1-8B-GGUF/HuatuoGPT-o1-8B-Q4_K_M.gguf`
- The Auto-US prompt template (model result + chief complaint + physical exam + extra info → diagnosis + justification + follow-ups) is still the right prompt structure to adapt — swap in HuatuoGPT-o1-8B as the model behind it instead of DeepSeek-R1-7B.
- RAG knowledge base (curated emergency medicine sources) is still unbuilt — that remains the next real task for this agent, independent of model choice.

## Still open

- **Actual compute budget** — everything above assumes Colab-tier/single-GPU. If real GPU access turns out to be bigger (or smaller), these picks may need revisiting, especially the LLM quantization level.
- **RAG knowledge base** for Clinical Reasoning Agent — not yet curated.
