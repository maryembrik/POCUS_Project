# POCUS-Emergency -- the deployed application.
#
#   docker build -t pocus-emergency .
#   docker run --rm -p 8501:8501 pocus-emergency
#   -> http://localhost:8501   (French interface at /fr)
#
# What this image contains: serve.py, the pipeline under src/, the generated interface in web/,
# and the three ultrasound checkpoints with their calibration files (~110 MB together).
#
# What it deliberately does not contain:
#
#   the datasets          not redistributable under their source licences, and the running
#                         application never reads them -- only training does
#   the language model    4.9 GB, and the deployed clinical agent runs on a test backend
#   the training stack    MLflow, notebooks, the triage classifier: none is reached by a
#                         request, and requirements-train.txt keeps them out
#
# The result is an image that serves every screen the interface has, computes every value on
# them, and runs the whole deterministic safety layer -- which is the part that must behave
# identically wherever it runs.

FROM python:3.13-slim AS base

# Python behaviour, set before anything is installed so every later layer inherits it.
#   PYTHONDONTWRITEBYTECODE  no .pyc files; they only bloat the layer
#   PYTHONUNBUFFERED         logs appear immediately instead of when a buffer fills, which is
#                            the difference between useful and useless container logs
#   HOST                     0.0.0.0, because 127.0.0.1 inside a container is the container's
#                            own loopback: the process would answer itself and nothing else,
#                            and a published port would connect to no listener
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8501

WORKDIR /app

# Requirements first, and only requirements. Docker caches by layer, so copying the source
# before installing would rebuild the whole dependency tree -- torch included -- on every edit
# to a Python file. This ordering is why a rebuild after a code change takes seconds.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# ── the application ────────────────────────────────────────────────────────────────────
# A second stage on top of the first, and the split is not cosmetic. `base` contains the
# interpreter and the pinned dependencies and nothing from this repository except the
# requirements file, so it can be built anywhere -- including in CI, where the model weights
# are not present and every COPY below would fail.
#
# Building `--target base` therefore checks the three things that rot quietly between releases:
# that the base image is still pullable, that the pinned set still resolves and installs on a
# clean machine, and that this file still parses. The layers below need the weights and are
# exercised by a full build, which is run where the weights are.
FROM base AS app

# Ordered least- to most-frequently changed, for the layer cache.
#
# models/ is copied file by file rather than wholesale. The directory holds 77 MB, of which
# 76 MB is the triage classifier in three serialisations -- and the deployed application never
# loads it, because the interface takes the urgency tier from the clinician. Only these two
# files are read at runtime.
COPY models/safety_benchmark.json ./models/
COPY models/clinical_reasoning_v4_final/results.json ./models/clinical_reasoning_v4_final/
# The lung module loads the per-fold checkpoints, not the final one. agent.py globs
#
#     lung_finding_classifier_efficientnet_b0_split*_best.pth
#
# so `..._final_best.pth` never matches it, and an image built with only that file starts
# cleanly and reports "weights absent" for lung while the other two modules work. That is
# precisely what the first build did, and /api/health is how it was found.
#
# The per-split results files come too: with no lung_calibration.json in the repository, the
# module falls back to the tuned thresholds recorded in them. Without those it would fall back
# again, to a bare 0.5 -- a container that silently applies different decision thresholds from
# the host is worse than one that fails to start.
COPY Pulmonary/lung_finding_classifier_efficientnet_b0_split0_best.pth \
     Pulmonary/lung_finding_classifier_efficientnet_b0_split1_best.pth \
     Pulmonary/lung_finding_classifier_efficientnet_b0_split2_best.pth \
     Pulmonary/results_efficientnet_b0_split0.json \
     Pulmonary/results_efficientnet_b0_split1.json \
     Pulmonary/results_efficientnet_b0_split2.json \
     ./Pulmonary/
COPY Cardiac/cardiac_effnet_unet_best.pth \
     Cardiac/cardiac_calibration.json \
     ./Cardiac/
COPY Abdominal_Gallbladder/gallbladder_effnetb0_best.pth \
     Abdominal_Gallbladder/gallbladder_calibration.json \
     ./Abdominal_Gallbladder/
COPY src/ ./src/
COPY web/ ./web/
COPY serve.py ./

# Run as a non-root user. A container that serves clinical software should not be able to write
# to its own application directory, and root inside a container is root on the host kernel if
# anything escapes. Created without a login shell and given no ownership of /app.
RUN useradd --create-home --shell /usr/sbin/nologin pocus
USER pocus

EXPOSE 8501

# The health endpoint reports whether each perception module actually loaded, so an unhealthy
# container is one that cannot do its job -- not merely one whose socket is closed. Start
# period allows for the torch import on a cold container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "serve.py"]
