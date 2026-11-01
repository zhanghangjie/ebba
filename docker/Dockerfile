# EBBA / EBBA+ verification image.
#
#   CPU (default):  docker build -t ebba:cpu .
#   GPU (sm_120):   docker build --build-arg GPU=1 -t ebba:gpu .
#
# The proxy artifacts (3 checkpoints + data subsets) are generated at build
# time and verified by SHA256 at run time, so `verify` works with no network.
ARG BASE_IMAGE=python:3.11.10-slim-bookworm
FROM ${BASE_IMAGE} AS base

ARG GPU=0
ARG SKIP_TRAIN=0
# fastai: PNG mirror on S3 (default, fast everywhere)
# torchvision: canonical pickles from cs.toronto.edu (often rate-limited)
ARG DATA_SOURCE=fastai
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EBBA_THREADS=4 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-common.txt /tmp/requirements-common.txt
RUN pip install -r /tmp/requirements-common.txt

COPY docker/requirements-cpu.txt docker/requirements-gpu.txt /tmp/
RUN if [ "$GPU" = "1" ]; then \
        pip install -r /tmp/requirements-gpu.txt; \
    else \
        pip install -r /tmp/requirements-cpu.txt; \
    fi

# Fail the GPU build early if sm_120 is not in the wheel's compiled arch list.
# Use _cuda_getArchFlags(): get_arch_list() returns [] when no GPU is visible
# during `docker build`, even though the wheel is compiled for sm_120.
RUN if [ "$GPU" = "1" ]; then \
        python -c "import torch; flags=torch._C._cuda_getArchFlags(); print('arch flags:', flags); \
assert 'sm_120' in flags, 'torch build lacks sm_120 (need cu128)';"; \
    fi

WORKDIR /app
COPY src ./src
# Only the artifact builder is needed before training.  Keeping it separate
# from the rest of scripts/ means later edits to train.py do not invalidate
# the expensive 3-model training layer.
COPY scripts/build_artifacts.py ./scripts/build_artifacts.py
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Build the deterministic proxy artifacts inside the image.  Raw archives are
# deleted in the same layer, so they do not add ~450 MB to the final image.
RUN if [ "$SKIP_TRAIN" = "0" ]; then \
        python scripts/build_artifacts.py --device cpu --data-source "$DATA_SOURCE" \
        && rm -rf artifacts/data/raw; \
    else \
        echo "SKIP_TRAIN=1: artifacts must be mounted or copied"; \
    fi

COPY scripts ./scripts
COPY notebooks ./notebooks
COPY reference ./reference
COPY tests ./tests

# Runtime writes only to mounted report/work directories.  The full profile
# mounts artifacts-full, checkpoints-full and reports-full from the host.
RUN mkdir -p /app/reports/claims /app/reports/executed \
    /app/reports-full/executed /app/checkpoints-full /app/artifacts-full

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["verify"]
