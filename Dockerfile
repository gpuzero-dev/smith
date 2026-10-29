# syntax=docker/dockerfile:1
#
# smith in a container on CPU: a training environment that takes jobs from
# a backend's queue.
#
# Moonclip and Ravex come from one of two places, picked by LIBS:
#
#   docker build -t smith .
#       from PyPI, at MOONCLIP_VERSION and RAVEX_VERSION. What anybody can
#       build from this repository alone.
#
#   docker build -t smith --build-arg LIBS=source \
#       --build-context moonclip=../moonclip --build-context ravex=../ravex .
#       built from checkouts, for a Ravex that is ahead of PyPI.
#
# BuildKit builds only the stage LIBS names, so the default build never asks
# for the two contexts.
ARG LIBS=pypi

FROM python:3.12-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# CPU wheels: the CUDA build is gigabytes this image would never use. The index
# pair is ravex CI's: `--index-url` alone replaces PyPI, and torch's own build
# dependencies are not on the CPU index.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple torch numpy

# Ravex samples CPU and RAM through psutil, and only if it is importable: without
# it the run ships no system metrics and says nothing about it. (GPUs go
# through pynvml, which belongs in the GPU image, where there is one to read.)
RUN pip install psutil

# ─── libraries from PyPI ───────────────────────────────────────────────
FROM base AS libs-pypi
ARG MOONCLIP_VERSION=0.3.1
ARG RAVEX_VERSION=0.10.1
RUN pip install "moonclip==${MOONCLIP_VERSION}" "ravex==${RAVEX_VERSION}"
# The scripts smith.docker.toml offers are Ravex's examples, which the wheel
# does not carry: taken from the release's tag, so they match the library.
RUN mkdir -p /app \
    && curl -fsSL "https://github.com/gpuzero-dev/ravex/archive/refs/tags/v${RAVEX_VERSION}.tar.gz" \
       | tar -xz -C /tmp \
    && mv "/tmp/ravex-${RAVEX_VERSION}/examples" /app/examples \
    && rm -rf "/tmp/ravex-${RAVEX_VERSION}"

# ─── libraries from checkouts ──────────────────────────────────────────
FROM base AS libs-source
ENV PATH="/root/.cargo/bin:${PATH}"
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
# For Moonclip, the version its rust-toolchain.toml pins (Ravex is pure Python
# since GPU-267). Naming another one here only makes rustup fetch a second
# compiler to satisfy that file - and clippy is installed now because that file
# asks for it, and a component added lazily while maturin runs cargo can race
# itself.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.98.1 --profile minimal --component clippy
# Moonclip before Ravex, and each in its own layer, so a change in one does
# not rebuild the other. The cache mounts keep cargo's downloads and build tree
# between image builds without putting either in the image.
COPY --from=moonclip . /src/moonclip
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/src/moonclip/target \
    pip install /src/moonclip
COPY --from=ravex . /src/ravex
RUN pip install /src/ravex \
    && mkdir -p /app && cp -r /src/ravex/examples /app/examples

# ─── smith ─────────────────────────────────────────────────────────────
FROM libs-${LIBS}
COPY . /app/smith
WORKDIR /app

CMD ["python", "smith/smith.py", "--config", "smith/smith.docker.toml"]
