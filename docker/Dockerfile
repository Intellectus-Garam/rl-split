# syntax=docker/dockerfile:1
# Learner and worker image for DGX Spark (aarch64).
# Build from the repo root: docker build -f docker/Dockerfile -t rl-split:0.1 .

FROM rust:1.89-slim AS int2dds
ARG INT2DDS_REF=v1.0.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "$INT2DDS_REF" https://github.com/IntellectusCorp/int2DDS.git /src/int2DDS
WORKDIR /src/int2DDS
RUN cargo build --release -p int2dds-ffi

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/int2DDS/python

COPY requirements/ /opt/rl-split/requirements/
RUN pip install --no-cache-dir -r /opt/rl-split/requirements/worker.txt tensorboard

# int2dds finds the library under <int2DDS>/target/release on its own.
COPY --from=int2dds /src/int2DDS/python/ /opt/int2DDS/python/
COPY --from=int2dds /src/int2DDS/target/release/libint2dds_ffi.so /opt/int2DDS/target/release/

COPY configs/ /opt/rl-split/configs/
COPY rl_split/ /opt/rl-split/rl_split/
COPY scripts/ /opt/rl-split/scripts/
WORKDIR /opt/rl-split
