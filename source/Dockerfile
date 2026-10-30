# PI0.5 full-parameter training only. Models and datasets are bind-mounted.
ARG BASE_IMAGE=ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
FROM ${BASE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface HF_LEROBOT_HOME=/cache/lerobot \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    PIPER_TOKENIZER=/tokenizer HOME=/tmp/piper-home
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-venv python3-dev ffmpeg git build-essential \
    pkg-config libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python -m pip install --no-cache-dir pip==25.2 uv==0.12.11 setuptools==81.0.0 wheel
COPY constraints.txt requirements-torch.txt /opt/piper-build/
RUN uv pip install --python /opt/venv/bin/python --no-cache \
    -c /opt/piper-build/constraints.txt -r /opt/piper-build/requirements-torch.txt
COPY vendor/lerobot-0.6.1-source.tar.gz /opt/piper-build/
RUN tar -xzf /opt/piper-build/lerobot-0.6.1-source.tar.gz -C /opt \
    && uv pip install --python /opt/venv/bin/python --no-cache \
       -c /opt/piper-build/constraints.txt -e '/opt/lerobot[pi,training]' \
    && uv pip check --python /opt/venv/bin/python \
    && python -m pip freeze > /opt/piper-build/resolved-environment.txt
COPY container/ /opt/piper/
COPY evidence/source_manifest.json /opt/piper-build/source_manifest.json
RUN python /opt/piper/entrypoint.py check --cpu
WORKDIR /tmp
LABEL org.opencontainers.image.title="Piper PI0.5 full training" \
      org.opencontainers.image.version="20261030" \
      piper.training.method="full-parameter; no LoRA" \
      piper.lerobot.recorded_revision="7e241bd630a3719a56157a497ce5d08f244784f1"
ENTRYPOINT ["python", "/opt/piper/entrypoint.py"]
CMD ["check"]
