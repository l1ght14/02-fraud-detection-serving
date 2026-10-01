# syntax=docker/dockerfile:1
# Multi-stage: the builder keeps build-only wheels out of the runtime image, which
# is most of the size difference. The xgboost/libgomp runtime dependency is the
# other half, installed explicitly below.

FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target /deps -r requirements.txt


FROM python:3.12-slim AS runtime

# libgomp1 is xgboost's OpenMP runtime; without it the import fails at startup
# with a bare "libgomp.so.1: cannot open shared object file" that looks like a
# scikit-learn problem and is not.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run unprivileged. A model-scoring service has no reason to be root.
RUN useradd --create-home --uid 1000 app
WORKDIR /srv
COPY --from=builder /deps /srv/deps
ENV PYTHONPATH=/srv/deps:/srv

COPY --chown=app:app api.py train.py ./
COPY --chown=app:app src ./src
COPY --chown=app:app artifacts ./artifacts

USER app
EXPOSE 8000

# No --reload, no --workers>1 note: the model is loaded once per process, so
# multiple workers means multiple copies in RAM. For 300 trees that is fine; for
# a much larger model, serve one process behind a load balancer instead.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
