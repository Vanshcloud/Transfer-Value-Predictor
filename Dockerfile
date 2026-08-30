# ============================================================
# API image
# ============================================================
# Multi-stage: the build stage compiles wheels, the runtime stage carries only
# the installed packages. The saving is real — build tooling is larger than the
# application — but the bigger point is that a runtime image without a compiler
# has less to exploit.

FROM python:3.13-slim AS build

# Pinned to the same minor Python the project develops against
# (pyproject: requires-python >=3.13,<3.14). A silent 3.14 would change
# behaviour with no diff to point at.

WORKDIR /build

# Requirements first, and only requirements: Docker caches this layer, so
# editing source does not reinstall the whole ML stack.
# The SERVE lock, not the full one, and the lock rather than the declaration.
#
# Serving loads one model; the full set trains nine. xgboost brings 291 MB of
# CUDA libraries that a CPU inference path never opens, and catboost 269 MB of
# itself plus plotly — roughly 700 MB of a 1.6 GB site-packages, carried by an
# image whose only command is `uvicorn`. requirements-serve.txt states the
# coupling this creates: it is correct only while every shipped variant's
# winning family is one it lists, and tests/unit/test_dependencies.py fails
# in CI if that stops being true.
#
# The lock rather than the declaration because an image rebuilt in March must
# contain the same packages as one built today, or "it worked when we shipped
# it" is unfalsifiable.
COPY requirements-serve-lock.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve-lock.txt


FROM python:3.13-slim AS runtime

# libgomp is what LightGBM and XGBoost link for OpenMP. On macOS this is
# `brew install libomp`; the manylinux wheels expect the system copy here.
# Without it, both import with "OSError: libgomp.so.1: cannot open shared
# object file" — the Linux twin of the libomp failure in the README.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# A non-root user, because the container serves untrusted HTTP input and there
# is no reason for the process to be able to write to its own image.
RUN useradd --create-home --uid 10001 app

COPY --from=build /install /usr/local

WORKDIR /app
COPY --chown=app:app src ./src
COPY --chown=app:app api ./api
COPY --chown=app:app configs ./configs

# scripts/ is deliberately absent. This image serves; it does not train. The
# pipeline scripts need xgboost, catboost and requests, none of which are
# installed here, so shipping them would mean shipping commands that fail with
# an ImportError. The README runs them on the host, which is also where the
# data they need lives.

# Mount points for the two things the image deliberately does not contain.
RUN mkdir -p /app/data/processed /app/models && chown -R app:app /app/data /app/models

USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# The health endpoint is unversioned precisely so an orchestrator can call it
# without knowing about API versions. It returns 200 while degraded, so this
# checks liveness; readiness is the `ready` field in the body.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
