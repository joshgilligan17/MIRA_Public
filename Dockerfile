# syntax=docker/dockerfile:1

FROM node:22-slim AS web-build

WORKDIR /app/webapp
COPY webapp/package*.json ./
RUN npm ci
COPY webapp/ ./
RUN npm run build


FROM python:3.11-slim AS runtime

ENV MIRA_JOB_ROOT=/data/mira/jobs \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[web]" \
    && python -m pip check

COPY --from=web-build /app/webapp/dist ./webapp/dist

ARG MIRA_UID=10001
ARG MIRA_GID=10001

RUN addgroup --system --gid "${MIRA_GID}" mira \
    && adduser --system --uid "${MIRA_UID}" --ingroup mira mira \
    && mkdir -p /data/mira/jobs \
    && chown -R mira:mira /data/mira /app

USER mira
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()"

CMD ["uvicorn", "structagent.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
