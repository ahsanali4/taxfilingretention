FROM python:3.11-slim-buster as python-base
LABEL authors="ahsanali4"
RUN adduser --disabled-password worker
ENV PIP_NO_CACHE_DIR=off \
    PYTHON_PATH=/usr/local/lib/python3.11/site-packages/

RUN apt-get update && apt-get install libpq-dev gcc -y
ENV PATH="/root/.local/bin:$VENV_PATH/bin:$PATH"
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
FROM python-base as poetry
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y curl build-essential \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry --version \
    && poetry config virtualenvs.create false \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --only main --no-root \
    && poetry cache clear --all --no-interaction .

FROM python-base
WORKDIR /app
COPY --from=poetry /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=poetry $PYTHON_PATH $PYTHON_PATH
COPY ./taxfilingretention taxfilingretention
COPY ./saved_models saved_models
RUN chown -R worker:worker /app
USER worker

ENV PYTHONPATH="$PYTHON_PATH:${PYTHONPATH}"
CMD uvicorn taxfilingretention.app:app --host 0.0.0.0 --port $PORT

