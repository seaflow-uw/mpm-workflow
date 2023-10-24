ARG ARG_PYTHON_VERSION=3.8
ARG ARG_BUILDER_IMAGE=${ARG_PYTHON_VERSION}-bookworm
ARG ARG_RUNTIME_IMAGE=${ARG_PYTHON_VERSION}-slim-bookworm

FROM python:${ARG_BUILDER_IMAGE} as builder

RUN apt-get update -qq \
    && apt-get install -qq -y build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

RUN python -m venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN pip install -U setuptools wheel pip

ARG REQUIREMENTS_FILE
RUN test -n "$REQUIREMENTS_FILE" || (echo "REQUIREMENTS_FILE not set" && false)
COPY "$REQUIREMENTS_FILE" requirements.txt
RUN pip install --no-cache-dir --compile -r requirements.txt

FROM python:${ARG_RUNTIME_IMAGE} as runtime

ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN adduser --quiet --disabled-password --gecos '' seaflow

COPY --from=builder requirements.txt requirements.txt
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --chmod=755 fit_models.py fit_models.py

CMD ["bash"]
