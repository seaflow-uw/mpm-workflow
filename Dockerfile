ARG ARG_PYTHON_VERSION=3.8

FROM python:${ARG_PYTHON_VERSION}-slim-bookworm

RUN apt-get update -qq \
    && apt-get install -qq -y build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --quiet --disabled-password --gecos '' seaflow

ENV PYTHONUNBUFFERED=1

RUN python -m venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN pip install -U setuptools wheel pip

ARG REQUIREMENTS_FILE
RUN test -n "$REQUIREMENTS_FILE" || (echo "REQUIREMENTS_FILE not set" && false)
COPY "$REQUIREMENTS_FILE" requirements.txt
RUN pip install --no-cache-dir --compile -r requirements.txt

COPY --chmod=755 fit_models.py fit_models.py

CMD ["bash"]
