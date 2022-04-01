FROM ubuntu:20.04

ARG REQUIREMENTS_FILE
ENV PYTHONUNBUFFERED=1

RUN test -n "$REQUIREMENTS_FILE" || (echo "REQUIREMENTS_FILE not set" && false)

RUN apt-get update -qq \
    && apt-get install -qq -y build-essential git python3 python3-pip parallel \
    && rm -rf /var/lib/apt/lists/*
RUN adduser --quiet --disabled-password --gecos '' ubuntu

RUN pip3 install --no-cache-dir --upgrade pip setuptools
COPY "$REQUIREMENTS_FILE" requirements.txt
RUN pip3 install --no-cache-dir -r ./requirements.txt

CMD ["bash"]
