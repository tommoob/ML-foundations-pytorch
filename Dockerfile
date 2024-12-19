FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get -y update \
    && apt-get install -y software-properties-common cmake build-essential  \
    && apt-get -y update \
    && add-apt-repository universe \
    && apt-get -y update \
    && apt-get -y install python3.11 \
    python3-pip python3-dev \
    python3.11-dev \
    && pip3 install --upgrade pip

ENV POETRY_VERSION=1.8.3

# Install poetry
RUN pip3 install "poetry==$POETRY_VERSION"

# Copy Python code to the Docker image
COPY . /code/ML-foundations-pytorchh

WORKDIR /code/ML-foundations-pytorchh

# Project initialization:
RUN poetry install --no-interaction

CMD [ "poetry", "run", "python", "workflow/main.py"]