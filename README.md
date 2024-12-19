# ML-foundations-pytorch

A repo in which to investigate various pytorch architectures in calssification of MNIST and imagenet

# Development

Instructions on how to run code for `ML-foundations-pytorch`.

## Pre-setup

Some configuration for development you may need to use, depending on the services you need to access for your project.

### Poetry (Required for local development)

1. Install [poetry](https://python-poetry.org/docs/) into your system (see `pyproject.toml` for version)
2. Run `poetry install`
3. Done! See the [poetry docs](https://python-poetry.org/docs/basic-usage/) for basic usage instructions.

### Weights and Biases (Required)

1. Modify `.env` to include your [Weights and Biases API key](https://wandb.ai/authorize). 

> [!CAUTION]
> Do NOT commit your modified .env file to Git.

```
WANDB_API_KEY=<your-api-key>
```

You will also need to set this variable even if you are using a docker container.

### GCP (Optional)

If you need to pull data from GCP using the CLI or Python API, ensure that you can access the Zelim bucket with your credentials.

#### Local

```shell
gcloud auth login
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT='model-training-1'
```

#### Docker 

If accessing GCP from inside the docker container, set environment variables as [described in the GCP docs](https://cloud.google.com/docs/authentication/provide-credentials-adc).

### Docker (Optional)

1. Install [docker](https://docs.docker.com/engine/install/)
2. Install [docker compose](https://docs.docker.com/compose/install/) 


## Installation

### Docker installation

1. `docker compose up --build` which should show a default output similar to:

```
 docker compose up --build                                                    
[+] Building 359.9s (9/10)                                                                                                                                                                                                                                                                    docker:desktop-linux
...                                                                                                                                                                                                                                          0.0s 
[+] Running 1/1
 ✔ Container ml_workflow  Recreated                                                                                                                                                                                                                                                                           0.1s 
Attaching to ml_workflow
ml_workflow  | 
ml_workflow  | ==========
ml_workflow  | == CUDA ==
ml_workflow  | ==========
ml_workflow  | 
ml_workflow  | CUDA Version 12.1.0
ml_workflow  | 
ml_workflow  | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
ml_workflow  | 
ml_workflow  | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
ml_workflow  | By pulling and using the container, you accept the terms and conditions of this license:
ml_workflow  | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
ml_workflow  | 
ml_workflow  | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
ml_workflow  | 
ml_workflow  | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
ml_workflow  |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
ml_workflow  |    https://docs.nvidia.com/datacenter/cloud-native/ .
ml_workflow  | 
ml_workflow  | *************************
ml_workflow  | ** DEPRECATION NOTICE! **
ml_workflow  | *************************
ml_workflow  | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
ml_workflow  |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
ml_workflow  | 
ml_workflow  | The currently activated Python version 3.10.12 is not supported by the project (~3.11).
ml_workflow  | Trying to find and use a compatible version. 
ml_workflow  | Using python3.11 (3.11.0)
ml_workflow  | 2024-07-23 10:28:14 INFO     Using device: cpu
ml_workflow  | 2024-07-23 10:28:14 INFO     Preparing data for ML model...
ml_workflow  | 2024-07-23 10:28:14 INFO     Training ML model...
ml_workflow  | 2024-07-23 10:28:14 INFO     Testing ML model...
ml_workflow  | 2024-07-23 10:28:14 INFO     Packaging ML model...
ml_workflow  | 2024-07-23 10:28:14 INFO     Evaluating ML model...
ml_workflow exited with code 0

```

### Poetry installation

1. `poetry install`
2. `poetry run python workflow/main.py`

```
$ poetry run python workflow/main.py
2024-07-23 11:44:13 INFO     Using device: cpu
2024-07-23 11:44:13 INFO     Preparing data for ML model...
2024-07-23 11:44:13 INFO     Training ML model...
2024-07-23 11:44:13 INFO     Testing ML model...
2024-07-23 11:44:13 INFO     Packaging ML model...
2024-07-23 11:44:13 INFO     Evaluating ML model...
```

## Project structure

```
ML-foundations-pytorchh
├── README.md                           <- The top level README for developers using this project.
│
├── data
│   ├── interim                         <- Intermediate data that has been transformed.
│   └── processed                       <- Data processed as part of the pipeline.
│
├── experiments                         <- Model experiments
│   └── runs                            <- Where model runs are saved, where each subfolder is a unique experiment containing checkpoints, results, figures.
│
├── reports                             <- Larger analysis results and reports that are not tied to experiments, such as model comparisons and PDFs.
│   └── figures                         <- Subfolder to contain any figures used in reports.
│
├── config                              <- Workflow configuration YAML files
│   └── yolov7                              <- YOLOv7 specific config files
│       ├── params.yaml                         <- Parameters to control training, testing, packaging and inference.
│       ├── data.yaml                           <- Parameters to control data processing, conversion, label creation etc.
│       ├── data_splits.yaml                    <- Parameters to control data splitting.
│       └── hyperparameters.yaml                <- Parameters to control model hyperparameters.
│
├── notebooks                           <- Jupyter notebooks for experimental work.
│
├── Dockerfile                          <- Dockerfile.
│
├── docker-compose.yml                  <- Docker compose file which runs workflow/main.py as an entrypoint.
│
├── pyproject.toml                      <- Project configuration file with package metadata and requirements for 
│                                           ml_foundations_pytorch and configuration for tools like black.
│
├── workflow                            <- Standardised workflow components to run key ML operations.
│   ├── main.py                             <- Runs the entire pipeline of train, test, packaging and evaluation.
│   ├── train.py                            <- Code to run data preprocessing before training e.g YOLO label creation.
│   ├── train.py                            <- Code to run model training.
│   ├── test.py                             <- Code to run model testing on a test/validation set.
│   ├── package.py                          <- Code to export a model as an ONNX file for downstream testing.          
│   └── evaluate.py                         <- Code to run deeper evaluation on a model checkpoint, such as model comparison, benchmarking and explainability.
│
├── .env                                <- For secrets, authentication tokens and environment variables.
│
├── .gitignore                          <- For ignoring files/directories that shouldn't be added to version control.
│
└── ml_foundations_pytorch     <- Source code for the project.
    │
    ├── __init__.py                     <- Makes ml_foundations_pytorch a Python module
    │
    ├── data                
    │   ├── __init__.py 
    │   ├── preprocessing.py            <- Custom functions for ml_foundations_pytorch specific pre-processing.
    │   ├── dataloader.py               <- Custom functions for ml_foundations_pytorch specific data loading.       
    │   └── augmentations.py            <- Custom functions for ml_foundations_pytorch specific augmentations.
    │
    ├── utils                           <- Miscellaneous helper functions like file utilities.
    │   ├── __init__.py 
    │   ├── gpu_utils.py                <- Custom functions for ml_foundations_pytorch specific GPU operations.     
    │   └── wandb_utils.py              <- Custom functions for ml_foundations_pytorch specific WandB operations.
    │
    ├── evaluate                
    │   ├── __init__.py 
    │   ├── metrics.py                  <- Custom metrics for ml_foundations_pytorch specific analysis.          
    │   └── compare.py                  <- Custom functions for ml_foundations_pytorch specific model comparisons.
    │
    └── visualisation                
        ├── __init__.py 
        ├── plots.py                    <- Custom functions for ml_foundations_pytorch specific plots.        
        └── video.py                    <- Custom functions for ml_foundations_pytorch specific video plotting. 
```