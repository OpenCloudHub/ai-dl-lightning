<a id="readme-top"></a>

<!-- PROJECT LOGO & TITLE -->

<div align="center">
  <a href="https://github.com/opencloudhub">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-light.svg">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-dark.svg">
    <!-- Fallback -->
    <img alt="OpenCloudHub Logo" src="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-dark.svg" style="max-width:700px; max-height:175px;">
  </picture>
  </a>

<h1 align="center">Fashion MNIST - MLOps Demo</h1>

<p align="center">
    End-to-end MLOps pipeline demonstrating distributed training with Ray, experiment tracking with MLflow, and production serving — all orchestrated through GitHub Actions and Argo Workflows.<br />
    <a href="https://github.com/opencloudhub"><strong>Explore OpenCloudHub »</strong></a>
  </p>
</div>

______________________________________________________________________

<details>
  <summary>📑 Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#thesis-context">Thesis Context</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

______________________________________________________________________

<h2 id="about">🎯 About</h2>

This repository demonstrates a **production-grade MLOps implementation** for image classification using **PyTorch Lightning** and the **Fashion MNIST** dataset. It showcases best practices for combining modern ML engineering tools including:

- **Experiment Tracking & Model Registry** with MLflow
- **Distributed Training** with Ray Train (DDP strategy)
- **Model Serving** with Ray Serve + FastAPI
- **Data Versioning** with DVC (backed by S3/MinIO)
- **CI/CD Automation** with GitHub Actions triggering Argo Workflows

The project serves as a reference implementation within the **OpenCloudHub** project, demonstrating how to build reproducible, scalable ML pipelines for Kubernetes environments.

______________________________________________________________________

<h2 id="thesis-context">📚 Thesis Context</h2>

This repository is part of a thesis project exploring **MLOps best practices for enterprise ML platforms**. Key concepts demonstrated:

### MLflow Integration
- **Experiment Tracking**: All training runs are logged with hyperparameters, metrics, and artifacts
- **Model Registry**: Trained models are registered with semantic versioning (`ci.fashion-mnist-classifier`)
- **Workflow Tagging**: CI/CD metadata (Argo Workflow UID, Docker image tag, DVC data version) is attached to each run for full traceability
- **Automatic Model Logging**: PyTorch Lightning models are logged with input examples for signature inference

### Data Versioning with DVC
- Training data is versioned in a separate [data-registry](https://github.com/OpenCloudHub/data-registry) repository
- Normalization parameters (mean, std) are stored in DVC metadata and fetched at training/serving time
- This ensures **training-serving consistency** — models always use the same preprocessing as during training

### Distributed Training with Ray
- **Ray Train** with `TorchTrainer` enables multi-worker distributed data parallel (DDP) training
- Checkpoints are stored in S3/MinIO for fault tolerance
- The driver/worker architecture separates orchestration (MLflow logging) from computation

### Production Serving
- **Ray Serve** provides autoscaling inference with FastAPI integration
- Models are loaded from MLflow registry with automatic normalization parameter fetching
- Hot-reload capability via `reconfigure()` for zero-downtime model updates

______________________________________________________________________

<h2 id="features">✨ Features</h2>

- 🔬 **Experiment Tracking**: MLflow integration with automatic metric logging and model registry
- 📊 **Data Versioning**: DVC-managed datasets with normalization metadata
- ⚡ **Distributed Training**: Ray Train with PyTorch Lightning DDP strategy
- 🚀 **Production Serving**: Ray Serve + FastAPI with autoscaling and health checks
- 🐳 **Multi-stage Docker**: Optimized images for training and serving with shared base layers
- 🔄 **CI/CD Pipeline**: GitHub Actions + Argo Workflows for automated training and deployment
- 🏷️ **Full Traceability**: Every model tagged with workflow UID, image tag, and data version
- 🧪 **Development Environment**: VS Code DevContainer with pre-configured tooling

______________________________________________________________________

<h2 id="architecture">🏗️ Architecture</h2>

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GitHub Actions                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │ CI Code Quality │  │ CI Docker Build  │  │     MLOps Pipeline          │ │
│  │   (Ruff/Pre-    │  │  (Training +     │  │  (Trigger Argo Workflow)    │ │
│  │    commit)      │  │   Serving)       │  │                             │ │
│  └─────────────────┘  └──────────────────┘  └──────────────┬──────────────┘ │
└────────────────────────────────────────────────────────────┼────────────────┘
                                                             │
                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        Argo Workflows                                   ││
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  ││
│  │  │  Resolve │───▶│   Train  │───▶│  Evaluate│───▶│  Deploy/Promote  │  ││
│  │  │   Data   │    │ (RayJob) │    │  Model   │    │    (Optional)    │  ││
│  │  └──────────┘    └────┬─────┘    └──────────┘    └──────────────────┘  ││
│  └───────────────────────┼─────────────────────────────────────────────────┘│
│                          │                                                  │
│  ┌───────────────────────▼─────────────────────────────────────────────────┐│
│  │                     Ray Cluster (KubeRay)                               ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │                    Training Job                                  │   ││
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                          │   ││
│  │  │  │ Worker 0│  │ Worker 1│  │ Worker N│  ◀── Ray Data Shards     │   ││
│  │  │  │  (DDP)  │  │  (DDP)  │  │  (DDP)  │                          │   ││
│  │  │  └────┬────┘  └────┬────┘  └────┬────┘                          │   ││
│  │  │       └────────────┼───────────-┘                               │   ││
│  │  │                    ▼                                             │   ││
│  │  │            Ray Train Driver                                      │   ││
│  │  │         (MLflow logging, checkpoints)                            │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                         ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │                    Serving Deployment                            │   ││
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   ││
│  │  │  │  Replica 1  │  │  Replica 2  │  │  Replica N  │ ◀── Autoscale│   ││
│  │  │  │  (FastAPI)  │  │  (FastAPI)  │  │  (FastAPI)  │              │   ││
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘              │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │    MLflow    │  │    MinIO     │  │  Istio       │                      │
│  │   Tracking   │  │  (S3/DVC)    │  │  Gateway     │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Data Flows

1. **Training Flow**: GitHub Action → Argo Workflow → RayJob → DVC Data → Train → MLflow Registry
2. **Serving Flow**: MLflow Model → Ray Serve Deployment → Istio Gateway → External Traffic
3. **Traceability**: Every MLflow run is tagged with `argo_workflow_uid`, `docker_image_tag`, `dvc_data_version`

______________________________________________________________________

<h2 id="getting-started">🚀 Getting Started</h2>

### Prerequisites

- Docker
- VS Code with DevContainers extension (recommended)

### Setup

1. **Clone the repository**

   ```bash
      git clone https://github.com/opencloudhub/ai-ml-lighting.git
      cd ai-ml-lighting
   ```

1. **Open in DevContainer** (Recommended)

   VSCode: `Ctrl+Shift+P` → `Dev Containers: Rebuild and Reopen in Container`

   Or **setup locally without DevContainer**:

   ```bash
      # Install UV
      curl -LsSf https://astral.sh/uv/install.sh | sh

      # Install dependencies
      uv sync --dev
   ```

1. **Start local MLflow tracking server**

   ```bash
      mlflow server --host 0.0.0.0 --port 8081
   ```

   Access at `http://localhost:8081`

1. **Start local Ray cluster**

   ```bash
      ray start --head --num-cpus 8
   ```

   Access dashboard at `http://127.0.0.1:8265`

You're now ready to develop, train and serve models locally!

### Training

**Basic training:**

```bash
python src/training/train.py --lr 0.005
```

or use the Job API like we would do in practise too

```bash
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python src/training/train.py
```

### Model Serving

Ensure you have a trained model to load either from local folder or from mlflow by setting the 'MODEL_URI' environment variable.

**Start the serving application:**

```bash
serve run src.serving.serve:app_builder model_uri="models:/ci.fashion-mnist-classifier/8" --reload
```

or even better and more production ready, run:

```bash
serve build src.serving.serve:app_builder -o src/serving/serve_config.yaml
serve deploy src/serving/serve_config.yaml
```

Access Swagger docs at `http://localhost:8000/docs`

### Testing

**Test the deployed endpoint:**

```bash
python tests/test_mnist_classifier.py
```

Or use the interactive Swagger UI at `http://localhost:8000/docs`

### Production Training

Trigger the workflow dispatch in Github Actions at `https://github.com/OpenCloudHub/ai-ml-sklearn/actions/workflows/train.yaml`

______________________________________________________________________

<h2 id="project-structure">📁 Project Structure</h2>

```
ai-ml-lightning/
├── src/
│   ├── training/                       # Training and optimization scripts
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── serving/                        # Model serving (Ray Serve/FastAPI)
│   │   └── serve.py
│   └── _utils/                         # Shared utilities
│       ├── logging_config.py
│       └── mlflow_tags.py              # Set up mandatory MLflow tags needed for production workflow
├── tests/                              # Unit tests
├── .devcontainer/                      # VS Code DevContainer config
├── .github/workflows/                  # CI/CD workflows
├── Dockerfile                          # Multi-stage container build
├── MLproject                           # MLflow project definition
├── pyproject.toml                      # Project dependencies and config
└── uv.lock                             # Dependency lock file
```

______________________________________________________________________

<h2 id="contributing">👥 Contributing</h2>

Contributions are welcome! This project follows OpenCloudHub's contribution standards.

Please see our [Contributing Guidelines](https://github.com/opencloudhub/.github/blob/main/.github/CONTRIBUTING.md) and [Code of Conduct](https://github.com/opencloudhub/.github/blob/main/.github/CODE_OF_CONDUCT.md) for more details.

______________________________________________________________________

<h2 id="license">📄 License</h2>

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

______________________________________________________________________

<h2 id="contact">📬 Contact</h2>

Organization Link: [https://github.com/OpenCloudHub](https://github.com/OpenCloudHub)

Project Link: [https://github.com/opencloudhub/ai-ml-lightning](https://github.com/opencloudhub/ai-ml-lightning)

______________________________________________________________________

<h2 id="acknowledgements">🙏 Acknowledgements</h2>

- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Ray](https://ray.io/) - Distributed computing and serving
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager

<p align="right">(<a href="#readme-top">back to top</a>)</p>
