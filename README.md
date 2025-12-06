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

This repository demonstrates a **demo MLOps implementation** for image classification using **PyTorch Lightning** and the **Fashion MNIST** dataset. It showcases best practices for combining modern ML engineering tools including:

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
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      GitHub                                                  │
│                                                                                              │
│  ┌────────────────────────────────────────────────────┐    ┌──────────────────────────────┐  │
│  │              ai-dl-lightning (this repo)           │    │   data-registry (DVC repo)   │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────┐  │    │  ┌────────────────────────┐  │  │
│  │  │CI Code Quality│ │CI Docker Build│ │MLOps Pipeline│  │    │  │  .dvc files (pointers) │  │  │
│  │  └──────────────┘ └──────────────┘ └─────┬──────┘  │    │  │  fashion-mnist-v1.0.0  │  │  │
│  └──────────────────────────────────────────┼─────────┘    │  │  fashion-mnist-v1.1.0  │  │  │
│                                             │              │  └────────────────────────┘  │  │
│                                             │              └───────────────┬──────────────┘  │
└─────────────────────────────────────────────┼──────────────────────────────┼─────────────────┘
                                              │                              │
                      Trigger Argo Workflow   │     dvc.api.get_url()        │
                                              ▼     (resolve data paths)     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Kubernetes Cluster                                         │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              Argo Workflows                                            │  │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────────────────────┐  │  │
│  │  │  Resolve   │───▶│   Train    │───▶│  Evaluate  │───▶│     Deploy / Promote       │  │  │
│  │  │ DVC Data   │    │  (RayJob)  │    │   Model    │    │       (Optional)           │  │  │
│  │  │  Version   │    │            │    │            │    │                            │  │  │
│  │  └─────┬──────┘    └─────┬──────┘    └────────────┘    └────────────────────────────┘  │  │
│  │        │                 │                                                             │  │
│  │        │ DVC_DATA_VERSION│                                                             │  │
│  │        │ (env var)       │                                                             │  │
│  └────────┼─────────────────┼─────────────────────────────────────────────────────────────┘  │
│           │                 │                                                                │
│           │                 ▼                                                                │
│  ┌────────┼───────────────────────────────────────────────────────────────────────────────┐  │
│  │        │            Ray Cluster (KubeRay)                                              │  │
│  │        │                                                                               │  │
│  │        │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │        │  │                         Training Job                                    │  │  │
│  │        │  │                                                                         │  │  │
│  │        │  │   ┌──────────────────────────────────────────────────────────────────┐  │  │  │
│  │        │  │   │  Ray Train Driver                                                │  │  │  │
│  │        │  │   │  • Load data via DVC (dvc.api) ◀───────────────────────┐         │  │  │  │
│  │        │  │   │  • Fetch normalization params (mean, std)              │         │  │  │  │
│  │        │  │   │  • Log to MLflow (metrics, params, artifacts)          │         │  │  │  │
│  │        │  │   │  • Register model to MLflow Registry                   │         │  │  │  │
│  │        └──┼───┼────────────────────────────────────────────────────────┼─────────┼──┼──┘  │
│  │           │   └──────────────────────────────────────────────────────┘  │         │  │    │
│  │           │                           │                                 │         │  │    │
│  │           │        Shard data         ▼                                 │         │  │    │
│  │           │   ┌───────────────────────────────────────────┐             │         │  │    │
│  │           │   │  ┌─────────┐  ┌─────────┐  ┌─────────┐    │             │         │  │    │
│  │           │   │  │Worker 0 │  │Worker 1 │  │Worker N │    │◀── Parquet  │         │  │    │
│  │           │   │  │  (DDP)  │  │  (DDP)  │  │  (DDP)  │    │    from S3  │         │  │    │
│  │           │   │  └─────────┘  └─────────┘  └─────────┘    │             │         │  │    │
│  │           │   └───────────────────────────────────────────┘             │         │  │    │
│  │           │                                                             │         │  │    │
│  │           └─────────────────────────────────────────────────────────────┘         │  │    │
│  │                                                                                   │  │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐  │  │    │
│  │  │                         Serving Deployment                                  │  │  │    │
│  │  │                                                                             │  │  │    │
│  │  │  Load model from ──▶ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │  │    │
│  │  │  MLflow Registry     │  Replica 1  │  │  Replica 2  │  │  Replica N  │      │  │  │    │
│  │  │                      │  (FastAPI)  │  │  (FastAPI)  │  │  (FastAPI)  │      │  │  │    │
│  │  │  Fetch norm params ◀─┴─────────────┴──┴─────────────┴──┴─────────────┴──────┼──┘  │    │
│  │  │  from DVC metadata   ▲                                                      │     │    │
│  │  │                      │ Autoscale (1-N replicas)                             │     │    │
│  │  └──────────────────────┼──────────────────────────────────────────────────────┘     │    │
│  └─────────────────────────┼────────────────────────────────────────────────────────────┘    │
│                            │                                                                 │
│  ┌─────────────────────────┼─────────────────────────────────────────────────────────────┐   │
│  │                         │           Platform Services                                 │   │
│  │  ┌──────────────────┐   │   ┌───────────────────────────────┐   ┌──────────────────┐  │   │
│  │  │     MLflow       │   │   │           MinIO               │   │      Istio       │  │   │
│  │  │  ┌────────────┐  │   │   │  ┌─────────────────────────┐  │   │     Gateway      │  │   │
│  │  │  │ Tracking   │  │   │   │  │    s3://dvcstore/       │◀─┼───┘                  │  │   │
│  │  │  │ Server     │  │   │   │  │  ┌───────────────────┐  │  │                      │  │   │
│  │  │  ├────────────┤  │   │   │  │  │ fashion-mnist/    │  │  │   External Traffic   │  │   │
│  │  │  │ Model      │  │   │   │  │  │  train.parquet    │  │  │         ▲            │  │   │
│  │  │  │ Registry   │  │   │   │  │  │  val.parquet      │  │  │         │            │  │   │
│  │  │  └────────────┘  │   │   │  │  │  metadata.json    │◀─┼──┼─ normalization params│  │   │
│  │  └──────────────────┘   │   │  │  └───────────────────┘  │  │                      │  │   │
│  │                         │   │  │                         │  │                      │  │   │
│  │                         │   │  │  ray-results/ (ckpts)   │  │                      │  │   │
│  │                         │   │  └─────────────────────────┘  │                      │  │   │
│  │                         │   └───────────────────────────────┘                      │  │   │
│  └─────────────────────────┼──────────────────────────────────────────────────────────┘  │   │
│                            │                                                              │   │
└────────────────────────────┼──────────────────────────────────────────────────────────────┘   │
                             │                                                                  │
                             └──────────────────────────────────────────────────────────────────┘
```

### Key Data Flows

1. **Data Versioning (DVC)**

   - Data is versioned in separate `data-registry` repo with Git tags (e.g., `fashion-mnist-v1.0.0`)
   - DVC `.dvc` files point to Parquet files stored in MinIO (`s3://dvcstore/`)
   - `metadata.json` contains normalization parameters (mean, std) computed from training set

1. **Training Flow**

   - Argo Workflow resolves DVC version → sets `DVC_DATA_VERSION` env var
   - Ray Train Driver fetches data paths via `dvc.api.get_url()`
   - Parquet files loaded directly from S3 into Ray Data, sharded across workers
   - Model + normalization params logged to MLflow with `dvc_data_version` tag

1. **Serving Flow**

   - Model loaded from MLflow Registry
   - `dvc_data_version` tag extracted from training run metadata
   - Normalization params fetched from DVC `metadata.json` (same version as training)
   - **This ensures training-serving consistency!**

1. **Traceability**

   - Every MLflow run tagged with: `argo_workflow_uid`, `docker_image_tag`, `dvc_data_version`
   - Model can be traced back to: exact data version, Docker image, and workflow run

______________________________________________________________________

<h2 id="getting-started">🚀 Getting Started</h2>

### Prerequisites

- Docker
- VS Code with DevContainers extension (recommended)
- Access to MLflow tracking server (local or remote)
- Access to MinIO/S3 for DVC data (optional for local development)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/opencloudhub/ai-dl-lightning.git
   cd ai-dl-lightning
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

1. **Configure environment variables**

   ```bash
   # For local Docker Compose setup
   source .env.docker

   # For Minikube/Kubernetes setup
   source .env.minikube
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

**Basic training (local):**

```bash
python src/training/train.py --lr 0.005 --max-epochs 5
```

**Using Ray Job API (production-like):**

```bash
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- \
    python src/training/train.py --batch-size 128 --lr 0.001 --max-epochs 10 --num-workers 2
```

**Training CLI Arguments:**

| Argument        | Default        | Description             |
| --------------- | -------------- | ----------------------- |
| `--run-name`    | auto-generated | MLflow run name         |
| `--batch-size`  | 128            | Training batch size     |
| `--lr`          | 0.001          | Learning rate           |
| `--max-epochs`  | 2              | Maximum training epochs |
| `--num-workers` | from config    | Number of Ray workers   |

### Model Serving

**Development mode (with hot reload):**

```bash
serve run src.serving.serve:app_builder model_uri="models:/dev.fashion-mnist-classifier/1" --reload
```

**Production mode (using config file):**

```bash
# Build serve config
serve build src.serving.serve:app_builder -o src/serving/serve_config.yaml

# Deploy
serve deploy src/serving/serve_config.yaml
```

Access:

- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Model info: `http://localhost:8000/info`

### Testing

**Test the deployed endpoint:**

```bash
python tests/test_mnist_classifier.py
```

This runs comprehensive tests including:

- Health check validation
- Batch predictions with real Fashion MNIST images
- Single image predictions
- Error handling for invalid inputs
- Different batch sizes

### Production Training

Trigger the MLOps pipeline via GitHub Actions:

1. Navigate to [Actions → MLOps Pipeline](https://github.com/OpenCloudHub/ai-dl-lightning/actions/workflows/train.yaml)
1. Click "Run workflow"
1. Configure parameters:
   - `dvc_data_version`: Data version tag (e.g., `fashion-mnist-v1.0.0`)
   - `compute_type`: CPU/GPU configuration
   - `comparison_metric`: Metric for model comparison (default: `val_acc`)
   - `comparison_threshold`: Minimum improvement threshold

______________________________________________________________________

<h2 id="configuration">⚙️ Configuration</h2>

### Environment Variables

The application uses `pydantic-settings` for configuration management. All settings can be overridden via environment variables.

#### Training Configuration (`src/training/config.py`)

| Variable                       | Default                        | Description                           |
| ------------------------------ | ------------------------------ | ------------------------------------- |
| `MLFLOW_TRACKING_URI`          | *required*                     | MLflow tracking server URL            |
| `MLFLOW_EXPERIMENT_NAME`       | `fashion-mnist`                | MLflow experiment name                |
| `MLFLOW_REGISTERED_MODEL_NAME` | `dev.fashion-mnist-classifier` | Model registry name                   |
| `RAY_STORAGE_ENDPOINT`         | `http://minio...`              | S3/MinIO endpoint for Ray checkpoints |
| `RAY_STORAGE_PATH`             | `ray-results`                  | S3 path for Ray checkpoints           |
| `RAY_NUM_WORKERS`              | `1`                            | Default number of training workers    |
| `DVC_REPO_URL`                     | GitHub URL                     | DVC data registry repository          |

#### CI/CD Data Contract (Workflow Tags)

These environment variables are **required** and set by Argo Workflows in production:

| Variable            | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `ARGO_WORKFLOW_UID` | Unique identifier for the Argo workflow run              |
| `DOCKER_IMAGE_TAG`  | Docker image tag used for training (for reproducibility) |
| `DVC_DATA_VERSION`  | Data version from DVC (e.g., `fashion-mnist-v1.0.0`)     |

For local development, set these to `"DEV"` in your `.env.docker` or `.env.minikube` file.

#### Serving Configuration (`src/serving/config.py`)

| Variable             | Default | Description                        |
| -------------------- | ------- | ---------------------------------- |
| `REQUEST_MAX_LENGTH` | `1000`  | Maximum batch size for predictions |

### Environment Files

- **`.env.docker`**: For local Docker Compose development
- **`.env.minikube`**: For Minikube/Kubernetes development

______________________________________________________________________

<h2 id="project-structure">📁 Project Structure</h2>

```
ai-dl-lightning/
├── src/
│   ├── training/                       # Training pipeline
│   │   ├── train.py                    # Main training script (Ray Train + MLflow)
│   │   ├── model.py                    # PyTorch Lightning model (ResNet18)
│   │   ├── data.py                     # DVC data loading with Ray Data
│   │   └── config.py                   # Training configuration (pydantic-settings)
│   ├── serving/                        # Model serving (Ray Serve + FastAPI)
│   │   ├── serve.py                    # Ray Serve deployment with FastAPI
│   │   ├── schemas.py                  # Pydantic request/response schemas
│   │   ├── config.py                   # Serving configuration
│   │   └── serve_config.yaml           # Ray Serve deployment config
│   └── _utils/                         # Shared utilities
│       └── logging.py                  # Rich logging configuration
├── tests/
│   └── test_mnist_classifier.py        # API integration tests
├── .github/workflows/                  # CI/CD workflows
│   ├── ci-code-quality.yaml            # Ruff linting and pre-commit checks
│   ├── ci-docker-build-push.yaml       # Multi-stage Docker builds
│   └── train.yaml                      # MLOps pipeline trigger
├── .devcontainer/                      # VS Code DevContainer config
├── Dockerfile                          # Multi-stage build (training + serving)
├── pyproject.toml                      # Project dependencies (UV)
├── uv.lock                             # Dependency lock file
├── .env.docker                         # Local Docker Compose environment
└── .env.minikube                       # Minikube/K8s environment
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

Project Link: [https://github.com/opencloudhub/ai-dl-lightning](https://github.com/opencloudhub/ai-dl-lightning)

______________________________________________________________________

<h2 id="acknowledgements">🙏 Acknowledgements</h2>

- [MLflow](https://mlflow.org/) - ML lifecycle management and model registry
- [Ray](https://ray.io/) - Distributed computing, training, and serving
- [PyTorch Lightning](https://lightning.ai/) - Deep learning framework
- [DVC](https://dvc.org/) - Data version control
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager
- [Argo Workflows](https://argoproj.github.io/argo-workflows/) - Kubernetes-native workflow orchestration

<p align="right">(<a href="#readme-top">back to top</a>)</p>
