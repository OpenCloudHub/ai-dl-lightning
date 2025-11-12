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

<h1 align="center">Fashin MNIST - MLOps Demo</h1>

<p align="center">
    Pytorch Lightning fashion MNIST classification with MLOps pipeline featuring MLflow tracking and Ray for distributed training and serving.<br />
    <a href="https://github.com/opencloudhub"><strong>Explore OpenCloudHub »</strong></a>
  </p>
</div>

______________________________________________________________________

<details>
  <summary>📑 Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

______________________________________________________________________

<h2 id="about">🍷 About</h2>

This repository demonstrates an example implementation for classification using pytorch lighning and the FashinMNIST dataset. It showcases combining machine learning practices including experiment tracking, model registration, and containerized training and deployment and serves as demonstration within the OpenCloudHub project.\\

______________________________________________________________________

<h2 id="features">✨ Features</h2>

- 🔬 **Experiment Tracking**: MLflow integration with model registry
- 🐳 **Containerized Training**: Docker-based training environment with UV
- ⚡ **Distributed Training & Serving**: Ray for scalable workflows
- 🚀 **CI/CD Ready**: GitHub Actions workflows for automated training and CI
- 🧪 **Development Environment**: VS Code DevContainer setup

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

2. **Open in DevContainer** (Recommended)

   VSCode: `Ctrl+Shift+P` → `Dev Containers: Rebuild and Reopen in Container`

   Or **setup locally without DevContainer**:

   ```bash
      # Install UV
      curl -LsSf https://astral.sh/uv/install.sh | sh

      # Install dependencies
      uv sync --dev
   ```

3. **Start local MLflow tracking server**

   ```bash
      mlflow server --host 0.0.0.0 --port 8081
   ```

   Access at `http://localhost:8081`

4. **Start local Ray cluster**

   ```bash
      ray start --head
   ```

   Access dashboard at `http://127.0.0.1:8265`

You're now ready to develop, train and serve models locally!

### Training

**Basic training:**

```bash
python src/train.py --lr 0.005
```

or use the Job API like we would do in practise too

```bash
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python src/train.py
```


### Model Serving

Ensure you have a trained model to load either from local folder or from mlflow by setting the 'MODEL_URI' environment variable.

**Start the serving application:**

```bash
serve run src.serve:app model_uri="models:/ci.fashion-mnist-classifier/1" --reload
```
or even better and more production ready, run:
```bash
serve build src.serve:app -o src/serve_config.yaml
serve deploy src/serve_config.yaml
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
├── src
│   ├── _utils.py
│   ├── data.py
│   ├── model.py
│   ├── serve_config.py
│   ├── serve.py
│   ├── train.py
│   ├── tune.py
│   └── serve.py
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
