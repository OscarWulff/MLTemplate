# machinelearningtemplate

Standardized template for Machine Learning Project

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

### Prerequisites

Make sure you have the following installed:
- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/machinelearningtemplate.git
    cd machinelearningtemplate
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Downloading the Data

To download the data, follow these steps:

1. Install `gdown`:
    ```sh
    pip install gdown
    ```

2. Create a `data/raw` directory in the root of the project:
    ```sh
    mkdir -p data/raw
    ```

3. Download the data files using `gdown`:
    ```sh
    gdown --folder 'https://drive.google.com/drive/folders/1ddWeCcsfmelqxF8sOGBihY9IU98S9JRP?usp=sharing' -O data/raw
    ```
