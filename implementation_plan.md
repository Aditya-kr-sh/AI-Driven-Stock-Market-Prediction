# Implementation Plan - Phase 1: Project Setup

Establish the foundations of the **AI-Driven Stock Market Prediction and Portfolio Optimization** system. This includes laying out the folder structure, configuring Python package metadata, setting up project configuration, defining dependencies, and preparing the Git repository.

---

## User Review Required

We are structuring the project to ensure that the core engineering logic in `ai_engine` can be packaged and installed as an editable module. This allows:
1. Local FastAPI code in `/backend` to import `ai_engine`.
2. Jupyter Notebooks in `/notebooks` to import `ai_engine`.
3. Scripts on the Ramanujan Universe HPC Cluster (RU) to import `ai_engine` seamlessly.

We will use standard Python `setuptools` via a modern `pyproject.toml` configuration to manage this package installation.

---

## Open Questions
No immediate blocker questions. We proceed with standard production-grade configuration structures.

---

## Proposed Changes

### Project Structure Setup

We will create all required directories and subdirectories matching the requested architecture layout.

#### [NEW] [.gitignore](file:///d:/stockproject/.gitignore)
Create a `.gitignore` containing comprehensive rules for:
- Python (`__pycache__`, virtual environments, `.pytest_cache`, `.eggs`, `.egg-info`)
- Data files, local database (SQLite `.db` files)
- Saved models (`saved_models/` weights, PKL, PT files, logs)
- Node/React build files (`node_modules`, `dist`, `.env.local`)
- IDE / OS artifacts (`.vscode`, `.idea`, `.DS_Store`)

#### [NEW] [pyproject.toml](file:///d:/stockproject/pyproject.toml)
Create package configuration for `ai_engine` using `setuptools` to enable editable installations (`pip install -e .`).

#### [NEW] [requirements.txt](file:///d:/stockproject/requirements.txt)
Define clean dependencies for the overall application:
- `yfinance` for NIFTY 50 data ingestion.
- `pandas`, `numpy`, `scikit-learn`, `scipy` for analytics.
- `xgboost`, `torch` for AI/ML modeling.
- `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings` for APIs and settings.
- `sqlalchemy` for SQLite database operations.
- `python-dotenv` for local environment configurations.

#### [NEW] [ai_engine Directories and Package Initializers](file:///d:/stockproject/ai_engine/)
Create the core directories with `__init__.py` files to make them importable packages:
- `ai_engine/data/`
- `ai_engine/features/`
- `ai_engine/models/`
- `ai_engine/training/`
- `ai_engine/portfolio/`
- `ai_engine/utils/`

#### [NEW] [ai_engine/utils/config.py](file:///d:/stockproject/ai_engine/utils/config.py)
A configuration class utilizing `pydantic-settings` to load project configuration from environment variables or a `.env` file. Settings will cover:
- SQLite Database path
- Model storage directory
- Default yfinance download settings
- Feature engineering configs (periods, indicator choices)
- Trading configuration (NIFTY 50 tickers, date ranges)

#### [NEW] [.env.example](file:///d:/stockproject/.env.example)
Provide template environment configuration for users setting up their Windows environment.

#### [NEW] [README.md](file:///d:/stockproject/README.md)
Provide an initial overview of the repository layout and setup instructions for Phase 1.

---

## Verification Plan

### Automated Verification
1. Run `pip install -e .` on Windows to confirm the package installs successfully in editable mode.
2. Run a simple script importing `ai_engine` and reading config options to verify configuration loading.

### Manual Verification
- Verify the directory hierarchy is fully formed.
- Ensure no unintended files are tracked by Git.
