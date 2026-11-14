# MVPURE-PY package
<div style="text-align: center;">
<img width="256" height="256" alt="mvpure_py logo" src="https://github.com/user-attachments/assets/ee880cfc-bb19-486b-be65-8c59f61c9e87" />
</div>

EEG/MEG Multi-Source Localizers with **mvpure-py**  
Extends [MNE-Python](https://mne.tools/) workflows with multi-source neural activity indices.  
Based on [*“Multi-source neural activity indices for EEG/MEG localization: A two-stage spatial filtering framework and extension to MNE-Python”*](https://www.sciencedirect.com/science/article/pii/S1053811926004416).

**Repository:** [GitHub](https://github.com/julia-jurkowska/mvpure-tools) \
**PyPI:** https://pypi.org/project/mvpure-py/ \
**Tutorial & online docs:** [mvpure_py documentation](https://julia-jurkowska.github.io/mvpure-tools)

---

## Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
   - [Using pip](#using-pip)
   - [Using conda](#using-conda)
   - [Optional dependencies](#optional-dependencies)
   - [Installing from source](#installing-from-source)
- [Verify the installation](#verify-the-installation)
- [Tutorials and examples](#tutorials-and-examples)
- [Citation](#citation)
- [License](#license)
---
## Overview

The **mvpure-tools** package implements multi-source neural activity indices for EEG / MEG source localization, extending MNE-Python’s API. It enables improved reconstruction when sources may be correlated, overcoming limitations of single-source beamforming methods (e.g., LCMV) in certain settings.

Key contributions include:

- Compact algebraic forms for multi-source neural activity indices
- Seamless integration into MNE-Python workflows (covariance, forward/inverse handling, visualization)
- A step-by-step tutorial using real EEG data (oddball paradigm) and synthetic data

---

## Features

- Neural activity indices (MAI_MVP) in the multi-source setting with fixed source orientations
- Iterative localization algorithms compatible with large candidate source sets
- Built on top of MNE-Python, so you can combine it with standard EEG / MEG toolchains
- Tutorials and example scripts included

---
## Requirements

- **Python 3.13 or newer.**
- The package's required Python dependencies are installed automatically by pip or conda. They include MNE-Python, NumPy, SciPy, pandas, scikit-learn, and supporting libraries.

---
## Installation

Choose **one** of the installation methods below.

### Using pip

Install the latest published release from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install mvpure-py
```

For an isolated installation, create and activate a virtual environment first:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install mvpure-py
```

### Using conda

`mvpure-py` is a pure-Python package on PyPI. It can also be installed with the **conda-pypi** channel, which makes supported PyPI wheels available through `conda install`.

```bash
# Update conda, if needed
conda self update

# Create an environment and install mvpure-py
conda create -n mvpure-py -c conda-forge -c conda-pypi python=3.13 mvpure-py

# Activate the environment
conda activate mvpure-py
```
See the [conda-pypi installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/install-packages-from-pypi.html) for details and troubleshooting.

### Optional dependencies

The package defines optional extras for visualization and testing. Install them with pip as needed:

```bash
# Interactive / 3D visualization dependencies
python -m pip install "mvpure-py[visualization]"

# Test dependencies
python -m pip install "mvpure-py[test]"
```

The `visualization` extra includes PySide6, PyVista, and PyVistaQt. The `test` extra installs pytest.

### Installing from source

To work with the development version, clone the repository and install it in editable mode:

```bash
git clone https://github.com/julia-jurkowska/mvpure-tools.git
cd mvpure-tools
python -m pip install -e .
```

To include the test dependencies in the editable installation:

```bash
python -m pip install -e ".[test]"
```

Dependencies are declared in `pyproject.toml`, so installing the project also installs its required dependencies.

## Verify the installation

Run the following command in the environment where you installed the package:

```bash
python -c "import mvpure_py; from mvpure_py.localizer import localize; print(mvpure_py.__version__)"
```

The distribution name used by pip/conda is **`mvpure-py`**; the Python import package is **`mvpure_py`**.
---
## Tutorials and examples

The [online documentation](https://julia-jurkowska.github.io/mvpure-tools/) contains the API reference and step-by-step tutorials. Topics include:

1. EEG preprocessing for an oddball paradigm (filtering, artifact rejection, ICA, and epoching).
2. Estimating the number of sources and data rank.
3. Multi-source localization using neural activity indices.
4. Reconstructing source time courses.
5. Simulating and localizing multi-source EEG activity.
---

## Citation

If you use `mvpure-py` in your research, please cite:

> Jurkowska, J., Dreszer, J., Lewandowska, M., Tolpa, K., & Piotrowski, T. (2026). Multi-source neural activity indices for EEG/MEG localization: A two-stage spatial filtering framework and extension to MNE-Python. *NeuroImage, 339*, 122126. https://doi.org/10.1016/j.neuroimage.2026.122126

## License

This project is distributed under the [BSD 3-Clause License](LICENSE).