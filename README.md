# MVPURE-PY package
<div style="text-align: center;">
<img width="256" height="2516" alt="mvpure_py logo" src="https://github.com/user-attachments/assets/ee880cfc-bb19-486b-be65-8c59f61c9e87" />
</div>

EEG/MEG Multi-Source Spatial Filters with **mvpure-py**  
Extends **MNE-Python** with multi-source neural activity indices and spatial filters.  
Based on *“Multi-Source Neural Activity Indices and Spatial Filters for EEG/MEG Inverse Problem: An Extension to MNE-Python”*.

**Repository:** [GitHub](https://github.com/julia-jurkowska/mvpure-tools) \
**Tutorial & online docs:** [mvpure_py documentation](https://julia-jurkowska.github.io/mvpure-tools)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
    - From source
5. [Tutorial & Examples](#tutorial--examples)
6. [Citation](#citation)
---
## Overview

The **mvpure-tools** package implements multi-source neural activity indices and spatial filters for EEG / MEG source localization, extending MNE-Python’s API. It enables improved reconstruction when sources may be correlated, overcoming limitations of single-source beamforming methods (e.g. LCMV) in certain settings.

Key contributions include:

- Algebraic, compact forms for multi-source spatial filters and neural activity indices
- Automated parameter selection (e.g. suggested number of sources, rank estimation) based on eigenvalue spectra
- Seamless integration into MNE-Python workflows (covariance, forward/inverse handling, visualization)
- A step-by-step tutorial using real EEG data (oddball paradigm)

---

## Features

- Estimation of number of active sources and optimal rank via eigenvalue spectra
- Neural activity indices (MAI_MVP, MPZ_MVP) in the multi-source setting
- Spatial filters of reduced rank for reconstruction
- Iterative localization algorithms compatible with large candidate source sets
- Built on top of MNE-Python, so you can combine it with standard EEG / MEG toolchains
- Tutorial and example scripts included  

---
## Dependencies

Before installing, make sure you have:

- Python
- **MNE-Python**
- NumPy, SciPy
- (Other dependencies listed in `requirements.txt`)  

---

## Installation
### From Source (recommended)
To install from GitHub:

1. Clone the repository
```bash
git clone https://github.com/julia-jurkowska/mvpure-tools.git
cd mvpure-tools
```
2. (Optional) create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install the package
```bash
pip install .
```
or for development mode:
```bash
pip install -e .
```

5. Verify installation:

```python
import mvpure_py
from mvpure_py import localizer, beamformer
```
---
## Tutorial & Examples
See tutorials on documentation site: [documentation](https://julia-jurkowska.github.io/mvpure-tools).

The tutorials walk you through:

1. Preprocessing (filtering, artifact rejection, ICA, epoching)
2. Estimation of number of sources and rank
3. Localization of sources using neural activity indices
4. Reconstruction of time courses
5. Visualization of estimated sources
---

## Citation

If you use **mvpure_py** in your work, please cite:

> Jurkowska, J., Dreszer, J., Lewandowska, M., Tołpa, K., Piotrowski, T. (2025).  
> *Multi-Source Neural Activity Indices and Spatial Filters for EEG/MEG Inverse Problem: An Extension to MNE-Python*. (preprint)
