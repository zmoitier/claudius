[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4281267.svg)](https://doi.org/10.5281/zenodo.4281267) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# claudius: Analytic computations for scattering

Python toolbox to compute scattered and total field when we have analytical computation so typically when scatters are invariant by rotation and the incident field is a plane wave.

The word _claudius_ is an acronym for _CaLculs AnalytiqUes pour la DIffUSion des ondes_ the French translation of analytic computations for scattering or it can also be an acronym for _Computing anaLyticAlly and Uniquely Diverse fIeld Used in Scattering_.

## Requirements

- require: [Numba](https://github.com/numba/numba), [NumPy](https://github.com/numpy/numpy), and [SciPy](https://github.com/scipy/scipy)
- optional: [Matplotlib](https://github.com/matplotlib/matplotlib) (it is only use for plotting)
- development: [black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort), [pytest](https://github.com/pytest-dev/pytest), and [pytest-cov](https://github.com/pytest-dev/pytest-cov)

## Install

### With PyPI

```bash
python3 -m pip install --user claudius
```

### From GitHub

```bash
git clone https://github.com/zmoitier/claudius.git
```

## Short description

### 2D/3D Helmholtz

### 3D Maxwell

## To Do

- Docs
- 3D Maxwell

## Acknowledgment

I would like to thank [Camille Carvalho](https://github.com/carvalhocamille) and Friedelinde for helping with the name and [Matthias Bussonnier](https://github.com/Carreau) for helping with PyPI.
