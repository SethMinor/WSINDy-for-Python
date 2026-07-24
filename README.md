# WSINDy for PDEs
A Python 3 implementation of the Weak-form Sparse Identification of Nonlinear Dynamics (WSINDy) algorithm for partial differential equations.

Based on the [JCP paper by **D. A. Messenger**, **D. M. Bortz** (2021)](https://www.sciencedirect.com/science/article/pii/S0021999121004204).
- See the original authors' [**MatLab** code repository](https://github.com/MathBioCU/WSINDy_PDE)
- Also see the [**PyWSINDy for ODEs** code repository](https://github.com/MathBioCU/PyWSINDy_ODE)

For other existing implementations, also see the [**PySINDy** documentation](https://pysindy.readthedocs.io/en/latest/examples/12_weakform_SINDy_examples/example.html).
###### Stable as of June, 2025.
[![Python 3.11](https://img.shields.io/badge/python-%3E=3.11-blue?logo=python)](https://img.shields.io/badge/python-%3E=3.11-blue?logo=python)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SethMinor/PyWSINDy-for-PDEs/blob/main/wsindy_examples.ipynb)
---
![wsindy_github_pic](https://github.com/SethMinor/WSINDy-for-Python/assets/97004318/8e567430-7368-420c-bf94-6eee224f7dc5)

## Python Files [![Python 3.11](https://img.shields.io/badge/python-%3E=3.11-blue?logo=python)](https://img.shields.io/badge/python-%3E=3.11-blue?logo=python)
The core functionality of this codebase is contained within two files:
- ###### [`wsindy.py`](https://github.com/SethMinor/PyWSINDy-for-PDEs/blob/main/wsindy.py) <br> The fundamental WSINDy class definition.
- ###### [`helper_fcns.py`](https://github.com/SethMinor/PyWSINDy-for-PDEs/blob/main/helper_fcns.py) <br> A list of utilities and helper functions.

## Examples [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SethMinor/PyWSINDy-for-PDEs/blob/main/wsindy_examples.ipynb)
The [**`wsindy_examples.ipynb`**](https://github.com/SethMinor/PyWSINDy-for-PDEs/blob/main/wsindy_examples.ipynb) notebook illustrates the Weak SINDy algorithm being applied to various spatiotemporal systems:

- ###### [`KURAMOTO SIVASHINKSY`](https://en.wikipedia.org/wiki/Kuramoto%E2%80%93Sivashinsky_equation) <br> Numerical simulation of the $(1+1)$-dimensional Kuramoto-Sivashinksy equation (pictured above). See the [`KS.txt`](https://github.com/SethMinor/PyWSINDy-for-PDEs/blob/main/Datasets/KS.txt) file (1.3 MB). The data were sourced from [this GitHub repository](https://github.com/MathBioCU/WSINDy_PDE/blob/master/datasets/KS.mat).

- ###### [`SWIFT HOHENBERG`](https://en.wikipedia.org/wiki/Swift%E2%80%93Hohenberg_equation) <br> Numerical simulation of the $(2+1)$-dimensional Swift-Hohenberg $(23)$ equation. Simulated data were obtained using MatLab's [Chebfun](https://www.chebfun.org/examples/pde/SwiftHohenberg.html) package, see [`sh23_simulation.m`](https://github.com/SethMinor/PyWSINDy-for-PDEs/blob/main/Datasets/SH23_simulation.m).

- ###### [`MHD EQUATIONS`](https://turbulence.pha.jhu.edu/docs/README-MHD.pdf) <br> Numerical simulation of forced turbulence in the $(3+1)$-dimensional incompressible MHD equations, sourced from the [Johns Hopkins Tubulence Database](https://turbulence.pha.jhu.edu/Forced_MHD_turbulence.aspx).

*Note:* to access a dataset stored in Google Drive (e.g., `/content/drive/My Drive/WSINDy/dataset_name.txt`) while using Google Colab, use the following commands to change directories.
```python3
# Create directory if necessary
!mkdir -p "/content/drive/My Drive/WSINDy"

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/My Drive/WSINDy
```

## Install
###### In a Bash environment:
```python3
wget -q https://raw.githubusercontent.com/SethMinor/PyWSINDy-for-PDEs/main/wsindy.py
wget -q https://raw.githubusercontent.com/SethMinor/PyWSINDy-for-PDEs/main/helper_fcns.py
```

## Dependencies
###### See [`environment.yml`](https://github.com/SethMinor/PyWSINDy-for-PDEs/blob/main/environment.yml). This codebases uses the following modules and naming conventions:
```python3
import torch
import scipy
import numpy as np
import itertools
import symengine as sp

import torch.linalg as la
from scipy.signal import convolve
from scipy.special import factorial
import matplotlib.pyplot as plt
from tqdm import tqdm

from wsindy import *
from helper_fcns import *
```

## Basic Usage
###### For a dataset `U`, derivative library `alpha`, function library `beta`, and coordinate axes `X`, the syntax to create a class instance is:
```python3
model = wsindy(U, alpha, beta, X, **params)
```

###### Example hyperparameter specification:
```python3
# Coordinate axes (x,y,...,t)
X = [x,t]

# Candidate derivatives (dx,dy,...,dt)
alpha = [[0,1], # d/dt
         [0,0], # 1
         [1,0], # d/dx
         [2,0], # d^2/dx^2
         [3,0], # d^3/dx^3
         [4,0]] # d^4/dx^4

# Candidate monomial powers for (u1,...,ud) := u1
beta = [[0], # u1^0 = 1
        [1], # u1^1
        [2]] # u1^2

# Candidate monomial powers for (u1,...,ud) := (u1,u2)
beta = [[0,0], # u1^0 * u2^0 = 1
        [1,0], # u1^1 * u2^0
        [0,1]] # u1^0 * u2^1
```

###### Full list of hyperparameters:
```python3
params = {
    'V': [],           # Extra variables [u2,...,ud]
    'names': None,     # Variable names ['u_1',...,'u_d']
    'm': None,         # Test fcn support [mx,...,mt]
    'p': None,         # Test fcn degrees [px,...,pt]
    's': None,         # Subsampling [sx,...,st]
    'jacobian': 1.,    # Volume element
    'tau': 1e-10,      # Test fcn tolerance in real space
    'tau_hat': 2,      # Fourier decay in standard deviations
    'verbosity': True, # Print out details
    'rescale': True,   # Use preconditioner for LS solves
}
```

When `m` is omitted, spectral matching uses `tau_hat=2` by default and checks
that the learned changepoint satisfies the root-existence condition from
Appendix A. Degenerate spectra fall back to the largest valid support with a
warning; pass `m` explicitly to bypass automatic matching.

###### Create library of candidate terms and solve for sparse coefficients:
```python3
# Create standard library terms
[G,powers,derivs,rhs_names] = model.create_default_library()

# Set lhs
lhs_name = 'u' + model.derivative_names[0]
model.build_lhs(lhs_name)

# Set library
model.set_library(G, powers, derivs, rhs_names)

# Find sparse weights
w = model.MSTLS()
model.print_report()
```

## Library Customization
###### Specify a sequence of terms by applying derivative $\mathcal{D}^i$ to function $f_j(u_1, \dots, u_d)$:
```python3
G = [tensor1, ..., tensorN]
powers = [list1, ..., listN]
derivs = [int1, ..., intN]
rhs_names = [string1, ..., stringN]

model.set_library(G, powers, derivs, rhs_names)
```

###### Print library terms in formatted LaTeX:
```python3
display(Math(r'\Theta=' + r'\{' + r', \, '.join(rhs_names) + r'\}'))
```

###### For non-homogeneous functions, set the corresponding `powers[j] = [0, ..., 0]`. <br> Also see the following [helper functions](https://github.com/SethMinor/PyWSINDy-for-PDEs/blob/main/helper_fcns.py):
```python3
compute_weak_poly(...)
compute_weak_multipoly(...)
compute_weak_trig(...)
```

###### Convolved tensors should be evaluated over query points $\{(\boldsymbol{x}_k, t_k)\}$ using `tensor[model.conv_mask]`. For example:
```python3
term = compute_weak_poly(model.U, kernels, model.spacing, yu=model.yu, yxyt=yxyt)
evaluated_term = term[model.conv_mask]
```

###### An example of trimming terms out of the library:
```python3
# Trim out the redundant u_x term
remove_cols = [6]
print(f'Removing: {rhs_names[6]}\n')

for column in sorted(remove_cols, reverse=True):
  G.pop(column)
  powers.pop(column)
  derivs.pop(column)
  rhs_names.pop(column)

display(Math(r'\Theta=' + r'\{' + r',\,'.join(rhs_names) + r'\}'))
del remove_cols,column
```

###### An example of combining two columns in a library:
```python3
# Augment the library to include an advection term, (u·∇)ζ
print(f'Combining: {rhs_names[14]}, {rhs_names[31]}')

columns = [14,31]
coeffs = [1,1]
name = '(𝘂·∇)ζ'
info = [G, powers, derivs, rhs_names]

[G,powers,derivs,rhs_names] = composite_term(columns, coeffs, name, model, info)
```
