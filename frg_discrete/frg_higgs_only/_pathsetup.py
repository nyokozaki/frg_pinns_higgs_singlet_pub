"""
Add the sibling folder frg_discrete to sys.path so its validated modules
(perturbation/, running_couplings.py, seed_potential.py, flow_equation.py)
can be imported.

This folder reuses the UV matching and seed potential from frg_discrete as-is,
and, instead of "shooting" (forward integration as an initial-value problem from
t=0), tries a relaxation method (in the sense of Numerical Recipes Sec. 17.3):
discretize space and time together and solve the whole trajectory globally.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRG_DISCRETE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "frg_discrete"))

if _FRG_DISCRETE_DIR not in sys.path:
    sys.path.insert(0, _FRG_DISCRETE_DIR)
