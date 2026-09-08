# config_params.py

import numpy as np

# FRG / potential physical parameters

k_IR = 150.0
t = -2.0    # range of t from 0
t_range = t

mssq = -90.0 **2
lamS   = 0.27
lamHS  = 1.0 # mixing parameter

# thermal
T_RAW = 200.0
tau_uv    = (T_RAW/k_IR)*np.exp(t)   # T_RAW/k_UV
finite_T  = True
fixed_tau = False

mh = 125.2
vew = 174.1 # <H>

lamH = 0.25*mh**2/vew**2
mhsq = -vew**2 * 2.0 * lamH

lamS_min = lamH * mssq**2/mhsq**2
print('lamS_min=', lamS_min, 'lamS=', lamS)
tmp1= mssq + lamHS * vew**2
print('Singlet mass (GeV)=', np.sqrt(np.clip(tmp1,0,None)))

# Tree level values at UV scale
#aH     = (mhsq/k_IR**2) * np.exp(2.0*t)
#aS     = (mssq/k_IR**2) * np.exp(2.0*t)
