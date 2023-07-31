# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 11:35:28 2023

@author: Niclas
"""

import cantera as ct
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata, interp2d
from efr import TaylorWave
import cantera as ct
import numpy as np


def TW_ODE(tau,y, dS_dxi):
    """

    Parameters
    ----------
    tau : float
        normalized integration time.
    y : ndarray of float
        Solution vector with [R_plus,R_minus,S]'. Can be a stacked vector.

    Returns
    -------
    dy_dtau.

    """
    
    # extract variables from stacked solution vector
    R_plus = y[0]
    R_minus = y[1]
    
    # calculate normalized fields from Rieman variables
    U = (R_plus - R_minus)/2
    C = (R_plus + R_minus) /4 * (gamma_eq-1)
    
    # DGL
    dS_dtau = -beta * 2 * cf * abs(U) / (gamma_eq-1) / C**2 * ((gamma_eq-1)/2 * U**2 + C**2 - C0**2)
    dR_plus_dtau = C * (dS_dtau + dS_dxi) + (gamma_eq-1) * C * dS_dtau # approximating D+-/Dtau as D/Dtau +- d/dxi |_tau
    dR_minus_dtau = C * (dS_dtau - dS_dxi) + (gamma_eq-1) * C * dS_dtau
    
    dy_dtau = [dR_plus_dtau, dR_minus_dtau, dS_dtau]
    
    return dy_dtau
    

def MoC_step(y0,dS_dxi,tau):       
    
    y = solve_ivp(TW_ODE, tau, y0, args=(dS_dxi,))
    
    # extract variables from stacked solution vector
    R_plus = np.split(y,3)[0]
    R_minus = np.split(y,3)[1]
    S = np.split(y,3)[2]
    
    return R_plus,R_minus,S


# %%
# det = TaylorWave(293.15, ct.one_atm, 'H2:42, O2:21, N2:79', mech='gri30.yaml')

# gamma_eq = det.gamma_eq
# c_CJ = det.a2_eq
# V_CJ = det.CJspeed
# C0 = (1.4 * ct.gas_constant /28.87 * 293)**0.5 / c_CJ # normalized speed of sound at wall temperature

# beta = 1.85/30e-3
# cf = 0#0.0063

# %%
# values at CJ plane
S_CJ = det.postShock_eq.entropy_mass / gamma_eq / (ct.gas_constant/det.postShock_eq.mean_molecular_weight)
U_CJ = V_CJ/c_CJ - 1
C_CJ = 1

# initial grid
xi_0 = np.linspace(0,1,20)
spacing = 0.1 # approx "radial" spacing between C- lines

# initialize grid
xi = [xi_0]
tau = [np.zeros_like(xi_0)]


#%% 1. define test case Sod shock tube

rho = np.where(xi_0 < 0.5, 1.0, 0.125)
U = [np.ones_like(xi_0)*0]
p = np.where(xi_0 < 0.5, 1.0, 0.1)
C = [(gamma_eq * p/rho)**0.5]
S = [np.zeros_like(xi_0)]


#%% 2. find new xi and tau for C+ characteristic
C_plus = U[-1] + C[-1]
xi_ = xi[-1] + spacing / (1/(C_plus)+1)**0.5
xi_[xi_ > 1] = 1

tau_ = tau[-1] + 1/C_plus * (xi_ - xi[-1])

# xi.append(xi_)
# tau.append(tau_)


# 3. calculate Rieman invariants 
R_plus = 2/(gamma_eq-1) * C[-1] + U[-1]
R_minus = 2/(gamma_eq-1) * C[-1] - U[-1]

#%% 4. interpolate S on a equidistant grid to obtain dS_dxi
xi_vec = np.array(xi).flatten()
tau_vec = np.array(tau).flatten()
S_vec = np.array(S).flatten()
XI, TAU = np.meshgrid(np.linspace(0,1,100), np.linspace(0,np.array(tau).max(),100))

# try 2D interpolation on equidistant xi, tau grid
try:
    S_interp = griddata((xi_vec, tau_vec), S_vec, (XI, TAU), method='linear')
    dS_interp_dxi = np.gradient(S_interp, 1/100, axis=1)
    dS_interp_dxi[np.isnan(dS_interp_dxi)] = 0
    dS_dxi = interp2d(XI, TAU, dS_interp_dxi)
    dS_dxi = np.array([dS_dxi(xi,tau) for xi,tau in zip(xi_,tau_)]) 
except: # just use the 1D gradient over latest xi grid points 
    dS_dxi = np.gradient(S[-1], xi[-1])

#%% 5. integrate along C+

y = []
for R_p, R_m, S_, dtau, dS_dxi_  in zip(R_plus,R_minus,S[-1], tau_ - tau[-1], dS_dxi):
    
    y0 = [R_p,R_m,S_]
    
    results = solve_ivp(TW_ODE, [0,dtau], y0, args=(dS_dxi_,))
    
    y.append(results.y[:,1])

y = np.array(y)
R_plus = y[:,0]

def find_new_point()

















# y0 = np.concatenate([R_plus,R_minus,S])


# # calculate Riemann variables


# # calculate normalized fields from Rieman variables
# U = (R_plus - R_minus)/2
# C = (R_plus + R_minus) /4 * (gamma_eq-1)
    