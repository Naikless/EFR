#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:15:53 2023

@author: hanraths
"""
import cantera as ct
from efr import TaylorWave

T1 = 293.15
P1 = ct.one_atm
X1 = 'H2:32,O2:16,N2:84'#'H2:42,O2:21,N2:79' #'C3H8:1,O2:5'
mech = 'Klippenstein_noCarbon.cti'
det = TaylorWave(T1, P1, X1, mech)

L = 0.5
det.create_MoC(L, 3e-2, 2e-6, 300, cf=0.0063)

# def f(x,t):
#     return det.MoC.U(x/L,t/L*det.a2_eq) * det.a2_eq
# x,t = det.streamline(x0, t0, u_func=f, dt=1e-5)

x0, t0 = L, 1e-4
det.znd(relTol = 1e-8, absTol = 1e-15, Method='Radau')
# point_history = det.point_history(x0, t0, dt=1e-5)
profile = det.profile(t0, 1e-3, L=L, multiprocessing=True)
