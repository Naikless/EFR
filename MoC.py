# -*- coding: utf-8 -*-
"""
Created on Thu Sep 31 23:03:28 2023

MoC for one-dimensional, unsteady and non-isentropic detonation wave. 
Algorithm based on:
    J.H. Skinner - Plane-flame simulation of the wake behind an internally 
    propelled vehicle Part 1, Simulation of a supersonic vehicle by a 
    detonation
    
    M.I. Radulescu & R.K. Hanson - Effect of Heat Loss on 
    Pulse-Detonation-Engine Flow Fields and Performance - 10.2514/1.10286
    
    K. Kawane et al. - The influence of heat transfer and friction on the 
    impulse of a detonation tube - 10.1016/j.combustflame.2011.02.017

@author: Niclas
"""

import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
from collections import namedtuple
from scipy.interpolate import griddata
import sdtoolbox as sdt
from scipy.interpolate import LinearNDInterpolator
from efr import TaylorWave

# values are normalized:
#   t = t' * c_CJ / L  # corresponds to tau in above papers, t' is time
#   x = x' / L         # corresponds to xi in above papers, x' is space
#   U = u / c_CJ
#   C = c / c_CJ
#   S = s / s_CJ


# Points are named tuples with x, t, U, C, S
Point = namedtuple('Point', (field for field in 'xtUCS'),defaults=(0,0,0))

class PointList(list):
    """
    List of points that allows for access to each of the fields as a list.
    """
    def __init__(self,*args,**kwargs):
        list.__init__(self,*args, **kwargs)
    
    def __getattr__(self, name):
        return [getattr(p,name) for p in self]
        

def plot_point(p,radius=5e-4,invert = False,**kwargs):
    """
    Visualize point as scatter with lines indicating the characteristics.

    Parameters
    ----------
    p : Point
        Point to plot.
    radius : Float, optional
        Length of the characteristics around the point. The default is 5e-4.
    invert : Bool, optional
        Invert direction of characteristics. The default is False.
    **kwargs : Dict
        Keyword arguments for matplotlibs scatter function.

    Returns
    -------
    None.

    """
    sign = -1 if invert else 1
    
    plt.scatter(p.x,p.t,**kwargs)
    t_Cplus = np.linspace(0, radius / ((1+(p.U+p.C)**2))**0.5, 10)
    plt.plot(p.x + (p.U+p.C) * t_Cplus * sign, p.t + t_Cplus  *sign, 'k--')
    
    t_Cminus = np.linspace(0, radius / ((1+(p.U-p.C)**2))**0.5, 10)
    plt.plot(p.x + (p.U-p.C) * t_Cminus *sign, p.t + t_Cminus *sign, 'k-.')
    
    t_U = np.linspace(0, radius / ((1+(p.U)**2))**0.5, 10)
    plt.plot(p.x + p.U * t_U*sign, p.t + t_U*sign, 'k:')

class MoC:
    """
    Object to contain the calculated fields from the MoC.
    """
    def __init__(self, det : TaylorWave, beta, cf=None, T_w=None, P_out=None):
        self.det = det
        self.beta = beta
        self.cf = cf if cf else det.cf
        
        T_w = T_w if T_w else det.T1
        P_out = P_out if P_out else det.P1
        
        CJspeed = det.CJspeed
        a2_eq = det.a2_eq
        u2 = CJspeed - a2_eq  #CJspeed-a2_eq
        self.T2 = det.postShock_eq.T
        self.P2 = det.postShock_eq.P
        self.gamma_eq = det.gamma_eq
        s_CJ = det.postShock_eq.entropy_mass
        R = ct.gas_constant/det.postShock_eq.mean_molecular_weight
        
        self.P_crit = P_out/(2/(self.gamma_eq+1))**(self.gamma_eq/(self.gamma_eq-1))
        self.u0 = det.u0
        
        gas = det.postShock_eq
        gas.TP = T_w,gas.P
        gamma = sdt.thermo.soundspeed_eq(gas)**2/gas.P*gas.density
        self.C0 = (gamma * ct.gas_constant /gas.mean_molecular_weight * T_w)**0.5 / a2_eq # normalized speed of sound at wall temperature
        
        self.U_CJ = u2/a2_eq
        self.C_CJ = 1
        self.S_CJ = s_CJ/self.gamma_eq/R
        

    def solve_new_point(self, p1, p2, tol=1e-5):
        """
        Calculate new internal point. Based on the algorithm by Skinner.

        Parameters
        ----------
        p1 : Point
            Left point.
        p2 : Point
            Right Point.
        tol : Float, optional
            Convergence tolerance. The default is 1e-5.

        Returns
        -------
        tuple(P3 : Point, P4 : Point)
            New internal point P3 and interpolated U-characteristic point P4.

        """
        x1,t1,U1,C1,S1 = p1
        x2,t2,U2,C2,S2 = p2
        gamma_eq, cf, beta, C0 = self.gamma_eq, self.cf, self.beta, self.C0
        
        # first approximation assumes constant characteristic slopes and invariants, i.e. isentopropic flow
        t3 = (x1-x2 + t2*(U2-C2)-t1*(U1+C1)) / (U2-C2-U1-C1)
        x3 = (U1+C1) * (t3-t1) + x1
        
        t4 = (t2 + t1)/2
        x4 = (x1 + x2)/2
        
        R_plus1 = 2/(gamma_eq-1) * C1 + U1
        R_minus2 = 2/(gamma_eq-1) * C2 - U2
        
        U3 = (R_plus1 - R_minus2)/2
        C3 = (R_plus1 + R_minus2) /4 * (gamma_eq-1)
        
        # approx point 4
        U4 = 0.5*(U1+U2)
        U34 = (U3+U4)/2
        
        n = 0
        while 1:
            
            t4 = (((t2-t1)/(x2-x1) * (x3 - U34*t3 - x1) + t1) /
                  (1 - (t2-t1)/(x2-x1) * U34))
            x4 = x3 - U34 * (t3-t4)
            
            n4 = 0
            while 1:
                U4 = (x4-x1)/(x2-x1) * U1 +  (x2-x4)/(x2-x1) * U2
                C4 = (x4-x1)/(x2-x1) * C1 +  (x2-x4)/(x2-x1) * C2
                S4 = (x4-x1)/(x2-x1) * S1 +  (x2-x4)/(x2-x1) * S2
                
                U34 = (U3+U4)/2
                C34 = (C3+C4)/2
                
                t4_ = (((t2-t1)/(x2-x1) * (x3 - U34*t3 - x1) + t1) /
                      (1 - (t2-t1)/(x2-x1) * U34))
                x4_ = x3 - U34 * (t3-t4_)
                
                
                if abs(x4_ - x4) < tol and abs(t4_ - t4) < tol:
                    break
                else:
                    if n4 > 100:
                        raise Exception("Couldn't converge to new grid point, maybe reduce tmax.")
                    x4 = x4_; t4 = t4_
                    n4 += 1
            
            U13 = (U1+U3)/2
            U23 = (U2+U3)/2
            
            C13 = (C1+C3)/2
            C23 = (C2+C3)/2
                    
            S3 = (S4 - beta * 2 *cf *abs(U34) /(gamma_eq-1)/C34**2
                  * (C34**2 - (gamma_eq-1)/2 * U34**2 - C0**2)
                  * (t3-t4)
                  )
             
            R_plus3 = (R_plus1 - beta * 2 *cf *abs(U13) /C13
                       * (C13**2 - (gamma_eq-1)/2 * U13**2 - C0**2 + U13*C13)
                       * (t3-t1)
                       + C13 * (S3-S1)
                       )
            
            R_minus3 = (R_minus2 - beta * 2 *cf *abs(U23) /C23
                       * (C23**2 - (gamma_eq-1)/2 * U23**2 - C0**2 - U23*C23)
                       * (t3-t2)
                       + C23 * (S3-S2)
                       )
            
            U3 = (R_plus3 - R_minus3)/2
            C3 = (R_plus3 + R_minus3)/4 * (gamma_eq-1)
            
            U13 = (U1+U3)/2
            U23 = (U2+U3)/2
            
            C13 = (C1+C3)/2
            C23 = (C2+C3)/2
            
            t3_ = (x1-x2 + t2*(U23-C23)-t1*(U13+C13)) / (U23-C23-U13-C13)
            x3_ = (U13+C13) * (t3_-t1) + x1
            
            if abs(x3_ - x3) < tol and abs(t3_ - t3) < tol:
                break
            else:
                if n > 100:
                    raise Exception("Couldn't converge to new grid point, maybe reduce tmax.")
                x3 = x3_; t3 = t3_
                n += 1
        
        return Point(x3,t3,U3,C3,S3), Point(x4,t4,U4,C4,S4)

