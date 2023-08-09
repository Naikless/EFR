# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:39:58 2023

@author: Niclas
"""
# from scipy.optimize import minimize,root,least_squares
import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
from collections import namedtuple
from scipy.interpolate import griddata

from efr import TaylorWave

T1 = 293.15
P1 = ct.one_atm
X1 = 'H2:42,O2:21,N2:79'
mech = 'Klippenstein_noCarbon.cti'
det = TaylorWave(T1, P1, X1, mech)

CJspeed = det.CJspeed
a2_eq = det.a2_eq
u2 = CJspeed - a2_eq  #CJspeed-a2_eq
T2 = det.postShock_eq.T
P2 = det.postShock_eq.P
gamma_eq = det.gamma_eq
u0 = 0
s_CJ = det.postShock_eq.entropy_mass
R = ct.gas_constant/det.postShock_eq.mean_molecular_weight

beta = 1.85/30e-3
cf = 0#0.0063
C0 = (1.4 * ct.gas_constant /28.87 * 293)**0.5 / a2_eq # normalized speed of sound at wall temperature

U_CJ = u2/a2_eq
C_CJ = 1
S_CJ = s_CJ/gamma_eq/R
C_end = 1/CJspeed -(gamma_eq-1)/(gamma_eq+1)/a2_eq

Point = namedtuple('Point', (field for field in 'xtUCS'),defaults=(0,0,0))

class PointList(list):
    def __init__(self,*args,**kwargs):
        list.__init__(self,*args, **kwargs)
    
    def __getattr__(self, name):
        return [getattr(p,name) for p in self]
        

def plot_point(p,radius=0.05):
    plt.scatter(p.x,p.t)
    t_Cplus = np.linspace(0, radius / ((1+(p.U+p.C)**2))**0.5, 10)
    plt.plot(p.x + (p.U+p.C) * t_Cplus, p.t + t_Cplus, 'k--')
    
    t_Cminus = np.linspace(0, radius / ((1+(p.U-p.C)**2))**0.5, 10)
    plt.plot(p.x + (p.U-p.C) * t_Cminus, p.t + t_Cminus, 'k-.')
    
    t_U = np.linspace(0, radius / ((1+(p.U)**2))**0.5, 10)
    plt.plot(p.x + p.U * t_U, p.t + t_U, 'k:')


def solve_new_point(p1,p2, tol=1e-5):
    x1,t1,U1,C1,S1 = p1
    x2,t2,U2,C2,S2 = p2
    
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
    
    while 1:
        
        if U4 == 0:
            x4 = x3
            t4 = (t2-t1) * (x4-x1)/(x2-x1) + t1
        else:
            x4 = ((x1*x3 - x1*t3*U4 + x1*t2*U4 - x3*x2 + t3*x2*U4 - t1*x2*U4) 
                  / (t2*U4 - t1*U4 + x1 - x2))
            t4 = - (x3 - x4 -t3*U4) / U4
        
        while 1:
            U4 = (x4-x1)/(x2-x1) * U1 +  (x2-x4)/(x2-x1) * U2
            C4 = (x4-x1)/(x2-x1) * C1 +  (x2-x4)/(x2-x1) * C2
            S4 = (x4-x1)/(x2-x1) * S1 +  (x2-x4)/(x2-x1) * S2
            
            if U4 == 0:
                x4_ = x3
                t4_ = (t2-t1) * (x4-x1)/(x2-x1) + t1
            else:
                x4_ = ((x1*x3 - x1*t3*U4 + x1*t2*U4 - x3*x2 + t3*x2*U4 - t1*x2*U4) 
                      / (t2*U4 - t1*U4 + x1 - x2))
                t4_ = - (x3 - x4 - t3*U4) / U4
            
            if abs(x4_ - x4) < tol and abs(t4_ - t4) < tol:
                break
            else:
                x4 = x4_; t4 = t4_
        
        U13 = (U1+U3)/2
        U23 = (U2+U3)/2
        
        C13 = (C1+C3)/2
        C23 = (C2+C3)/2
        
        C34 = (C3+C4)/2
        U34 = (U3+U4)/2
        
        S3 = (S4 - beta * 2 *cf *abs(U34) /(gamma_eq-1)/C34**2
              * (C34**2 - (gamma_eq-1)/2 * U34**2 - C0**2)
              * (t3-t4)
              )
         
        R_plus3 = (R_plus1 - beta * 2 *cf *abs(U13) /C13
                   * (C13**2 - (gamma_eq-1)/2 * U13**2 - C0**2 - U13*C13)
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
            x3 = x3_; t3 = t3_
    
    return Point(x3,t3,U3,C3,S3), Point(x4,t4,U4,C4,S4)
    


def initial_grid_det(ds_inner,ds_outer,t0):
    
    def phi(x,t): 
        return 2/(gamma_eq+1)*(x/t/CJspeed-1) + u2/CJspeed

    def eta(x,t):
        return (gamma_eq-1)/(gamma_eq+1)*(x/t/CJspeed-1) + a2_eq/CJspeed
    
    def u(x,t):
        if  x/t > CJspeed:
            return u0 
        else:
            return np.heaviside(phi(x,t),0)*phi(x,t)*CJspeed + u0
    
    def c(x,t):
        if x/t <= CJspeed - u2 * (gamma_eq+1)/2:
            return eta(CJspeed - u2 * (gamma_eq+1)/2,1) * CJspeed
        else:
            return eta(x,t) * CJspeed
    
    def T(x,t):
        if x/t > CJspeed:
            return T1
        elif x/t <= CJspeed - u2 * (gamma_eq+1)/2:
            return T2 * (eta(CJspeed - u2 * (gamma_eq+1)/2,1)*CJspeed/ a2_eq)**2
        else:
            return T2 * (eta(x,t)*CJspeed/ a2_eq)**2
    
    def P(x,t):
        if x/t > CJspeed:
            return P1
        elif x/t <= CJspeed - u2 * (gamma_eq+1)/2:
            return P2 * (eta(CJspeed - u2 * (gamma_eq+1)/2,1)*CJspeed/ a2_eq)**(2*gamma_eq/(gamma_eq-1))
        else:
            return P2 * (eta(x,t)*CJspeed/ a2_eq)**(2*gamma_eq/(gamma_eq-1))
    
    
    p0 = Point(CJspeed/a2_eq * t0, t0, U_CJ, C_CJ, S_CJ)
    pl = PointList([p0])
    
    while 1:
        p_old = pl[-1]
        if p_old.U > 0:
            ds = ds_inner
        else:
            ds = ds_outer
        dt = (ds**2 / (1+(p_old.U-p_old.C)**2))**0.5
        dx = (p_old.U-p_old.C) * dt
        
        if p_old.x + dx < 0:
            break
        
        x_new = p_old.x+dx
        t_new = p_old.t+dt
        
        p_ = P(x_new,t_new/a2_eq)
        u_ = u(x_new,t_new/a2_eq)
        c_ = c(x_new,t_new/a2_eq)
        s_ = s_CJ - R * (np.log(p_/P2) + 2*gamma_eq/(gamma_eq-1)*np.log(a2_eq/c_))
        
        p_new = Point(x_new, t_new, u_/a2_eq, c_/a2_eq, s_/gamma_eq/R )
        
        pl.append(p_new)

    
    return pl


#%%
def add_new_Cminus(plist,ds,**kwargs):
    
    dt = (ds**2 / (1+(CJspeed/a2_eq)**2))**0.5
    
    p_new = Point(CJspeed/a2_eq * (plist[0].t + dt), plist[0].t + dt, U_CJ, C_CJ, S_CJ)
    
    plist_new = PointList([p_new])
    
    
    
    
    for p_old in plist[1:]:
        # plt.scatter(x_[-1],t_[-1])
        p_new_cand,p4 = solve_new_point(p_old, p_new,**kwargs)
        if p_new_cand.x < 0:
            dt =  p_old.x / (p_old.U - p_old.C)
            p_new = Point(0, p_new.t + dt, u0/a2_eq, C_end, S_CJ)
            pl_new.append(p_new)
            break
        p_new = p_new_cand
        plist_new.append(p_new)
        
    # plt.scatter(x_[-1],t_[-1])    
    return plist_new

#%% inititial grid from isentropic solution
plt.cla()
# plt.ylim([0,0.1])
# plt.xlim([0,0.1])
# plt.plot(np.linspace(0,1,100),np.linspace(0,1,100)/CJspeed*a2_eq,'k--')
pl = initial_grid_det(4e-5,5e-4,0.001)
plt.scatter(pl.x,pl.t)

#%%     
ds = 5e-4

points = PointList([pl])

while np.max(points.x) <= 1:
    pl_new = add_new_Cminus(points[-1],ds)
    points.append(pl_new)
    print(f'x_max = {np.max(points.x)}')


#%% transform coordinates
tmax = np.max(points.t)
xvec = np.array(points.x).flatten()
tvec = np.array(points.t).flatten()
Uvec = np.array(points.U).flatten()
Cvec = np.array(points.C).flatten()
Svec = np.array(points.S).flatten()

tau = (tvec - xvec/CJspeed*a2_eq) / (tmax - xvec/CJspeed*a2_eq)


#%% interpolate results
x_eq, tau_eq = np.meshgrid(np.linspace(0, 1,100),np.linspace(0, 1, 100))

U_int = griddata((xvec, tau), Uvec, (x_eq,tau_eq))
# U_int = griddata((xvec, np.array(t).flatten()), np.array(U).flatten(), (x_eq,tau_eq))

S_int = griddata((xvec, tvec), Svec, (x_eq,tau_eq))
# S_int = griddata((xvec, tau), Svec, (x_eq,tau_eq))



