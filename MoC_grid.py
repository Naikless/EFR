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
S_CJ = s_CJ/gamma_eq/ct.gas_constant

Point = namedtuple('Point', (field for field in 'xtUCS'))

def plot_point(x,t,U,C,S,radius=0.05):
    plt.scatter(x,t)
    t_Cplus = np.linspace(0, radius / ((1+(U+C)**2))**0.5, 10)
    plt.plot(x + (U+C) * t_Cplus, t + t_Cplus, 'k--')
    
    t_Cminus = np.linspace(0, radius / ((1+(U-C)**2))**0.5, 10)
    plt.plot(x + (U-C) * t_Cminus, t + t_Cminus, 'k-.')
    
    t_U = np.linspace(0, radius / ((1+(U)**2))**0.5, 10)
    plt.plot(x + U * t_U, t + t_U, 'k:')


def solve_new_point(x,t,U,C,S, tol=1e-5):
    x1,x2 = x
    t1,t2 = t
    U1,U2 = U
    C1,C2 = C
    S1,S2 = S
    
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
                   * (C13**2 - (gamma_eq-1/2) * U13**2 - C0**2 - U13*C13)
                   * (t3-t1)
                   + C13 * (S3-S1)
                   )
        
        R_minus3 = (R_minus2 - beta * 2 *cf *abs(U23) /C23
                   * (C23**2 - (gamma_eq-1/2) * U23**2 - C0**2 - U23*C23)
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
    
    return x3,t3,U3,C3,S3,x4,t4,U4,C4,S4
    


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
    
    t = [t0]
    x = [CJspeed/a2_eq * t0]
    U = [U_CJ]
    C = [C_CJ]
    S = [S_CJ]
    
    
    while 1:
        if U[-1] > 0:
            ds = ds_inner
        else:
            ds = ds_outer
        dt = (ds**2 / (1+(U[-1]-C[-1])**2))**0.5
        dx = (U[-1]-C[-1]) * dt
        
        if x[-1] + dx < 0:
            break
        
        x.append(x[-1]+dx)
        t.append(t[-1]+dt)
        
        p_ = P(x[-1],t[-1]/a2_eq)
        u_ = u(x[-1],t[-1]/a2_eq)
        c_ = c(x[-1],t[-1]/a2_eq)
        s_ = s_CJ - R * (np.log(p_/P2) + 2*gamma_eq/(gamma_eq-1)*np.log(a2_eq/c_))
        U.append(u_/a2_eq)
        C.append(c_/a2_eq)
        S.append(s_/gamma_eq/ct.gas_constant)
    
    return x,t,U,C,S


#%%
def add_new_Cminus(x,t,U,C,S,ds):
    dt = (ds**2 / (1+(CJspeed/a2_eq)**2))**0.5
    t_ = [t[0] + dt]
    x_ = [CJspeed/a2_eq * t_[0]]
    U_ = [U_CJ]
    C_ = [C_CJ]
    S_ = [S_CJ]
    
    
    for i in range(1,len(x)):
        # plt.scatter(x_[-1],t_[-1])
        x3,t3,U3,C3,S3,x4,t4,U4,C4,S4 = solve_new_point([x[i],x_[-1]],[t[i],t_[-1]],[U[i],U_[-1]],[C[i],C_[-1]],[S[i],S_[-1]],tol=1e-3)
        if x3 < 0:
            dt =  x[-1] / (U[-1]-C[-1])
            x_.append(0)
            t_.append(t_[-1] + dt)
            U_.append(u0/a2_eq)
            C_.append(1/CJspeed -(gamma_eq-1)/(gamma_eq+1)/a2_eq)
            S_.append(S_CJ)
            break
        x_.append(x3)
        t_.append(t3)
        U_.append(U3)
        C_.append(C3)
        S_.append(S3)
    # plt.scatter(x_[-1],t_[-1])    
    return x_,t_,U_,C_,S_

#%% inititial grid from isentropic solution
plt.cla()
# plt.ylim([0,0.1])
# plt.xlim([0,0.1])
# plt.plot(np.linspace(0,1,100),np.linspace(0,1,100)/CJspeed*a2_eq,'k--')
x,t,U,C,S = initial_grid_det(5e-6,1e-4,1e-4)
plt.scatter(x,t)

#%%     
ds = 1e-2

x = [x]
t = [t]
U = [U]
C = [C]
S = [S]

while np.max(x[-1]) <= 1:
    x_,t_,U_,C_,S_ = add_new_Cminus(x[-1],t[-1],U[-1],C[-1],S[-1],ds)
    x.append(x_)
    t.append(t_)
    U.append(U_)
    C.append(C_)
    S.append(S_)
    print(f'x_max = {np.max(x[-1])}')


#%% transform coordinates
tmax = np.max(t[-1])
tau = (np.array(t).flatten() - np.array(x).flatten()/CJspeed*a2_eq) / (tmax - np.array(x).flatten()/CJspeed*a2_eq)


#%% interpolate results
x_eq, tau_eq = np.meshgrid(np.linspace(0, 1,100),np.linspace(0, 1, 100))

# U_int = griddata((np.array(x).flatten(), tau), np.array(U).flatten(), (x_eq,tau_eq))
U_int = griddata((np.array(x).flatten(), np.array(t).flatten()), np.array(U).flatten(), (x_eq,tau_eq))

S_int = griddata((np.array(x).flatten(), tau), np.array(S).flatten(), (x_eq,tau_eq))



