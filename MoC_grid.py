# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:39:58 2023

@author: Niclas
"""
# from scipy.optimize import minimize,root,least_squares
import matplotlib.pyplot as plt
import numpy as np
import cantera as ct

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
    
    






def initial_grid_det(ds,t0):
    
    CJspeed = 1969
    P1 = 1e5
    a2_eq = 1050
    T1 = 293.15
    u2 = CJspeed -a2_eq  #CJspeed-a2_eq
    T2 = 2950
    P2 = 16e5
    gamma_eq = 1.2
    u0 = 0
    s_CJ = 1e5
    
    def phi(x,t): 
        return 2/(gamma_eq+1)*(x/t/CJspeed-1) + u2/CJspeed

    def eta(x,t):
        return (gamma_eq-1)/(gamma_eq+1)*(x/t/CJspeed-1) + a2_eq/CJspeed
    
    def u(x,t):
        if  x/t > CJspeed:
            return u0 
        else:
            return np.heaviside(phi(x,t),0)*phi(x,t)*CJspeed + u0
    
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
    
    U_CJ = u2/a2_eq
    C_CJ = 1
    S_CJ = s_CJ/gamma_eq/ct.gas_constant
    
    t = [t0]
    x = [CJspeed/a2_eq * t0]
    U = [U_CJ]
    C = [C_CJ]
    S = [S_CJ]
    
    
    while 1:
        dt = ds#(ds**2 / (1+(U[-1]-C[-1])**2))**0.5
        dx = (U[-1]-C[-1]) * dt
        
        if x[-1] + dx < 0:
            break
        
        x.append(x[-1]+dx)
        t.append(t[-1]+dt)
        
        p_ = P(x[-1],t[-1]*a2_eq)
        u_ = phi(x[-1],t[-1]*a2_eq) * CJspeed
        c_ = eta(x[-1],t[-1]*a2_eq) * CJspeed
        s_ = s_CJ - ct.gas_constant * (np.log(p_/P2) + 2*gamma_eq/(gamma_eq-1)*np.log(c_/a2_eq))
        U.append(u_/a2_eq)
        C.append(c_/a2_eq)
        S.append(s_/gamma_eq/ct.gas_constant)
    
    return x,t,U,C,S
    
    
    
    
#%% test case Sod shock tube

gamma_eq = 1.2
beta = 1.85/30e-3
cf = 0.0063
C0 = 0.5

x_0 = np.linspace(0,1,50)

x = [x_0]
t = [np.zeros_like(x_0)]

rho = np.where(x_0 < 0.5, 1.0, 0.125)
U = [np.ones_like(x_0)*0]
p = np.where(x_0 < 0.5, 1.0, 0.1)
C = [(gamma_eq * p/rho)**0.5]
S = [np.zeros_like(x_0)]

plt.figure(1)
plt.cla()

plt.scatter(x[-1], t[-1])

while max(t[-1]) < 0.2:
    
    S_new = []; U_new = []; C_new = []; x_new = []; t_new = []
    x_ = x[-1]; t_ = t[-1]; U_ = U[-1]; C_ = C[-1]; S_ = S[-1]
    
    for i in range(len(x_)-1):
        
        x3,t3,U3,C3,S3,x4,t4,U4,C4,S4 = solve_new_point(x_[i:i+2],t_[i:i+2],U_[i:i+2],C_[i:i+2],S_[i:i+2],tol=1e-8)
        
        x_new.append(x3)
        t_new.append(t3)
        U_new.append(U3)
        C_new.append(C3)
        S_new.append(S3)
    
    x_new = [0, *x_new, 1]
    t_new = [t_new[0], *t_new, t_new[-1]]
    U_new = [0, *U_new, 0]
    C_new = [C_new[0], *C_new, C_new[-1]]
    S_new = [S_new[0], *S_new, S_new[-1]]
    
    plt.scatter(x_new, t_new)
    plt.pause(0.5)
    
    t.append(np.array(t_new))
    x.append(np.array(x_new))
    U.append(np.array(U_new))
    C.append(np.array(C_new))
    S.append(np.array(S_new))
    
    print(max(t[-1]))
    


plt.scatter(x[0], t[0])
plt.scatter(x[-1], t[-1])    
   
        
        