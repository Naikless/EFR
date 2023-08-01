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
        
        p_ = P(x[-1],t[-1]/a2_eq)
        u_ = phi(x[-1],t[-1]/a2_eq) * CJspeed
        c_ = eta(x[-1],t[-1]/a2_eq) * CJspeed
        s_ = s_CJ - ct.gas_constant * (np.log(p_/P2) + 2*gamma_eq/(gamma_eq-1)*np.log(c_/a2_eq))
        U.append(u_/a2_eq)
        C.append(c_/a2_eq)
        S.append(s_/gamma_eq/ct.gas_constant)
    
    return x,t,U,C,S
    
    
x,t,U,C,S = initial_grid_det(1e-2,0.1)   