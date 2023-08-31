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
import sdtoolbox as sdt
from scipy.interpolate import LinearNDInterpolator

from efr import TaylorWave

T1 = 293.15
P1 = ct.one_atm
X1 = 'H2:42,O2:21,N2:79' #'C3H8:1,O2:5'
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
cf = 0.0146#0.0063#0.0146#0.0146#0.0063

T_w = 293
P_out = 1e5
P_crit = P_out/(2/(gamma_eq+1))**(gamma_eq/(gamma_eq-1))
gas = det.postShock_eq
gas.TP = T_w,gas.P
gamma = sdt.thermo.soundspeed_eq(gas)**2/gas.P*gas.density
C0 = (gamma * ct.gas_constant /gas.mean_molecular_weight * T_w)**0.5 / a2_eq # normalized speed of sound at wall temperature

U_CJ = u2/a2_eq
C_CJ = 1
S_CJ = s_CJ/gamma_eq/R
# C_end = 1 - (gamma_eq-1) * u2/a2_eq/2

#%%

Point = namedtuple('Point', (field for field in 'xtUCS'),defaults=(0,0,0))

class PointList(list):
    def __init__(self,*args,**kwargs):
        list.__init__(self,*args, **kwargs)
    
    def __getattr__(self, name):
        return [getattr(p,name) for p in self]
        

def plot_point(p,radius=5e-4,invert = False,**kwargs):
    sign = -1 if invert else 1
    
    plt.scatter(p.x,p.t,**kwargs)
    t_Cplus = np.linspace(0, radius / ((1+(p.U+p.C)**2))**0.5, 10)
    plt.plot(p.x + (p.U+p.C) * t_Cplus * sign, p.t + t_Cplus  *sign, 'k--')
    
    t_Cminus = np.linspace(0, radius / ((1+(p.U-p.C)**2))**0.5, 10)
    plt.plot(p.x + (p.U-p.C) * t_Cminus *sign, p.t + t_Cminus *sign, 'k-.')
    
    t_U = np.linspace(0, radius / ((1+(p.U)**2))**0.5, 10)
    plt.plot(p.x + p.U * t_U*sign, p.t + t_U*sign, 'k:')


def solve_new_point(p1,p2, tol=1e-5):
    x1,t1,U1,C1,S1 = p1
    x2,t2,U2,C2,S2 = p2
    
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
    
#%%

def initial_grid_det(ds,t0):
    
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
    p_new = p0
    
    while 1:
        p_old = pl[-1]
        dt = (ds**2 / (1+(p_old.U-p_old.C)**2))**0.5
        dx = (p_old.U-p_old.C) * dt
        
        # if p_old.x + dx < 0:
        #     break
        
        x_new = p_old.x+dx
        t_new = p_old.t+dt
        
        p_ = P(x_new,t_new/a2_eq)
        u_ = u(x_new,t_new/a2_eq)
        c_ = c(x_new,t_new/a2_eq)
        s_ = s_CJ - R * (np.log(p_/P2) + 2*gamma_eq/(gamma_eq-1)*np.log(a2_eq/c_))
        
        p_new = Point(x_new, t_new, u_/a2_eq, c_/a2_eq, s_/gamma_eq/R )
        
        if p_new.U <= 0:
            p = pl[-1]
            p_new = Point(0,-p.x/(p.U-p.C)+p.t,*p_new[2:])
            pl.append(p_new)
            break
        
        pl.append(p_new)

    
    return pl

#%%

def add_zero_point(p, tol=1e-5):
    x2,t2,U2,C2,S2 = p
    
    x3 = 0
    t3 = (x3 - x2) / (U2-C2) + t2
    U3 = 0
    
    R_minus2 = 2/(gamma_eq-1) * C2 - U2
        
    C3 = R_minus2 * (gamma_eq-1) / 2
    S3 = S_CJ
    U23 = (U2+U3)/2
    
    while 1:
        C23 = (C2+C3)/2
                
        R_minus3 = (R_minus2 - beta * 2 *cf *abs(U23) /C23
                   * (C23**2 - (gamma_eq-1)/2 * U23**2 - C0**2 - U23*C23)
                   * (t3-t2)
                   + C23 * (S3-S2)
                   )
        C3 = R_minus3 * (gamma_eq-1) / 2
        C23 = (C2+C3)/2
        
        t3_ = (x3 - x2) / (U23-C23) + t2
        
        if abs(t3_ - t3) < tol:
            break
        else:
            t3 = t3_
            
    return Point(x3,t3,U3,C3,S3)

#%%

def add_end_point(p1, p2, tol=1e-5):
    x1,t1,U1,C1,S1 = p1
    x2,t2,U2,C2,S2 = p2
              
    x3 = 1
    t3 = (x3 - x1) / (U1 + C1) + t1
    
    t4 = (t2 + t1)/2
    x4 = (x1 + x2)/2
    
    R_plus1 = 2/(gamma_eq-1) * C1 + U1
    
    U3 = C3 = R_plus1/(2/(gamma_eq-1) + 1)
    
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
        # U23 = (U2+U3)/2
        
        C13 = (C1+C3)/2
        # C23 = (C2+C3)/2
                
        S3 = (S4 - beta * 2 *cf *abs(U34) /(gamma_eq-1)/C34**2
              * (C34**2 - (gamma_eq-1)/2 * U34**2 - C0**2)
              * (t3-t4)
              )
         
        R_plus3 = (R_plus1 - beta * 2 *cf *abs(U13) /C13
                   * (C13**2 - (gamma_eq-1)/2 * U13**2 - C0**2 + U13*C13)
                   * (t3-t1)
                   + C13 * (S3-S1)
                   )
        
        # R_minus3 = (R_minus2 - beta * 2 *cf *abs(U23) /C23
        #            * (C23**2 - (gamma_eq-1)/2 * U23**2 - C0**2 - U23*C23)
        #            * (t3-t2)
        #            + C23 * (S3-S2)
        #            )
        
        # U3 = (R_plus3 - R_minus3)/2
        # C3 = (R_plus3 + R_minus3)/4 * (gamma_eq-1)
        U3 = C3 = R_plus3/(2/(gamma_eq-1) + 1)
        
        P = P2 * C3**(2*gamma_eq/(gamma_eq-1)) * np.exp(gamma_eq*(S_CJ - S3))
        if P < P_crit:
            print('not choked!')
            return None,None
            
            # P3 = P_out
            # C3 = (P3/P2 / np.exp(gamma_eq*(S_CJ - S3))) ** ((gamma_eq-1)/(2*gamma_eq))
            # U3 = R_plus3 - 2/(gamma_eq-1) *C3 
        
        U13 = (U1+U3)/2
        # U23 = (U2+U3)/2
        
        C13 = (C1+C3)/2
        # C23 = (C2+C3)/2
        
        # t3_ = (x1-x2 + t2*(U23-C23)-t1*(U13+C13)) / (U23-C23-U13-C13)
        t3_ = (x3 - x1) / (U13 + C13) + t1
        
        
        if abs(t3_ - t3) < tol:
            break
        else:
            if n > 100:
                raise Exception("Couldn't converge to new grid point, maybe reduce tmax.")
            t3 = t3_
            n += 1
        
    
    return Point(x3,t3,U3,C3,S3), Point(x4,t4,U4,C4,S4)
    
#     R_plus1 = 2/(gamma_eq-1) * C1 + U1
    
#     S3 = S
    
#     C_ = (p_out/P2 / np.exp(gamma_eq*(S_CJ-S_))) ** ((gamma_eq-1)/(2*gamma_eq))
#     U_ = R_plus - 2/(gamma_eq-1) * C_
    
#     if U_/C_ >= 1:
#         C_ = U_ = R_plus / (2/(gamma_eq-1) + 1)
    
#     return Point(x_,t_,U_,C_,S_)

#%%
def add_new_Cminus(p_start, plist,**kwargs):
    p_new = p_start
    
    plist_new = PointList([p_new])
    
    for p_old in plist:
        p_new,p4 = solve_new_point(p_old, p_new,**kwargs)
        plist_new.append(p_new)
        # plt.scatter(p_new.x,p_new.t)
    
    p0,p1,p2 = plist_new[:3]
    dist_p0p1 = ((p1.x-p0.x)**2 + (p1.t-p0.t)**2)**0.5
    dist_p1p2 = ((p2.x-p1.x)**2 + (p2.t-p1.t)**2)**0.5
    
    if dist_p0p1 > dist_p1p2:
        p_add = Point(*(np.array(p1) + np.array(p0))/2)
        plist_new = PointList([p0] + [p_add] + plist_new[1:])
        
    
    # dt =  p_new.x / (p_new.U - p_new.C) 
    # plist_new.append(Point(0, p_new.t + dt, 0, C_end, S_CJ))
        
    # plt.scatter(x_[-1],t_[-1])    
    return plist_new

#%% inititial grid from isentropic solution

# plt.ylim([0,0.1])
# plt.xlim([0,0.1])
# plt.plot(np.linspace(0,1,100),np.linspace(0,1,100)/CJspeed*a2_eq,'k--')
# pl = initial_grid_det(1e-5,1e-3,0.001)
pl = initial_grid_det(2e-6,0.001)

plt.figure('grid')
plt.cla()
plt.scatter(pl.x,pl.t)

#%%     
N = 300
dx = (1-pl[0].x)/N
tmax = 4
points = PointList([pl])
dt = dx / CJspeed * a2_eq 

for i in range(N):    
    pl = points[-1]   
    p_start = Point(pl[0].x + dx, pl[0].t + dt, U_CJ, C_CJ, S_CJ)
    
    pl_new = add_new_Cminus(p_start,points[-1][1:])
    zero_dist = 1 if np.max(pl_new.x) < 1 else 1
    if np.mod(len(points),zero_dist) == 0:
        p_zero = add_zero_point(pl_new[-1])
        if p_zero.t < tmax:
            pl_new.append(p_zero)
    
    points.append(pl_new)
    # print(f't_max = {np.max(points[-1].t)}')

#%% interpolate between last detonation C- and first expansion wave C-

last = points[-1]
skip = 100
p_new = add_end_point(last[skip+1], last[skip])[0]
pl_new = last

# interpolate some C- characteristics
for step in np.arange(0.1,1.1,0.2):
    p_start = Point(*(np.array(last[0])*(1-step) + np.array(p_new)*step))
    pl_new = [p for p in pl_new if p.t >= p_start.t]
    pl_new = add_new_Cminus(p_start,pl_new)
    points.append(pl_new)
    
#%% proceed to create new C- characteristics along the sonic boundary
last = points[-1]
skip = 20
p_new = add_end_point(last[skip+1], last[skip])[0]
pl_new = last
p_start = p_new

while 1:
    pl_new = add_new_Cminus(p_start,pl_new[skip+1:])
    points.append(pl_new)
    if len(pl_new) <= skip+1 or pl_new[-1].x > 1:
        break
    p_start = add_end_point(pl_new[skip+1], pl_new[skip])[0]
    if not p_start:
        break
        

#%% transform coordinates
xvec = np.concatenate((points.x))
tvec = np.concatenate((points.t))
Uvec = np.concatenate((points.U))
Cvec = np.concatenate((points.C))
Svec = np.concatenate((points.S))

tmax = np.max(tvec)

tau = (tvec - xvec/CJspeed*a2_eq) / (tmax - xvec/CJspeed*a2_eq)

plt.figure('grid')
plt.cla()
plt.scatter(xvec,tvec)

plt.figure('C')
plt.cla()
plt.scatter(xvec,tvec,c=Cvec)

plt.figure('S')
plt.cla()
plt.scatter(xvec,tvec,c=Svec)

plt.figure('U')
plt.cla()
plt.scatter(xvec,tvec,c=Uvec)

#%% interpolate results
x_eq, tau_eq = np.meshgrid(np.linspace(0, 1,100),np.linspace(0, 1,100))

U_int = griddata((xvec, tvec/tmax), Uvec, (x_eq,tau_eq))
# U_int = griddata((xvec, np.array(t).flatten()), np.array(U).flatten(), (x_eq,tau_eq))

S_int = griddata((xvec, tvec/tmax), Svec, (x_eq,tau_eq))
# S_int = griddata((xvec, tau), Svec, (x_eq,tau_eq))

C_int = griddata((xvec, tvec/tmax), Cvec, (x_eq,tau_eq))


T_int = (C_int*a2_eq)**2 / gamma_eq / R

P_int = P2 * C_int**(2*gamma_eq/(gamma_eq-1)) * np.exp(gamma_eq*(S_CJ - S_int))

plt.figure('heatmap')
plt.cla()
plt.imshow(P_int,origin='lower')
# plt.scatter(xvec,tvec,c=Svec)

# plt.imshow(U_int,origin='lower')





