#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:21:11 2019

@author: hanraths
"""

import sdtoolbox.postshock as sdps
from sdtoolbox.znd import zndsolve
from sdtoolbox.thermo import soundspeed_eq
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

class Detonation:
    
    def __init__(self, T1, P1, q, mech='Klippenstein.cti'):
        self.T1 = T1
        self.P1 = P1
        self.composition = q
        self.mech = mech
        self.t_znd = 1e-3
        # the following are currently necessary to ensure working multiprocessing
        self._ct = ct
        self._sdps = sdps
        self._zndsolve = zndsolve
        self._np = np
    
    @property
    def postShock_eq(self):
        if hasattr(self, 'postShock_eq_state'):
            postShock_eq = self._ct.Solution(self.mech)
            postShock_eq.state = self.postShock_eq_state
            return postShock_eq
        else:
            U1 = self.CJspeed
            T1,P1,X1 = self.T1, self.P1, self.composition
            mech = self.mech
            postShock_eq = self._sdps.PostShock_eq(U1,P1,T1,X1,mech)
            self.postShock_eq_state = postShock_eq.state
            return postShock_eq
    

    def postShock_fr(self, U1=None):
        T1,P1,X1 = self.T1, self.P1, self.composition
        mech = self.mech
        if U1 is None:
            if hasattr(self, 'postShock_fr_state'):
                postShock_fr = self._ct.Solution(mech)
                postShock_fr.state = self.postShock_fr_state
                return postShock_fr
            else:
                U1 = self.CJspeed
                postShock_fr = self._sdps.PostShock_fr(U1, P1, T1, X1, mech)
                self.postShock_fr_state = postShock_fr.state
                return postShock_fr
        else:
            postShock_fr = self._sdps.PostShock_fr(U1, P1, T1, X1, mech)
            return postShock_fr
            
    
    
    @property
    def CJspeed(self):
        if hasattr(self, '_cj_speed'):
            return self._cj_speed
        else:
            self._cj_speed = self._sdps.CJspeed(self.P1, self.T1, self.composition, mech=self.mech)
            return self._cj_speed

    
    @property
    def znd(self):
        if hasattr(self, '_znd_out'):
             return self._znd_out
        else:
            U1 = self.CJspeed
            T1,P1,X1 = self.T1, self.P1, self.composition
            mech = self.mech
            gas1 = self._ct.Solution(mech)
            gas1.TPX = T1,P1,X1
            
            gas = self._sdps.PostShock_fr(U1,P1,T1,X1,mech)
            
            print('\nSolving ZND reactor. This might take a while...\n')
            znd_out = self._zndsolve(gas,gas1,U1,self.t_znd)
            znd_out['gas1'] = znd_out['gas1'].state
            self._znd_out = znd_out
            return self._znd_out
        
        

class TaylorWave(Detonation):
    
    def __init__(self, T1, P1, q, mech='Klippenstein.cti', u0=0):
        Detonation.__init__(self, T1, P1, q, mech)
        self.a2_eq = soundspeed_eq(self.postShock_eq)
        self.gamma_eq = self.a2_eq**2*self.postShock_eq.density/self.postShock_eq.P
        self.u2 = self.CJspeed-self.a2_eq
        self.u0 = u0
        self.mech = mech
        self.nu = 1 # polytropic ratio defined as 1 + dq / vdp, with dq defined by heat losses to the environment ("Thermodynamik", Baehr)
        # self.C_f = 0.0062    # heat transfer coefficient directly from DOI:10.2514/1.10286
    
    
    def point_history(self,x,t,dt=1e-6):
        """
        calculate time history of particle at location x at time t with time 
        resolution dt
        """
        CJspeed = self.CJspeed
        P1 = self.P1
        a2_eq = self.a2_eq
        T1 = self.T1
        u2 = CJspeed-a2_eq
        T2 = self.postShock_eq.T
        P2 = self.postShock_eq.P
        n = 1 / (1 - self.nu * (1 - 1/self.gamma_eq)) # polytropic exponent n = gamma/(gamma-nu(gamma-1)) ("Thermodynamik", Baehr)
        
        def phi(x,t): 
            return 2/(n+1)*(x/t/CJspeed-1) + u2/CJspeed
    
        def eta(x,t):
            return (n-1)/(n+1)*(x/t/CJspeed-1) + a2_eq/CJspeed
        
        def u(x,t):
            if  x/t > CJspeed:
                return self.u0 
            else:
                return self._np.heaviside(phi(x,t),0)*phi(x,t)*CJspeed + self.u0
        
        def T(x,t):
            if x/t > CJspeed:
                return T1
            elif x/t <= CJspeed - u2 * (n+1)/2:
                return T2 * (eta(CJspeed - u2 * (n+1)/2,1)*CJspeed/ a2_eq)**2
            else:
                return T2 * (eta(x,t)*CJspeed/ a2_eq)**2
        
        def P(x,t):
            if x/t > CJspeed:
                return P1
            elif x/t <= CJspeed - u2 * (n+1)/2:
                return P2 * (eta(CJspeed - u2 * (n+1)/2,1)*CJspeed/ a2_eq)**(2*n/(n-1))
            else:
                return P2 * (eta(x,t)*CJspeed/ a2_eq)**(2*n/(n-1))
        
        
        gas = self._ct.Solution(self.mech)
        Y_init = self.znd['species'][:,-1]
        
        x = [x]
        t = [t]
        while 1:
            if (t[-1] - dt)*CJspeed > x[-1]:         
                x.append(x[-1] - u(x[-1],t[-1]) * dt)
                t.append(t[-1] - dt)
            else:
                break
        
        t = t[::-1]
        x = x[::-1]
        
        # set initial time to zero for reactor
        t_r = [t_ - t[0] for t_ in t]
        
        # first time step
        gas.TPY = T(x[0],t[0]) , P(x[0],t[0]) , Y_init
        
        
        # construct reactor network including heat losses        
        r = self._ct.IdealGasConstPressureReactor(gas)
        
        sim = self._ct.ReactorNet([r])
        states = self._ct.SolutionArray(r.thermo)
        
        
        if len(t) == 1:
            states.append(T=T1, P=P1, X=self.composition)
            stateMatrix, columns = states.collect_data(cols=('T','P','X','D','mean_molecular_weight'))
            
            return stateMatrix, columns, t, u(x[0],t[0])
            
        
        sim.advance(t_r[1])
        states.append(r.thermo.state)
        
        for x_,t_,t_r_ in zip(x[1:-1],t[1:-1],t_r[2:]):
            
            gas.TP = T(x_,t_) , P(x_,t_)
            r.syncState()
            
            sim.advance(t_r_)
            states.append(r.thermo.state)
            
            stateMatrix, columns = states.collect_data(cols=('T','P','X','D','mean_molecular_weight'))
        
        return stateMatrix, columns, self._np.array(t[1:]), self._np.array([u(x_,t_) for x_,t_ in zip(x[1:],t[1:])])
    
    
    def time_signal(self,x,t,dt=1e-6,multiprocessing=True):
        t_in = self._np.arange(dt,t,dt)
        
        if not multiprocessing:
            states = []
            t_out = []
            u_out = []
                
            for t_i in  tqdm(t_in):
                states_, columns, t_, u  = self.point_history(x, t_i, dt)
                states.append(states_[-1])
                t_out.append(t_[-1])
                u_out.append(u[-1])
        else:
            total = len(t_in)
                                  
            def f(t):
                states_, columns, t_, u  = self.point_history(x, t, dt)
                return states_[-1], columns, t_[-1], u[-1]
            
            with Pool(8) as p:
                results = list(tqdm(p.imap(f, t_in), total=total))
                states, columns, t_out, u_out = [list(tup) for tup in zip(*results)]
                p.clear()
            columns = columns[0]
            
        return self._np.array(states), columns, self._np.array(t_out), self._np.array(u_out)
    
    
    def profile(self,t0,dx,L=1.2,dt=1e-6,multiprocessing=True):
                
        x = self._np.arange(0,L,dx)
        
        if not multiprocessing:
            states = []
            
            for x_ in tqdm(x):
                states_, columns, t_, u_  = self.point_history(x_, t0, dt)
                states.append(states_[-1])
                
        else:
            def f(x):
                states_, columns, t_, u_  = self.point_history(x, t0, dt)
                return states_[-1], columns
        
            with Pool(8) as p:
                states, columns = list(tqdm(p.imap(f, x), total=len(x)))
                p.clear()
            columns = columns[0]
        
        return self._np.array(states), columns, x


#%%

if __name__ == '__main__':
    T0 = 295
    p0 = 1e5
    X0 = 'H2:42 ,O2:21, N2:79'
    wave = TaylorWave(T0,p0,X0, 'Klippenstein_noCarbon.cti')
    
    # det = Detonation(T0,p0,X0, 'Klippenstein_noCarbon.cti')
    # flame = det.postFlame(det.postShock_fr(0.8*det.CJspeed))
    # psr = det.postPSR(det.postShock_fr(0.8*det.CJspeed))
    
    # znd_out = wave.znd_out
    # wave = TaylorWave(T0,p0,X0, 'Klippenstein_noCarbon.cti')
    # wave.znd_out = znd_out
    
    
    x0 = 0.7714
    t0 = 2e-3

    
    states, columns, t, u = wave.point_history(x0, t0)
    
    wave.nu = 1.2
    
    states_poly, columns, t_poly, u_poly = wave.point_history(x0, t0)
    
    gas = ct.Solution('Klippenstein_noCarbon.cti')
    ct_states = ct.SolutionArray(gas)
    ct_states_poly = ct.SolutionArray(gas)
    
    for state in states:
        gas.state = state
        ct_states.append(gas.state)
    
    for state_poly in states_poly:
        gas.state = state_poly
        ct_states_poly.append(gas.state)
    
    plt.figure('P')
    plt.plot(t,ct_states.P*1e-5, label='isentropic')
    plt.plot(t_poly,ct_states_poly.P*1e-5, label='polytropic')
    plt.xlabel('t (s)')
    plt.ylabel('P (bar)')
    plt.legend()
    
    plt.figure('NO')
    plt.plot(t,ct_states.X[:,gas.species_index('NO')]*1e6, label='isentropic')
    plt.plot(t_poly,ct_states_poly.X[:,gas.species_index('NO')]*1e6, label='polytropic')
    plt.xlabel('t (s)')
    plt.ylabel('NO (ppm)')
    plt.legend()
    
    plt.figure('u')
    plt.plot(t,u, label='isentropic')
    plt.plot(t_poly,u_poly, label='polytropic')
    plt.xlabel('t (s)')
    plt.ylabel('u (m/s)')
    plt.legend()
    
    plt.figure('T')
    plt.plot(t,ct_states.T, label='isentropic')
    plt.plot(t_poly,ct_states_poly.T, label='polytropic')
    plt.xlabel('t (s)')
    plt.ylabel('T (K)')
    plt.legend()
    

