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
        self.D = 3e-2
        self.unburnt_ratio = 0
        self.t_znd = 1e-3
        # the following are currently necessary to ensure working multiprocessing
        self.ct = ct
        self.sdps = sdps
        self.zndsolve = zndsolve
        self.np = np
    
    @property
    def postShock_eq(self):
        if hasattr(self, 'postShock_eq_state'):
            postShock_eq = self.ct.Solution(self.mech)
            postShock_eq.state = self.postShock_eq_state
            return postShock_eq
        else:
            U1 = self.CJspeed
            T1,P1,X1 = self.T1, self.P1, self.composition
            mech = self.mech
            postShock_eq = self.sdps.PostShock_eq(U1,P1,T1,X1,mech)
            self.postShock_eq_state = postShock_eq.state
            return postShock_eq
    

    def postShock_fr(self, U1=None):
        T1,P1,X1 = self.T1, self.P1, self.composition
        mech = self.mech
        if U1 is None:
            if hasattr(self, 'postShock_fr_state'):
                postShock_fr = self.ct.Solution(mech)
                postShock_fr.state = self.postShock_fr_state
                return postShock_fr
            else:
                U1 = self.CJspeed
                postShock_fr = self.sdps.PostShock_fr(U1, P1, T1, X1, mech)
                self.postShock_fr_state = postShock_fr.state
                return postShock_fr
        else:
            postShock_fr = self.sdps.PostShock_fr(U1, P1, T1, X1, mech)
            return postShock_fr
            
    
    
    @property
    def CJspeed(self):
        if hasattr(self, '_cj_speed'):
            return self._cj_speed
        else:
            self._cj_speed = self.sdps.CJspeed(self.P1, self.T1, self.composition, mech=self.mech)
            return self._cj_speed

    
    @property
    def znd(self):
        if hasattr(self, '_znd_out'):
             return self._znd_out
        else:
            U1 = self.CJspeed
            T1,P1,X1 = self.T1, self.P1, self.composition
            mech = self.mech
            gas1 = self.ct.Solution(mech)
            gas1.TPX = T1,P1,X1
            
            gas = self.sdps.PostShock_fr(U1,P1,T1,X1,mech)
            
            print('\nSolving ZND reactor. This might take a while...\n')
            znd_out = self.zndsolve(gas,gas1,U1,self.t_znd)
            znd_out['gas1'] = znd_out['gas1'].state
            self._znd_out = znd_out
            return self._znd_out
        
    def unburntPockets(self, ratio = None, weakShockVel=None, dt=1e-8, reactor = '1D', **kwargs):
        if ratio is None:
            ratio = self.unburnt_ratio
        
        if weakShockVel is None:
            weakShockVel  = 0.8 * self.CJspeed
        
        # create mixture state of unburnt pockets with ZND products
        gas_znd = self.ct.Solution(self.mech)
        gas_znd.TPY = self.znd['T'][-1], self.znd['P'][-1], self.znd['species'][:,-1]
        
        if reactor == '1D':
            gas_flame = self.postFlame(self.postShock_fr(weakShockVel), **kwargs)
        elif reactor == 'PSR':
            gas_flame = self.postPSR(self.postShock_fr(weakShockVel), dt=dt, **kwargs)
        
        # include unreacted pockets by mixing cantera quantity objects
        znd = ct.Quantity(gas_znd, mass=1, constant='HP')
        unburnt = ct.Quantity(gas_flame, mass=ratio, constant='HP')
        
        mix = znd + unburnt
        
        mix_state = mix.phase
        
        return mix_state
    
    def postFlame(self,state, threshold=0.999, model='Mix'):
        
        if hasattr(self, 'post_flame_state'):
            gas = self.ct.Solution(self.mech)
            gas.state = self.post_flame_state
            return gas
        else:
            from scipy.integrate import cumtrapz, trapz 
            
            width = 0.001
            
            # Set up flame object
            f = self.ct.FreeFlame(state, width=width)
            f.set_refine_criteria(ratio=3, slope=0.01, curve=0.5)
            
            # Solve with chosen transport model
            f.transport_model = model
            print("\nSolving 1D free flame with transport model '" + model + "'\n")
            f.solve(loglevel=0, auto=True)
            
            # find point of threshold heat release
            rel_heat_release = cumtrapz(f.heat_release_rate, f.grid)/trapz(f.heat_release_rate, f.grid)
            idx = self.np.argmin(self.np.abs(rel_heat_release-threshold))
            
            
            gas = self.ct.Solution(self.mech)
            gas.TPX = f.T[idx], f.P, f.X[:,idx]
            self.post_flame_state = gas.state
            
            return gas
    
    def postPSR(self,state, threshold=0.999, constant='P', dt=1e-8, t_end = 1e-3):
        
        if hasattr(self, 'post_psr_state'):
            gas = self.ct.Solution(self.mech)
            gas.state = self.post_psr_state
            return gas
        else:
            from scipy.integrate import cumtrapz, trapz
            
            # Set up reactor
            if constant == 'P':
                reactor = self.ct.ConstPressureReactor(state)
            elif constant == 'V':
                reactor = self.ct.Reactor(state)
            
            # set up reactor net
            net = self.ct.ReactorNet([reactor])
            time = 0.
            states = self.ct.SolutionArray(self.ct.Solution(self.mech))
            
            print("\nSolving PSR with constant " + constant + "\n")
            while time < t_end:
                time += dt
                net.advance(time)
                states.append(reactor.thermo.state)
            
            # find point of threshold heat release
            heat_release_rate = -np.array([np.dot(state.net_rates_of_progress, state.delta_enthalpy) for state in states])
            rel_heat_release = cumtrapz(heat_release_rate)/trapz(heat_release_rate)
            idx = self.np.argmin(self.np.abs(rel_heat_release-threshold))
            
            
            gas = self.ct.Solution(self.mech)
            gas.TPX = states.T[idx], states.P[idx], states.X[idx]
            self.post_flame_state = gas.state
            
            return gas
        
        

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
                return self.np.heaviside(phi(x,t),0)*phi(x,t)*CJspeed + self.u0
        
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
        
        
        gas = self.ct.Solution(self.mech)
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
        # r = self.ct.IdealGasReactor(gas)
        r = self.ct.IdealGasConstPressureReactor(gas)
        
        # assuming reactor wall at atmospheric conditions
        # gas_out = self.ct.Solution(self.mech)
        # gas_out.TPX = 293, self.ct.one_atm, 'O2:21,N2:79'
        # outside = self.ct.Reservoir(gas_out)
        
        # wall = self.ct.Wall(r,outside)
        
        sim = self.ct.ReactorNet([r])
        states = self.ct.SolutionArray(r.thermo)
        
        
        if len(t) == 1:
            states.append(T=T1, P=P1, X=self.composition)
            stateMatrix, columns = states.collect_data(cols=('T','P','X','D','mean_molecular_weight'))
            
            return stateMatrix, columns, t, u(x[0],t[0])
            
        
        sim.advance(t_r[1])
        states.append(r.thermo.state)
        
        for x_,t_,t_r_ in zip(x[1:-1],t[1:-1],t_r[2:]):
            
            gas.TP = T(x_,t_) , P(x_,t_)
            r.syncState()
            
            # # adjust heat transfer coefficient according to Radulescu (DOI:10.2514/1.10286) based on Reynolds analogy
            # wall.area = self.np.pi * self.D * u(x_,t_) * dt
            # T_stag = (1+ (gamma_eq-1)/2 * (u(x_,t_)/soundspeed_eq(gas))**2) * gas.T
            # transfer_coefficient = - self.C_f / self.D * gas.density * gas.cp_mass * np.abs(u(x_,t_))
            # wall.set_heat_flux(transfer_coefficient * (T_stag - 293))
            
            sim.advance(t_r_)
            states.append(r.thermo.state)
            
            stateMatrix, columns = states.collect_data(cols=('T','P','X','D','mean_molecular_weight'))
        
        return stateMatrix, columns, self.np.array(t[1:]), self.np.array([u(x_,t_) for x_,t_ in zip(x[1:],t[1:])])
    
    def time_signal(self,x,t,dt=1e-6,multiprocessing=True):
        t_in = self.np.arange(dt,t,dt)
        
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
            
        return self.np.array(states), columns, self.np.array(t_out), self.np.array(u_out)
    
    def profile(self,t0,dx,L=1.2,dt=1e-6,multiprocessing=True):
                
        x = self.np.arange(0,L,dx)
        
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
        
        return self.np.array(states), columns, x


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
    

