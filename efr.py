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
import pandas as pd

class Detonation:
    
    class _DetProperty:
        def __init__(self, value=None):
            self.value = value
        def __get__(self, instance, owner=None):
            return self.value
        def __set__(self, instance, value):
            if value != self.value:
                self.value = value
                instance._force_recalc()
    
    @classmethod
    def _set_init_state(cls,values):
        cls.T1,cls.P1,cls.X1,cls.mech,cls.recalc = [cls._DetProperty(value) for value in values]
    
    
    def __init__(self, T1, P1, q, mech='Klippenstein.cti', recalc=True):
        self.t_znd = 1e-3
        # the following are currently necessary to ensure working multiprocessing
        self._ct = ct
        self._sdps = sdps
        self._zndsolve = zndsolve
        self._np = np
        self._soundspeed_eq = soundspeed_eq
        #
        self._set_init_state((T1,P1,q,mech,recalc))
        self._force_recalc()
    
    def _force_recalc(self):
        if not self.recalc:
            if not hasattr(self,'_recalc_warning'):
                print("\nDynamic recalculation is disabled! Detonation properties won't be updated unless called explicitly!\n")
                self._recalc_warning = True
            return
        for prop in ['_cj_speed','_postShock_eq_state','_postShock_fr_state', '_znd_out']:
            if hasattr(self,prop):
                delattr(self,prop)
        print('Calculating detonation properties')
        self.CJspeed
        self.postShock_eq
        self.postShock_fr()
                
    
    @property
    def postShock_eq(self):
        if hasattr(self, '_postShock_eq_state'):
            postShock_eq = self._ct.Solution(self.mech)
            postShock_eq.state = self._postShock_eq_state
            return postShock_eq
        else:
            U1 = self.CJspeed
            T1,P1,X1 = self.T1, self.P1, self.X1
            mech = self.mech
            postShock_eq = self._sdps.PostShock_eq(U1,P1,T1,X1,mech)
            self._postShock_eq_state = postShock_eq.state
            return postShock_eq
    

    def postShock_fr(self, U1=None):
        T1,P1,X1 = self.T1, self.P1, self.X1
        mech = self.mech
        if U1 is None:
            if hasattr(self, '_postShock_fr_state'):
                postShock_fr = self._ct.Solution(mech)
                postShock_fr.state = self._postShock_fr_state
                return postShock_fr
            else:
                U1 = self.CJspeed
                postShock_fr = self._sdps.PostShock_fr(U1, P1, T1, X1, mech)
                self._postShock_fr_state = postShock_fr.state
                return postShock_fr
        else:
            postShock_fr = self._sdps.PostShock_fr(U1, P1, T1, X1, mech)
            return postShock_fr
            
    
    
    @property
    def CJspeed(self):
        if hasattr(self, '_cj_speed'):
            return self._cj_speed
        else:
            self._cj_speed = self._sdps.CJspeed(self.P1, self.T1, self.X1, mech=self.mech)
            return self._cj_speed

    

    def znd(self, **kwargs):
        if hasattr(self, '_znd_out'):
             return self._znd_out
        else:
            U1 = self.CJspeed
            T1,P1,X1 = self.T1, self.P1, self.X1
            mech = self.mech
            gas1 = self._ct.Solution(mech)
            gas1.TPX = T1,P1,X1
            
            gas = self._sdps.PostShock_fr(U1,P1,T1,X1,mech)
            
            print('\nSolving ZND reactor. If this takes long, consider changing relTol and absTol\n')
            znd_out = self._zndsolve(gas,gas1,U1,self.t_znd, **kwargs)
            znd_out['gas1'] = znd_out['gas1'].state
            self._znd_out = znd_out
            return self._znd_out
        
        

class TaylorWave(Detonation):
    
    def __init__(self, T1, P1, q, mech='Klippenstein.cti', u0=0, recalc=True):
        Detonation.__init__(self, T1, P1, q, mech, recalc)
        self.u0 = u0
        self.nu = 1 # polytropic ratio defined as 1 + dq / vdp, with dq defined by heat losses to the environment ("Thermodynamik", Baehr)
        # self.C_f = 0.0062    # heat transfer coefficient directly from DOI:10.2514/1.10286
    
    @property
    def a2_eq(self):
        return self._soundspeed_eq(self.postShock_eq)
    
    @property
    def gamma_eq(self):
        return  self.a2_eq**2*self.postShock_eq.density/self.postShock_eq.P
    
    @property
    def u2(self):
        return self.CJspeed-self.a2_eq
    
    
    
    def point_history(self,x,t,dt=1e-6):
        """
        calculate time history of particle at location x at time t with time 
        resolution dt
        """
        CJspeed = self.CJspeed
        P1 = self.P1
        a2_eq = self.a2_eq
        T1 = self.T1
        u2 = CJspeed - self.znd()['U'][-1]  #CJspeed-a2_eq
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
        Y_init = self.znd()['species'][:,-1]
        
        x = [x]
        t = [t]
        while 1:
            if (t[-1] - dt)*CJspeed > x[-1]:         
                x.append(x[-1] - u(x[-1],t[-1]-dt) * dt)
                t.append(t[-1] - dt)
            else:
                break
        
        t = t[::-1]
        x = x[::-1]
        
        # set initial time to zero for reactor
        t_r = [t_ - t[0] for t_ in t]
        
        # first time step
        gas.TPY = T(x[0],t[0]) , P(x[0],t[0]) , Y_init
        
        
        # construct reactor network
        r = self._ct.IdealGasConstPressureReactor(gas)
        
        sim = self._ct.ReactorNet([r])
        states = self._ct.SolutionArray(r.thermo)
        
        
        if len(t) == 1:
            states.append(T=T1, P=P1, X=self.X1)
            # stateMatrix, columns = states.collect_data(cols=('T','P','X','density','mean_molecular_weight'))
            statesDF = states.to_pandas(cols=('T','P','X','density','mean_molecular_weight'))
            
            statesDF[['t','u']] = t[0],u(x[0],t[0]-dt)
            # return stateMatrix, columns, self._np.array(t), self._np.array([u(x[0],t[0])])
            return statesDF
            
        
        sim.advance(t_r[1])
        states.append(r.thermo.state)
        
        for x_,t_,t_r_ in zip(x[1:-1],t[1:-1],t_r[2:]):
            
            gas.TP = T(x_,t_) , P(x_,t_)
            r.syncState()
            
            sim.advance(t_r_)
            states.append(r.thermo.state)
            
        # stateMatrix, columns = states.collect_data(cols=('T','P','X','density','mean_molecular_weight'))
        statesDF = states.to_pandas(cols=('T','P','X','density','mean_molecular_weight'))
        statesDF['t'] = t[1:]
        statesDF['u'] = [u(x_,t_-dt) for x_,t_ in zip(x[1:],t[1:])]
        
        # return stateMatrix, columns, self._np.array(t[1:]), self._np.array([u(x_,t_) for x_,t_ in zip(x[1:],t[1:])])
        return statesDF
    
    
    def time_signal(self,x,t,dt=1e-6,multiprocessing=True):
        t_in = self._np.arange(dt,t,dt)
        
        if not multiprocessing:
            states = []
                
            for t_i in  tqdm(t_in):
                statesDF  = self.point_history(x, t_i, dt)
                states.append(statesDF.iloc[-1])
        else:
            total = len(t_in)
                                  
            def f(t):
                statesDF = self.point_history(x, t, dt)
                return statesDF.iloc[-1]
            
            with Pool(8) as p:
                states = list(tqdm(p.imap(f, t_in), total=total))
                p.clear()
        
        states_DF = pd.DataFrame(states)
            
        return states_DF
    
    
    def profile(self,t0,dx,L=1.2,dt=1e-6,multiprocessing=True,insertZND=True):
                
        x = self._np.arange(0,L,dx)
        
        if not multiprocessing:
            states = []
            
            for x_ in tqdm(x):
                statesDF = self.point_history(x_, t0, dt)
                states.append(statesDF.iloc[-1])
                
        else:
            def f(x):
                statesDF  = self.point_history(x, t0, dt)
                return statesDF.iloc[-1]
        
            with Pool(8) as p:
                states = list(tqdm(p.imap(f, x), total=len(x)))
                # states =  [list(tup) for tup in zip(*results)]
                p.clear()

        states_DF =  pd.DataFrame(states)       

        # combine DFR and ZND data, if detonation is still inside tube
        if not all(states_DF['T'] > 1.01*self.T1) and insertZND: #TODO! does not work for overdriven
            states = states_DF.to_numpy()
            x_front = x[states[:,0] == states[-1,0]][0]
            x_ZND = x_front-self.znd()['distance'][::-1]
            filter_array = x < x_ZND[0]
            x_comb = self._np.concatenate((x[filter_array],x_ZND,x[~(x < x_ZND[-1])]))
            
            gas = self._ct.Solution(self.mech)
            ZND_states = self._ct.SolutionArray(gas)
            
            u_znd = []
            for T,P,Y,u in zip(self.znd()['T'], self.znd()['P'], self.znd()['species'].transpose(), self.znd()['U1']-self.znd()['U']):
                ZND_states.append(TPY=(T,P,Y))
                u_znd.append(u)
            ZND_states = ZND_states.to_pandas(cols=('T','P','X','density','mean_molecular_weight'))
            ZND_states['t'] = t0
            ZND_states['u'] = u_znd
            
            ZND_states_array = ZND_states.to_numpy()[::-1]
            
            states_comb = list(self._np.concatenate((states[filter_array],ZND_states_array,states[~(x < x_ZND[-1])])))
            
            states_comb = pd.DataFrame(states_comb,columns=ZND_states.columns)
        
        else:
            states_comb = states_DF
            x_comb = x
      
        states_comb['x'] = x_comb
        
        return states_comb


#%%

if __name__ == '__main__': 
       
    
    T0 = 295
    p0 = 1e5
    X0 = 'H2:42 ,O2:21, N2:79'
    # wave = TaylorWave(T0,p0,X0, 'Klippenstein_noCarbon.cti')
    wave = TaylorWave(T0,p0,X0, 'gri30.cti')
    
    # wave.znd(relTol=1e-8,absTol=1e-11)
    
    profile_CJ = wave.profile(1e-4,1e-3,L=0.25,dt=1e-6, multiprocessing=True)
    
    wave._cj_speed = 2400
    del wave._postShock_fr_state
    del wave._postShock_eq_state
    
    # point = wave.point_history(0.01, 1e-4, dt=1e-6)
    profile_overdriven = wave.profile(1e-4,1e-3,L=0.25,dt=1e-6, multiprocessing=True)
    # states = wave.profile(1e-5,1e-3,L=0.1,dt=1e-6, multiprocessing=True)
    
    # det = Detonation(T0,p0,X0, 'Klippenstein_noCarbon.cti')
    # flame = det.postFlame(det.postShock_fr(0.8*det.CJspeed))
    # psr = det.postPSR(det.postShock_fr(0.8*det.CJspeed))
    
    # znd_out = wave.znd_out
    # wave = TaylorWave(T0,p0,X0, 'Klippenstein_noCarbon.cti')
    # wave.znd_out = znd_out
    
    
    # x0 = 0.7714
    # t0 = 2e-3

    
    # states, columns, t, u = wave.point_history(x0, t0)
    
    # wave.nu = 1.2
    
    # states_poly, columns, t_poly, u_poly = wave.point_history(x0, t0)
    
    # gas = ct.Solution('Klippenstein_noCarbon.cti')
    # ct_states = ct.SolutionArray(gas)
    # ct_states_poly = ct.SolutionArray(gas)
    
    # for state in states:
    #     gas.state = state
    #     ct_states.append(gas.state)
    
    # for state_poly in states_poly:
    #     gas.state = state_poly
    #     ct_states_poly.append(gas.state)
    
    # plt.figure('P')
    # plt.plot(t,ct_states.P*1e-5, label='isentropic')
    # # plt.plot(t_poly,ct_states_poly.P*1e-5, label='polytropic')
    # plt.xlabel('t (s)')
    # plt.ylabel('P (bar)')
    # plt.legend()
    
    # plt.figure('NO')
    # plt.plot(t,ct_states.X[:,gas.species_index('NO')]*1e6, label='isentropic')
    # # plt.plot(t_poly,ct_states_poly.X[:,gas.species_index('NO')]*1e6, label='polytropic')
    # plt.xlabel('t (s)')
    # plt.ylabel('NO (ppm)')
    # plt.legend()
    
    # plt.figure('u')
    # plt.plot(t,u, label='isentropic')
    # # plt.plot(t_poly,u_poly, label='polytropic')
    # plt.xlabel('t (s)')
    # plt.ylabel('u (m/s)')
    # plt.legend()
    
    # plt.figure('T')
    # plt.plot(t,ct_states.T, label='isentropic')
    # # plt.plot(t_poly,ct_states_poly.T, label='polytropic')
    # plt.xlabel('t (s)')
    # plt.ylabel('T (K)')
    # plt.legend()
    

