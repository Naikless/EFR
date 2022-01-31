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
import cantera as ct
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd

class Detonation:
    
    def __init__(self, T1, P1, X1, mech='Klippenstein.cti', recalc=True, julia=julia_import):
        super().__setattr__('recalc', recalc)
        self.t_znd = 1e-3
        # the following are currently necessary to ensure working multiprocessing
        self._ct = ct
        self._sdps = sdps
        self._zndsolve = zndsolve
        self._np = np
        self._soundspeed_eq = soundspeed_eq
        #
        self.T1, self.P1, self.X1, self.mech = T1, P1, X1, mech
        self._force_recalc()
    
    def __setattr__(self, name, value):
        old = getattr(self, name, value)
        super().__setattr__(name, value)
        if name in {'T1','P1','X1','mech','recalc'} and old != value:
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
            self._cj_speed = self._sdps.CJspeed(self.P1, self.T1, self.X1, self.mech)
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
            znd_out = self._zndsolve(gas,gas1,U1,t_end=self.t_znd, **kwargs)
            znd_out['gas1'] = znd_out['gas1'].state
            self._znd_out = znd_out
            return self._znd_out
        
        

class TaylorWave(Detonation):
    
    def __init__(self, *args, u0=0, **kwargs):
        super().__init__(*args, **kwargs)
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
            
            statesDF[['x','t','u']] = x[0],t[0],u(x[0],t[0]-dt)
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
        statesDF['x'] = x[1:]
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
            states = states_DF.drop(columns='x').to_numpy()
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
            states_comb['x'] = x_comb
        
        else:
            states_comb = states_DF
      
        return states_comb

