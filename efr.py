#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:21:11 2019

@author: hanraths
"""

try:
    from julia import SDtoolbox as SDT
    julia_import = True
except:
    print('Using Julia solver failed, falling back to scipy.')
    julia_import = False

import sdtoolbox.postshock as sdps
from sdtoolbox.znd import zndsolve
from sdtoolbox.thermo import soundspeed_eq
import numpy as np
import cantera as ct
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
from scipy.integrate import solve_ivp
from MoC import MoC

class Detonation:
        
    def __init__(self, T1, P1, X1, mech='Klippenstein.cti', recalc=True, julia=julia_import):
        super().__setattr__('recalc', recalc)
        self.t_znd = 1e-3
        # the following are currently necessary to ensure working multiprocessing
        self._ct = ct
        if julia:
            self._sdps = SDT
            self._zndsolve = SDT.zndsolve
        else:
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
    
    def _force_recalc(self, extra_attr = [None]):
        if not self.recalc:
            if not hasattr(self,'_recalc_warning'):
                print("\nDynamic recalculation is disabled! Detonation properties won't be updated unless called explicitly!\n")
                self._recalc_warning = True
            return
        for prop in ['_cj_speed','_postShock_eq_state','_postShock_fr_state', '_znd_out', *extra_attr]:
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
        if not hasattr(self, '_cj_speed'):
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


    def _force_recalc(self):
        super()._force_recalc(extra_attr=['_a2_eq', '_T_CJ', '_P_CJ', '_gamma_eq'])
    
    
    def create_MoC(self, *args, **kwargs):
        self.MoC = MoC(self, *args, **kwargs)
        self.MoC.solve_MoC()
    
    
    @property
    def n(self): # polytropic exponent n = gamma/(gamma-nu(gamma-1)) ("Thermodynamik", Baehr)
        return 1 / (1 - self.nu * (1 - 1/self.gamma_eq))


    @property
    def T_CJ(self):
        if not hasattr(self, '_T_CJ'):
            self._T_CJ = self.postShock_eq.T
        return self._T_CJ
    
    
    @property
    def P_CJ(self):
        if not hasattr(self, '_P_CJ'):
            self._P_CJ = self.postShock_eq.P
        return self._P_CJ


    @property
    def a2_eq(self):
        if not hasattr(self, '_a2_eq'):
            self._a2_eq = self._soundspeed_eq(self.postShock_eq)
        return self._a2_eq


    @property
    def gamma_eq(self):
        if not hasattr(self, '_gamma_eq'):
            self._gamma_eq = self.a2_eq**2*self.postShock_eq.density/self.postShock_eq.P
        return self._gamma_eq


    @property
    def u2(self):
        return self.CJspeed-self.a2_eq


    def _phi(self,x,t):
        CJspeed, u2, n = self.CJspeed, self.u2, self.n
        return 2/(n+1)*(x/t/CJspeed-1) + u2/CJspeed


    def _eta(self,x,t):
        CJspeed, a2_eq, n = self.CJspeed, self.a2_eq, self.n
        return (n-1)/(n+1)*(x/t/CJspeed-1) + a2_eq/CJspeed


    def u(self,x,t):
        CJspeed, phi = self.CJspeed, self._phi
        if  x/t > CJspeed:
            return self.u0 
        else:
            return self._np.heaviside(phi(x,t),0)*phi(x,t)*CJspeed + self.u0
    
    
    def c(self,x,t):
        CJspeed, eta, u2, n = self.CJspeed, self._eta, self.u2, self.n
        if x/t <= CJspeed - u2 * (n+1)/2:
            return eta(CJspeed - u2 * (n+1)/2,1) * CJspeed
        else:
            return eta(x,t) * CJspeed
        
    
    def T(self,x,t):
        CJspeed, u2, eta, n, a2_eq = self.CJspeed, self.u2, self._eta, self.n, self.a2_eq
        if x/t > CJspeed:
            return self.T1
        elif x/t <= CJspeed - u2 * (n+1)/2:
            return self.T_CJ * (eta(CJspeed - u2 * (n+1)/2,1)*CJspeed/ a2_eq)**2
        else:
            return self.T_CJ * (eta(x,t)*CJspeed/ a2_eq)**2
        
    
    def P(self,x,t):
        CJspeed, u2, eta, n, a2_eq = self.CJspeed, self.u2, self._eta, self.n, self.a2_eq
        if x/t > CJspeed:
            return self.P1
        elif x/t <= CJspeed - u2 * (n+1)/2:
            return self.P_CJ * (eta(CJspeed - u2 * (n+1)/2,1)*CJspeed/ a2_eq)**(2*n/(n-1))
        else:
            return self.P_CJ * (eta(x,t)*CJspeed/ a2_eq)**(2*n/(n-1))
    
    
    def streamline(self,x0,t0,dt=1e-6,u_func=None,**kwargs):
        
        if not u_func:
            u_func = self.u
        
        func = lambda t, x : [-u_func(x,t0-t)]
        
        result = solve_ivp(func,(0,t0),[x0],t_eval=np.arange(0,t0,dt), **kwargs)
        x = result.y[0]
        t = t0 - result.t
        
        return x,t


    def point_history(self,x0,t0,dt=1e-6):
        """
        calculate time history of particle at location x at time t with time 
        resolution dt
        """
        CJspeed, a2_eq = self.CJspeed, self.a2_eq
        P1, T1 = self.P1, self.T1
        
        gas = self._ct.Solution(self.mech)
        Y_init = self.znd()['species'][:,-1]
        
        if not hasattr(self, 'MoC'):
            # print('Using isentropic Taylor solution')
            u, P, T = self.u, self.P, self.T
            
            x = [x0]
            t = [t0]
            while 1:
                if (t[-1] - dt)*CJspeed > x[-1]:         
                    x.append(x[-1] - u(x[-1],t[-1]-dt) * dt)
                    t.append(t[-1] - dt)
                else:
                    break
            
            T_vec = np.array([T(x,t) for x,t in zip(x,t)])
            P_vec = np.array([P(x,t) for x,t in zip(x,t)])
            t = np.array(t)
            x = np.array(x)
        
        else:
            L = self.MoC.L
            
            T = lambda x,t : self.MoC.T(x/L,t/L*a2_eq)
            P = lambda x,t : self.MoC.P(x/L,t/L*a2_eq)
            u = lambda x,t : self.MoC.U(x/L,t/L*a2_eq) * a2_eq
            
            x,t = self.streamline(x0, t0, u_func=u, dt=dt)
            
            T_vec = T(x,t)
            P_vec = P(x,t)
        
        t = t[::-1]
        x = x[::-1]
        T_vec = T_vec[::-1]
        P_vec = P_vec[::-1]
    
        # set initial time to zero for reactor
        t_r = t - t[0]
        
        # first time step
        gas.TPY = T_vec[0] , P_vec[0] , Y_init
        
        
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
        
        for T_,P_,t_r_ in zip(T_vec[1:-1],P_vec[1:-1],t_r[2:]):
            
            gas.TP = T_ , P_
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

