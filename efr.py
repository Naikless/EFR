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
from collections import namedtuple
from scipy.interpolate import LinearNDInterpolator


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        if  t == 0 or x/t > CJspeed:
            return 0 
        else:
            return self._np.heaviside(phi(x,t),0)*phi(x,t)*CJspeed


    def c(self,x,t):
        CJspeed, eta, u2, n = self.CJspeed, self._eta, self.u2, self.n
        if t == 0 or x/t <= CJspeed - u2 * (n+1)/2:
            return eta(CJspeed - u2 * (n+1)/2,1) * CJspeed
        else:
            return eta(x,t) * CJspeed


    def T(self,x,t):
        CJspeed, u2, eta, n, a2_eq = self.CJspeed, self.u2, self._eta, self.n, self.a2_eq
        if t == 0 or x/t > CJspeed:
            return self.T1
        elif x/t <= CJspeed - u2 * (n+1)/2:
            return self.T_CJ * (eta(CJspeed - u2 * (n+1)/2,1)*CJspeed/ a2_eq)**2
        else:
            return self.T_CJ * (eta(x,t)*CJspeed/ a2_eq)**2


    def P(self,x,t):
        CJspeed, u2, eta, n, a2_eq = self.CJspeed, self.u2, self._eta, self.n, self.a2_eq
        if t == 0 or x/t > CJspeed:
            return self.P1
        elif x/t <= CJspeed - u2 * (n+1)/2:
            return self.P_CJ * (eta(CJspeed - u2 * (n+1)/2,1)*CJspeed/ a2_eq)**(2*n/(n-1))
        else:
            return self.P_CJ * (eta(x,t)*CJspeed/ a2_eq)**(2*n/(n-1))


    def streamline(self,x0,t0,u_func=None,dt=1e-6,**kwargs):

        if not u_func:
            u_func = self.u

        return streamline(x0,t0,u_func,dt=dt,**kwargs)


    def point_history(self,x0,t0,dt=1e-6):

        det_data = Det_data(self)

        return det_data.point_history(x0, t0, dt)


    def time_signal(self,x0,t0,dt=1e-6,multiprocessing=True):
        t = self._np.arange(dt,t0,dt)

        det_data = Det_data(self)
        t = t[~np.isnan(det_data.u(x0,t))]

        if not multiprocessing:
            states = []

            for t_ in  tqdm(t):
                statesDF = det_data.point_history(x0, t_, dt)
                states.append(statesDF.iloc[-1])
        else:                                  
            def f(t):
                statesDF = det_data.point_history(x0, t, dt)
                return statesDF.iloc[-1]

            with Pool(8) as p:
                try:
                    states = list(tqdm(p.imap(f, t), total=len(t)))
                except KeyboardInterrupt:
                    p.terminate()
                finally:
                    p.clear()

        states_DF = pd.DataFrame(states).reset_index(drop=True)
        
        # add ZND profile
        states_DF = add_ZND_to_EFR(self, states_DF)

        return states_DF


    def profile(self,t0,dx,L=1.2,dt=1e-6,multiprocessing=True,insertZND=True):

        x = self._np.arange(0,L,dx)

        det_data = Det_data(self)
        x = x[~np.isnan(det_data.u(x,t0))]

        if not multiprocessing:
            states = []

            for x_ in tqdm(x):
                statesDF = det_data.point_history(x_, t0, dt)
                states.append(statesDF.iloc[-1])

        else:
            def f(x):
                statesDF = det_data.point_history(x, t0, dt)
                return statesDF.iloc[-1]

            with Pool(8) as p:
                try:
                    states = list(tqdm(p.imap(f, x), total=len(x)))
                except KeyboardInterrupt:
                    p.terminate()
                finally:
                    p.clear()

        states_DF =  pd.DataFrame(states).reset_index(drop=True)  
        
        if insertZND:
            states_DF = add_ZND_to_EFR(self, states_DF)

        return states_DF



MoC_Container = namedtuple('MoC_Container', (field for field in 'TPu'))


class Det_data:
    def __init__(self, det):
        self.P1, self.T1, self.X1 = det.P1, det.T1, det.X1
        self.mech = det.mech
        trimmed_znd_states = trim_znd(det._znd_out, det.mech)

        self.Y_init = trimmed_znd_states.Y[-1]

        if hasattr(det, 'MoC'):
            L = det.MoC.L
            a2_eq = det.a2_eq
            x = det.MoC.x
            t = det.MoC.t

            T = LinearNDInterpolator(list(zip(x, t)), det.MoC.T)
            P = LinearNDInterpolator(list(zip(x, t)), det.MoC.P)
            U = LinearNDInterpolator(list(zip(x, t)), det.MoC.U)
            
            self.T = lambda x,t : T(x/L, t*a2_eq/L)
            self.P = lambda x,t : P(x/L, t*a2_eq/L)
            self.u = lambda x,t : U(x/L, t*a2_eq/L)*a2_eq

            # Calling the interpolators once is required to create their cache
            # Otherwise, each new parallel process starts from scratch again.
            self.T(0,0)
            self.P(0,0)
            self.u(0,0)

        else:
            u_, P_, T_ = det.u, det.P, det.T
            self.u = lambda x,t : np.squeeze([u_(x_,t_) for x_,t_ in zip(np.asarray(x).flatten(),np.asarray(t).flatten())])
            self.P = lambda x,t : np.squeeze([P_(x_,t_) for x_,t_ in zip(np.asarray(x).flatten(),np.asarray(t).flatten())])
            self.T = lambda x,t : np.squeeze([T_(x_,t_) for x_,t_ in zip(np.asarray(x).flatten(),np.asarray(t).flatten())])

    def point_history(self,x0,t0,dt):
        """
        calculate time history of particle at location x at time t with time 
        resolution dt
        """
        ct.suppress_thermo_warnings()
        gas = ct.Solution(self.mech)
        Y_init = self.Y_init

        u, P, T = self.u, self.P, self.T

        x,t = streamline(x0, t0, u, dt=dt)

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
        r = ct.IdealGasConstPressureReactor(gas)

        sim = ct.ReactorNet([r])
        states = ct.SolutionArray(r.thermo)

        if len(t) == 1:
            states.append(T=T_vec[0], P=P_vec[0], Y=Y_init)
            # stateMatrix, columns = states.collect_data(cols=('T','P','X','density','mean_molecular_weight'))
            statesDF = states.to_pandas(cols=('T','P','X','density','mean_molecular_weight'))

            statesDF[['x','t','u']] = x[0],t[0],float(u(x[0],t[0]))
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
        statesDF['u'] = [float(u(x_,t_)) for x_,t_ in zip(x[1:],t[1:])]

        # return stateMatrix, columns, self._np.array(t[1:]), self._np.array([u(x_,t_) for x_,t_ in zip(x[1:],t[1:])])
        return statesDF



def streamline(x0,t0,u_func,dt=1e-6,**kwargs):

    func = lambda t, x : [-u_func(x,t0-t)]

    result = solve_ivp(func,(0,t0),[x0],t_eval=np.arange(0,t0,dt), **kwargs)
    x = result.y[0]
    t = t0 - result.t

    return x,t

def trim_znd(znd_output, mech):
    states = ct.SolutionArray(ct.Solution(mech), extra=('t','distance','velocity'))
    for T, P, Y, t, x, v in zip(znd_output['T'],znd_output['P'],
                                znd_output['species'].T, znd_output['time'], 
                                znd_output['distance'], znd_output['U']):
        states.append(T=T, P=P, Y=Y, t=t, distance=x, velocity=v)
    
    Ma_eq = znd_output['U'] / soundspeed_eq(states)
    CJ_idx = np.argmin(np.abs(Ma_eq-1))
    
    states = states[:CJ_idx]
    
    return states

def initial_state(gas):
    states = ct.SolutionArray(gas, extra=('x','t','velocity'))
    for i in range(2):
        states.append(T=gas.T, P=gas.P, X=gas.X, x=0, t=0, velocity=0)
    
    return states

def add_ZND_to_EFR(det, efr_data):

    trimmed_znd_states = trim_znd(det._znd_out, det.mech)
    
    t_fix = trimmed_znd_states.distance/det.CJspeed + efr_data.t[0]
    trimmed_znd_states.t = t_fix
    
    # cut off ZND profile where Ma_eq ~ 1
    znd_DF = trimmed_znd_states.to_pandas(cols=('T','P','X','density',
                                                'mean_molecular_weight',
                                                't','distance','velocity'))
    znd_DF.rename(columns={'velocity':'u'},inplace=True)
    znd_DF.rename(columns={'distance':'x'},inplace=True)
    znd_DF['u'] = det.CJspeed - znd_DF['u']
    # znd_DF['x'] = efr_data.x
    
    # add initial states
    gas = ct.Solution(det.mech)
    gas.TPX = det.T1, det.P1, det.X1
    states_init = initial_state(gas)
    dt_ZND = znd_DF.t[1]- znd_DF.t[0]
    dx_ZND = znd_DF.x[1]- znd_DF.x[0]
    
    # decide whether EFR data is time signal or profile
    if len(efr_data.x.unique()) == 1:
        timeSignal = True
        states_init.t = [0, efr_data.t[0] - dt_ZND ]
        states_init.x = efr_data.x[0]
        # shift time signal by time intervall of ZND profile
        efr_data['t'] = efr_data.t + (znd_DF.t.iloc[-1] - znd_DF.t.iloc[0]) + dt_ZND
    else:
        # profile
        timeSignal = False
        znd_DF = znd_DF.reindex(index=znd_DF.index[::-1])
        znd_DF.reset_index(inplace=True, drop=True)
        # move ZND profile ahead of EFR data
        x_ZND = znd_DF.x.to_numpy()
        x_ZND = x_ZND[0] - x_ZND
        znd_DF.x = x_ZND + efr_data.x.iloc[-1] + dx_ZND
        # add initial values at the right end
        dx_EFR = efr_data.x[1] - efr_data.x[0]
        states_init.x = znd_DF.x.iloc[-1] + np.array([dx_ZND, dx_EFR])
        states_init.t = efr_data.t[0]
    
    states_init_DF = states_init.to_pandas(cols=('T','P','X','density',
                                                'mean_molecular_weight',
                                                'x','t','velocity'))
    
    states_init_DF.rename(columns={'velocity':'u'},inplace=True)
    
    if timeSignal:
        efr_data = pd.concat((states_init_DF, znd_DF, efr_data), ignore_index=True)
    else:
        efr_data = pd.concat((efr_data, znd_DF, states_init_DF), ignore_index=True)

    return efr_data