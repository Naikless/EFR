# -*- coding: utf-8 -*-
"""
Created on Thu Sep 31 23:03:28 2023

MoC for one-dimensional, unsteady and non-isentropic detonation wave. 
Algorithm based on:
    J.H. Skinner - Plane-flame simulation of the wake behind an internally 
    propelled vehicle Part 1, Simulation of a supersonic vehicle by a 
    detonation
    
    M.I. Radulescu & R.K. Hanson - Effect of Heat Loss on 
    Pulse-Detonation-Engine Flow Fields and Performance - 10.2514/1.10286
    
    K. Kawane et al. - The influence of heat transfer and friction on the 
    impulse of a detonation tube - 10.1016/j.combustflame.2011.02.017

@author: Niclas
"""

import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
from collections import namedtuple
import sdtoolbox as sdt
from scipy.interpolate import LinearNDInterpolator
import efr

# values are normalized:
#   t = t' * c_CJ / L  # = tau in the above papers, t' is non-normalized time
#   x = x' / L         # = xi in the above papers, x' is non-normalized space
#   U = u / c_CJ
#   C = c / c_CJ
#   S = s / s_CJ


# Points are named tuples with x, t, U, C, S
Point = namedtuple('Point', (field for field in 'xtUCS'),defaults=(0,0,0))



def plot_point(p,radius=5e-4,invert = False,**kwargs):
    """
    Visualize point as scatter with lines indicating the characteristics.

    Parameters
    ----------
    p : Point
        Point to plot.
    radius : Float, optional
        Length of the characteristics around the point. The default is 5e-4.
    invert : Bool, optional
        Invert direction of characteristics. The default is False.
    **kwargs : Dict
        Keyword arguments for matplotlibs scatter function.

    Returns
    -------
    None.

    """
    sign = -1 if invert else 1
    
    plt.scatter(p.x,p.t,**kwargs)
    t_Cplus = np.linspace(0, radius / ((1+(p.U+p.C)**2))**0.5, 10)
    plt.plot(p.x + (p.U+p.C) * t_Cplus * sign, p.t + t_Cplus  *sign, 'k--')
    
    t_Cminus = np.linspace(0, radius / ((1+(p.U-p.C)**2))**0.5, 10)
    plt.plot(p.x + (p.U-p.C) * t_Cminus *sign, p.t + t_Cminus *sign, 'k-.')
    
    t_U = np.linspace(0, radius / ((1+(p.U)**2))**0.5, 10)
    plt.plot(p.x + p.U * t_U*sign, p.t + t_U*sign, 'k:')



class PointList(list):
    """
    List of points that allows for access to each of the fields as a list.
    """
    def __init__(self,*args,**kwargs):
        list.__init__(self,*args, **kwargs)


    def __getattr__(self, name):
        return [getattr(p,name) for p in self]


    def flatten(self):
        """
        Turn the Pointlist one-dimensional by concatenation.

        Returns
        -------
        PointList
            Flatten PointList.

        """
        flattened_list = []
        for p in self:
            if isinstance(p, PointList):
                flattened_list.extend(p.flatten())
            else:
                flattened_list.append(p)
        return PointList(flattened_list)


    def find_point(self,x,t):
        """
        Identify the point in that is closest to (x,t).

        Parameters
        ----------
        x : Float
            Normalized axial position.
        t : Float
            Normalized time.

        Returns
        -------
        Point
            Identified Point object.
        idx : Int
            Flattened index of the point.

        """
        points = self.flatten()
        idx = np.argmin(((np.array(points.x) - x)**2 + (np.array(points.t)-t)**2)**0.5)
        return points[idx], idx



class MoC:
    """
    Object to contain the calculated fields from the MoC.
    """
    def __init__(self, det : 'efr.TaylorWave', beta, ds, N, cf=0, T_w=None, P_out=None):
        self.p_isentropic = det.P
        self.u_isentropic = det.u
        self.c_isentropic = det.c
        
        self.det = det
        self.beta = beta
        self.cf = cf
        self.ds = ds
        self.N = N
        
        T_w = T_w if T_w else det.T1
        P_out = P_out if P_out else det.P1
        
        self.P_CJ = det.postShock_eq.P
        self.s_CJ = det.postShock_eq.entropy_mass
        self.R = ct.gas_constant/det.postShock_eq.mean_molecular_weight
        
        self.P_crit = P_out/(2/(det.gamma_eq+1))**(det.gamma_eq/(det.gamma_eq-1))
        
        gas = det.postShock_eq
        gas.TP = T_w,gas.P
        gamma = sdt.thermo.soundspeed_eq(gas)**2/gas.P*gas.density
        self.C0 = (gamma * ct.gas_constant /gas.mean_molecular_weight * T_w)**0.5 / det.a2_eq # normalized speed of sound at wall temperature
        
        self.U_CJ = det.u2/det.a2_eq
        self.C_CJ = 1
        self.S_CJ = self.s_CJ/det.gamma_eq/self.R



    def solve_MoC(self, ds=None, N=None):
        """
        Calculate field data using MoC.

        Parameters
        ----------
        ds : Float
            Initial resolution of isentropic solution.
        N : Int
            Number of C- characteristics alongs the x axis.

        Returns
        -------
        points : PointList
            Points of the entire flow field.

        """
        ds = ds if ds else self.ds
        N = N if N else self.N
        gamma_eq = self.det.gamma_eq
        
        print('\nCalulating MoC\n')
        
        print('\nInitializing grid.\n')
        points_init = self.initial_grid(ds,1e-3)
        self.initial_points = points_init
        points = PointList(points_init)
        
        print('\nSolving detonation field.\n')
        points.extend(self._det_points(points_init,N))
        
        print('\nInterpolate between last detonation C- and first expansion wave C-.\n')
        points.extend(self._interpolate_points(points[-1]))

        print('\nCreate new C- characteristics along the sonic boundary.\n')
        skip = 20
        points.extend(self._new_Cminus_exit(points[-1], skip))
        
        
        flat_point_list = points.flatten()
        x = np.array(flat_point_list.x)
        t = np.array(flat_point_list.t)
                
        self.U = LinearNDInterpolator(list(zip(x, t)), flat_point_list.U)
        self.C = LinearNDInterpolator(list(zip(x, t)), flat_point_list.C)
        self.S = LinearNDInterpolator(list(zip(x, t)), flat_point_list.S)
        
        self.P = lambda x,t : self.P_CJ * self.C(x,t)**(2*gamma_eq/(gamma_eq-1)) * np.exp(gamma_eq*(self.S_CJ - self.S(x,t)))
        self.T = lambda x,t : (self.C(x,t)*self.det.a2_eq)**2 / self.det.gamma_eq / self.R
        
        self.points = points
        
        return points

    
    def _det_points(self, C_minus_start, N):
        """
        Solve characteristics along the detonation wave.

        Parameters
        ----------
        C_minus_start : PointList
            Initial C- characteristic from isentropic Taylor solution.
        N : Int
            Number of C- characteristics distributed along the x axis.

        Returns
        -------
        points : PointList
            Newly calculated points.

        """
        points = PointList([])
        pl = C_minus_start
        dx = (1-pl[0].x)/N
        dt = dx / self.det.CJspeed * self.det.a2_eq
        
        U_CJ, C_CJ, S_CJ = self.U_CJ, self.C_CJ, self.S_CJ
        
        for i in range(N):
            p_start = Point(pl[0].x + dx, pl[0].t + dt, U_CJ, C_CJ, S_CJ)
            pl = self._new_Cminus(p_start,pl[1:])
            p_zero = self.add_zero_point(pl[-1])
            pl.append(p_zero)
            points.append(pl) 
        
        self.det_points = points
        return points
    
    
    def _interpolate_points(self,last_C_minus):
        """
        Interpolate additional characteristics between last detonation C- and 
        first expansion C-.

        Parameters
        ----------
        last_C_minus : PointList
            Last C- characteristic when the detonation exits teh chamber.

        Returns
        -------
        points : PointList
            Newly calculated points.

        """
        p_new = self.add_end_point(last_C_minus[1], last_C_minus[0])[0]
        pl = last_C_minus
        points = PointList([])
        
        for step in np.arange(0.1,1.1,0.1):
            p_start = Point(*(np.array(last_C_minus[0])*(1-step) + np.array(p_new)*step))
            pl = [p for p in pl if p.t >= p_start.t]
            pl = self._new_Cminus(p_start,pl)
            points.append(pl)
        
        self.interp_points = points
        return points
    
    
    def _new_Cminus(self, p_start, last_C_minus,**kwargs):
        """
        Calculate new C- characteristic from a starting point along the
        detonation, following the previous C- characteristic.

        Parameters
        ----------
        p_start : Point
            Start of the C- characteristic at the CJ front.
        last_C_minus : PointList
            Previous C- characteristic.
        **kwargs : KW
            Keywords for the tolerance of the iterative point solver.

        Returns
        -------
        plist_new : PointList
            New C- characteristic.

        """
        p_new = p_start
        
        plist_new = PointList([p_new])
        
        for p_old in last_C_minus:
            p_new,p4 = self.new_point(p_old, p_new,**kwargs)
            plist_new.append(p_new)
        
        p0,p1,p2 = plist_new[:3]
        dist_p0p1 = ((p1.x-p0.x)**2 + (p1.t-p0.t)**2)**0.5
        dist_p1p2 = ((p2.x-p1.x)**2 + (p2.t-p1.t)**2)**0.5
        
        if dist_p0p1 > dist_p1p2:
            p_add = Point(*(np.array(p1) + np.array(p0))/2)
            plist_new = PointList([p0] + [p_add] + plist_new[1:])
               
        return plist_new

    
    def _new_Cminus_exit(self, last_C_minus, skip):
        """
        Calculate new C- characteristic from a starting point along the
        domain exit, following the previous C- characteristic.

        Parameters
        ----------
        last_C_minus : PointList
            Previous C- characteristic.
        skip : Int
            Defines how many points of the last C- are skipped  before solving
            for new points.
        **kwargs : KW
            Keywords for the tolerance of the iterative point solver.

        Returns
        -------
        plist_new : PointList
            New C- characteristic.

        """
        
        p_new = self.add_end_point(last_C_minus[skip+1], last_C_minus[skip])[0]
        pl_new = last_C_minus
        p_start = p_new
        
        points = PointList([])
        while 1:
            pl_new = self._new_Cminus(p_start,pl_new[skip+1:])
            if len(pl_new) <= skip+1 or any(np.abs(np.array(pl_new.x)[-10:-1]-1) < 1e-8):
                break
            points.append(pl_new)
            p_start = self.add_end_point(pl_new[skip+1], pl_new[skip])[0]
            if not p_start:
                break
        
        self.exit_points = points
        return points


    def new_point(self, p1, p2, tol=1e-3):
        """
        Calculate new internal point. Based on the algorithm by Skinner.

        Parameters
        ----------
        p1 : Point
            Left point.
        p2 : Point
            Right Point.
        tol : Float, optional
            Convergence tolerance. The default is 1e-5.

        Returns
        -------
        tuple(P3 : Point, P4 : Point)
            New internal point P3 and interpolated U-characteristic point P4.

        """
        x1,t1,U1,C1,S1 = p1
        x2,t2,U2,C2,S2 = p2
        gamma_eq = self.det.gamma_eq
        cf = self.cf
        beta = self.beta
        C0 = self.C0
        
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
                U4 = (x4-x1)/(x2-x1) * U2 +  (x2-x4)/(x2-x1) * U1
                C4 = (x4-x1)/(x2-x1) * C2 +  (x2-x4)/(x2-x1) * C1
                S4 = (x4-x1)/(x2-x1) * S2 +  (x2-x4)/(x2-x1) * S1
                
                U34 = (U3+U4)/2
                C34 = (C3+C4)/2
                
                t4_ = (((t2-t1)/(x2-x1) * (x3 - U34*t3 - x1) + t1) /
                      (1 - (t2-t1)/(x2-x1) * U34))
                x4_ = x3 - U34 * (t3-t4_)
                
                
                if abs(x4_ - x4)/x4 < tol and abs(t4_ - t4)/t4 < tol:
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
            
            if abs(x3_ - x3)/x3 < tol and abs(t3_ - t3)/t3 < tol:
                break
            else:
                if n > 100:
                    raise Exception("Couldn't converge to new grid point, maybe reduce tmax.")
                x3 = x3_; t3 = t3_
                n += 1
        
        return Point(x3,t3,U3,C3,S3), Point(x4,t4,U4,C4,S4)
    
    
    def initial_grid(self,ds,t0):
        """
        Defines the initial C- characteristic based on the isentropic Taylor 
        solution

        Parameters
        ----------
        ds : Float
            Spatial resolution.
        t0 : Float
            Initial time. Should be early enough to justify an isentropic flow
            field.

        Returns
        -------
        pl : PointList
            Initial isentropic C- characteristic.

        """
        
        gamma_eq = self.det.gamma_eq
        P_CJ = self.det.P_CJ
        a2_eq = self.det.a2_eq
        
        p0 = Point(self.det.CJspeed/a2_eq * t0, t0, self.U_CJ, self.C_CJ, self.S_CJ)
        pl = PointList([p0])
        p_new = p0
        
        p_isentropic, u_isentropic, c_isentropic = self.det.P, self.det.u, self.det.c
        
        s_CJ, R = self.s_CJ, self.R
        
        while 1:
            p_old = pl[-1]
            dt = (ds**2 / (1+(p_old.U-p_old.C)**2))**0.5
            dx = (p_old.U-p_old.C) * dt
            
            x_new = p_old.x+dx
            t_new = p_old.t+dt
            
            p_ = p_isentropic(x_new,t_new/a2_eq)
            u_ = u_isentropic(x_new,t_new/a2_eq)
            c_ = c_isentropic(x_new,t_new/a2_eq)
            s_ = s_CJ - R * (np.log(p_/P_CJ) + 2*gamma_eq/(gamma_eq-1)*np.log(a2_eq/c_))
            
            p_new = Point(x_new, t_new, u_/a2_eq, c_/a2_eq, s_/gamma_eq/R )
            
            if p_new.U <= 0:
                p = pl[-1]
                p_new = Point(0,-p.x/(p.U-p.C)+p.t,*p_new[2:])
                pl.append(p_new)
                break
            
            pl.append(p_new)
        
        return pl
    
    
    def add_zero_point(self, p, tol=1e-3):
        """
        Calculate point at the left boundary based on internal point p.

        Parameters
        ----------
        p : Point
            Internal point to use for boundary calculation.
        tol : Float, optional
            Iteration convergence tolerance. The default is 1e-3.

        Returns
        -------
        Point
            New point at left boundary.

        """
        x2,t2,U2,C2,S2 = p
        gamma_eq = self.det.gamma_eq
        cf = self.cf
        beta = self.beta
        C0 = self.C0
        S_CJ = self.S_CJ
        
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
            
            if abs(t3_ - t3) / t3 < tol:
                break
            else:
                t3 = t3_
                
        return Point(x3,t3,U3,C3,S3)
    
    
    def add_end_point(self, p1, p2, tol=1e-3):
        """
        Calculate point at the right boundary based on internal points p1 and 
        p2. p2 is primarily used to interpolate the entropy, position is 
        calculated based on p1. This assumes a choked exit and warns, if this 
        is not fullfilled.

        Parameters
        ----------
        p1 : Point
            Left point.
        p2 : Point
            Right point (usually already at the domain edge).
        tol : Float, optional
            Iteration convergence tolerance. The default is 1e-3.

        Raises
        ------
        Exception
            Exception if iterative solver could not converge.

        Returns
        -------
        tuple(P3 : Point, P4 : Point)
            New internal point P3 and interpolated U-characteristic point P4.

        """
        x1,t1,U1,C1,S1 = p1
        x2,t2,U2,C2,S2 = p2
        
        gamma_eq =  self.det.gamma_eq
        S_CJ = self.S_CJ
        beta = self.beta
        cf = self.cf
        C0 = self.C0
        P_CJ = self.P_CJ

                  
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
            
            C13 = (C1+C3)/2
                    
            S3 = (S4 - beta * 2 *cf *abs(U34) /(gamma_eq-1)/C34**2
                  * (C34**2 - (gamma_eq-1)/2 * U34**2 - C0**2)
                  * (t3-t4)
                  )
             
            R_plus3 = (R_plus1 - beta * 2 *cf *abs(U13) /C13
                       * (C13**2 - (gamma_eq-1)/2 * U13**2 - C0**2 + U13*C13)
                       * (t3-t1)
                       + C13 * (S3-S1)
                       )
            
            U3 = C3 = R_plus3/(2/(gamma_eq-1) + 1)
            
            P = P_CJ * C3**(2*gamma_eq/(gamma_eq-1)) * np.exp(gamma_eq*(S_CJ - S3))
            if P < self.P_crit:
                print('not choked!')
                return None,None
                         
            U13 = (U1+U3)/2            
            C13 = (C1+C3)/2
            
            t3_ = (x3 - x1) / (U13 + C13) + t1           
            
            if abs(t3_ - t3) / t3 < tol:
                break
            else:
                if n > 100:
                    raise Exception("Couldn't converge to new grid point, maybe reduce tmax.")
                t3 = t3_
                n += 1
            
        
        return Point(x3,t3,U3,C3,S3), Point(x4,t4,U4,C4,S4)

