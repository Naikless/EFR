# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 23:56:58 2022

@author: Niclas
"""

import efr
import pytest
import pandas as pd


@pytest.mark.parametrize('cls_type',[efr.Detonation, efr.TaylorWave])
def test_det_init(det,cls_type,initState):
    print("Initialize Detonation object")
    obj = det(cls_type=cls_type)
    assert isinstance(det(),efr.Detonation)
    print("Assert that initial state is correct")
    assert initState == [obj.T1,obj.P1,obj.X1,obj.mech]
    
def test_CJ_calc(det,expected):
    print("Assert correct CJspeed calculation")
    assert det().CJspeed == expected.CJspeed
    
def test_postShock_fr(det,expected):
    obj = det()
    print("Assert correct frozen post shock calculation")
    assert all((obj.postShock_fr().state - expected.postShock_fr.state) < 1e-10)

def test_postShock_eq(det,expected):
    obj = det()
    print("Assert correct equilibrium post shock calculation")
    assert all((obj.postShock_eq.state - expected.postShock_eq.state) < 1e-10)
    
def test_znd(det,expected):
    obj = det()
    print("Assert correct znd calculation")
    assert all((x==y for x,y in zip(obj.znd(),expected.znd)))

def test_no_recalc(det,capsys):
    no_recalc = det()
    assert capsys.readouterr().out == "\nDynamic recalculation is disabled! Detonation properties won't be updated unless called explicitly!\n\n"
    CJspeed = no_recalc.CJspeed
    no_recalc.T1 = 500
    assert no_recalc.CJspeed == CJspeed

@pytest.mark.parametrize('attribute,value',[('T1',500),('P1',5e5),('X1','H2:21,O2:21,N2:79')])
def test_recalc(det,capsys,attribute,value):
    recalc = det(recalc=True)
    assert capsys.readouterr().out == 'Calculating detonation properties\n'
    CJspeed = recalc.CJspeed
    setattr(recalc, attribute, value)
    assert recalc.CJspeed != CJspeed
    
def test_gamma(det,expected):
    assert round(det(efr.TaylorWave).gamma_eq,5) == round(expected.gamma_eq,5)
    
def test_u2(det,expected):
    assert det(efr.TaylorWave).u2 == expected.CJspeed - expected.a2_eq
    
def test_point_history(det):
    obj = det(efr.TaylorWave)
    hist = obj.point_history(1e-2,5e-5,dt=1e-6)
    assert isinstance(hist, pd.DataFrame)
    assert hist.u.iloc[-1] == 0

@pytest.mark.parametrize('multiprocessing',[False,True])    
def test_time_signal(det,multiprocessing,initState):
    obj = det(efr.TaylorWave)
    obj.znd()
    time_signal = obj.time_signal(1e-2,5e-5,multiprocessing=multiprocessing)
    assert isinstance(time_signal, pd.DataFrame)
    assert all(initState[:2] == time_signal[['T','P']].iloc[0])

@pytest.mark.parametrize('ZND', [False, True])
@pytest.mark.parametrize('multiprocessing',[False,True])    
def test_profile(det,multiprocessing,ZND,initState):
    obj = det(efr.TaylorWave)
    obj.znd()
    profile = obj.profile(1e-5,1e-3,2.5e-2,multiprocessing=multiprocessing,insertZND=ZND)
    assert isinstance(profile, pd.DataFrame)
    assert all(initState[:2] == profile[['T','P']].iloc[-1])
