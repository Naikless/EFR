# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 23:56:58 2022

@author: Niclas
"""

import efr
import pytest

T1 = 293.15
P1 = efr.ct.one_atm
X1 = 'H2:42,O2:21,N2:79'
mech = 'gri30.yaml'

@pytest.fixture
def det():
    
    def _det(cls_type=efr.Detonation, T1=T1, P1=P1, X1=X1, mech=mech, recalc=False):
        return cls_type(T1=T1, P1=P1, X1=X1, mech=mech, recalc=recalc)
    
    return _det


@pytest.mark.parametrize('cls_type',[efr.Detonation, efr.TaylorWave])
def test_det_init(det,cls_type):
    print("Initialize Detonation object")
    obj = det(cls_type=cls_type)
    assert isinstance(det(),efr.Detonation)
    print("Assert that initial state is correct")
    assert (T1,P1,X1,mech) == (obj.T1,obj.P1,obj.X1,obj.mech)
    
def test_CJ_calc(det):
    print("Assert correct CJspeed calculation")
    assert (det().CJspeed - 1969) < 1
    
def test_postShock_fr(det):
    obj = det()
    print("Assert correct frozen post shock calculation")
    assert isinstance(obj.postShock_fr(),efr.ct.Solution)
    assert (obj.postShock_fr().T - 1526) < 1

def test_postShock_eq(det):
    obj = det()
    print("Assert correct equilibrium post shock calculation")
    assert isinstance(obj.postShock_eq,efr.ct.Solution)
    assert (obj.postShock_eq.T - 2943) < 1
    
def test_znd(det):
    obj = det()
    print("Assert correct znd calculation")
    assert isinstance(obj.znd(),dict)
    assert isinstance(obj.znd()['T'],efr.np.ndarray)
    assert (obj.znd()['T'][-1] - 2922) < 1

def test_no_recalc(det,capsys):
    no_recalc = det()
    assert capsys.readouterr().out == "\nDynamic recalculation is disabled! Detonation properties won't be updated unless called explicitly!\n\n"
    CJspeed = no_recalc.CJspeed
    no_recalc.T1 = 500
    assert no_recalc.CJspeed == CJspeed

@pytest.mark.parametrize('attribute,value',[('T1',500),('P1',5e5),('X1','CH4:10.5,O2:21,N2:79')])
def test_recalc(det,capsys,attribute,value):
    recalc = det(recalc=True)
    assert capsys.readouterr().out == 'Calculating detonation properties\n'
    CJspeed = recalc.CJspeed
    setattr(recalc, attribute, value)
    assert recalc.CJspeed != CJspeed
    
def test_gamma(det):
    assert (det(efr.TaylorWave).gamma_eq - 1.16) < 0.1
    
def test_u2(det):
    assert (det(efr.TaylorWave).u2 - 878) < 1
    

pytest.main(['-k','gamma'])
