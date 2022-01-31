# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:46:28 2022

@author: Niclas
"""
import pytest
import cantera as ct
import efr
import sdtoolbox as sdt
from collections import namedtuple

def pytest_addoption(parser):
    parser.addoption(
        "--T1", action="store", default=293.15, help="Initial Temperature"
    )
    parser.addoption(
        "--P1", action="store", default=ct.one_atm, help="Initial Pressure"
    )
    parser.addoption(
        "--X1", action="store", default='H2:42,O2:21,N2:79', help="Initial Composition"
    )
    parser.addoption(
        "--mech", action="store", default='gri30.yaml', help="Kinetic mechanism"
    )

@pytest.fixture(scope='session', autouse=False)
def initState(request):
    return [request.config.getoption(opt) for opt in ['T1','P1','X1','mech']]

@pytest.fixture(scope='session')
def det(initState):
    T1,P1,X1,mech = initState
    def _det(cls_type=efr.Detonation, T1=T1, P1=P1, X1=X1, mech=mech, recalc=False):
        return cls_type(T1=T1, P1=P1, X1=X1, mech=mech, recalc=recalc)
    
    return _det

# expected results
@pytest.fixture
def expected(initState):
    T1,P1,X1,mech = initState
    CJspeed = sdt.postshock.CJspeed(P1, T1, X1, mech)
    postShock_eq = sdt.postshock.PostShock_eq(CJspeed, P1, T1, X1, mech)
    a2_eq = sdt.thermo.soundspeed_eq(postShock_eq)
    gamma_eq = a2_eq**2*postShock_eq.density/postShock_eq.P
    postShock_fr = sdt.postshock.PostShock_fr(CJspeed, P1, T1, X1, mech)
    gas1 = ct.Solution(mech)
    gas1.TPX = T1,P1,X1
    gas = ct.Solution(mech)
    gas.state = postShock_fr.state
    znd = sdt.znd.zndsolve(gas, gas1, CJspeed)
    
    Expected = namedtuple('Expected', ('CJspeed','postShock_fr','postShock_eq','znd','gamma_eq','a2_eq'))
    
    return Expected(CJspeed, postShock_fr, postShock_eq, znd, gamma_eq, a2_eq)
 