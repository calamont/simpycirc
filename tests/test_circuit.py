import numpy as np
from context import circuitlib
from circuitlib.circuit import NodalAnalysis

freq = 1000

R1_val = 1e3
R2_val = 2e3
R3_val = 3e3
R4_val = 4e3
R_ammeter = 1e-6  # approximate zero-resistance ammeter

C1_val = 100e-12
C2_val = 200e-12
C3_val = 300e-12

L1_val = 100e-6
L2_val = 200e-6
L3_val = 300e-6


def calc_impedance(V, node=-2):
    I = V[:, node] / R_ammeter
    return np.abs(1 / I).flatten()


# Test circuits composed of resistances
@NodalAnalysis(freq=freq)
def decorated_R_circuit():
    return V + R1 + R2 + R3


def test_R_circuit():
    R_correct = R1_val + R2_val
    V_sim = decorated_R_circuit(V=1, R1=R1_val, R2=R2_val, R3=R_ammeter)
    R_calc = calc_impedance(V_sim)
    assert abs(R_calc - R_correct) <= 1e-3


@NodalAnalysis(freq=freq)
def decorated_R_parallel_circuit():
    return V + R1 + (R2 | R3) + R4


def test_R_parallel_circuit():
    R_correct = R1_val + (R2_val * R3_val / (R2_val + R3_val))
    V_sim = decorated_R_parallel_circuit(
        V=1, R1=R1_val, R2=R2_val, R3=R3_val, R4=R_ammeter
    )
    R_calc = calc_impedance(V_sim)
    assert abs(R_calc - R_correct) <= 1e-3


# Test circuits composed of capacitances
@NodalAnalysis(freq=freq)
def decorated_C_circuit():
    return V + C1 + C2 + C3 + R1


def test_C_circuit():
    C_correct = 1 / ((1 / C1_val) + (1 / C2_val) + (1 / C3_val))
    V = decorated_C_circuit(V=1, C1=C1_val, C2=C2_val, C3=C3_val, R1=R_ammeter)[0, 3]
    I = V / R_ammeter
    Z = np.abs(1 / I)
    C_calc = 1 / (2 * np.pi * freq * Z)
    assert abs(C_calc - C_correct) <= 1e-3


# Test circuits composed of impedances
@NodalAnalysis(freq=freq)
def decorated_L_circuit():
    return V + L1 + L2 + L3 + R1


def test_L_circuit():
    L_correct = L1_val + L2_val + L3_val
    V = decorated_L_circuit(V=1, L1=L1_val, L2=L2_val, L3=L3_val, R1=R_ammeter)[0, 3]
    I = V / R_ammeter
    Z = np.abs(1 / I)
    L_calc = Z / (2 * np.pi * freq)
    assert abs(L_calc - L_correct) <= 1e-3
