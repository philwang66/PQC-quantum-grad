from typing import Dict, List, Optional, Union

import qulacs
import numpy as np
from qulacs.gate import X, Y, Z, CNOT, CZ, RX, RY, RZ, H, PauliRotation
from qulacs_core import QuantumGateBase
from qulacs import Observable
# def get_default_gates(
        #   qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
#     gates = []
#     for idx, qubit in enumerate(qubits):
#         next_qubit = qubits[(idx + 1) % len(qubits)]
#         gates += [
#             cirq.rz(np.pi / 4.)(qubit),
#             cirq.X(qubit),
#             cirq.Y(qubit),
#             cirq.Z(qubit),
#             cirq.H(qubit),
#             cirq.CNOT(qubit, next_qubit)
#         ]
#     return gates

def get_default_gates(
        n_qubits) -> List[QuantumGateBase]:
    gates = []
    for qubit in range(n_qubits):
        next_qubit = (qubit + 1) % n_qubits
        gates += [
            RX(qubit, np.pi / 4.),
            RY(qubit, np.pi / 4.),
            RZ(qubit, np.pi / 4.),
            # X(qubit),
            # Y(qubit),
            # Z(qubit),
            # H(qubit),
        ]
    if n_qubits > 1:
        for qubit in range(n_qubits):
            next_qubit = (qubit + 1) % n_qubits
            # gates += [
            #     CNOT(qubit, next_qubit),
            #     CZ(qubit, next_qubit)
            # ]
            for qubit2 in range(qubit+1, n_qubits):
                target_list = [qubit, qubit2]
                gates += [
                    PauliRotation(target_list, [1, 1], np.pi / 4.) ,
                    PauliRotation(target_list, [2, 2], np.pi / 4.) , 
                    PauliRotation(target_list, [3, 3], np.pi / 4.) 
                ]
    return gates

# def get_pauli_observables_twoqubits() -> List[cirq.GateOperation]:
#     assert(len(qubits)==2), 'Qubits number must be 2'
#     observables = []
#     for gate1 in [cirq.X, cirq.Y, cirq.Z, cirq.I]:
#         for gate2 in [cirq.X, cirq.Y, cirq.Z, cirq.I]:
#             observables += [
#                 cirq.PauliString(gate1(qubits[0]) * gate2(qubits[1])) ,
#             ]
#     print("Pauli observables - 2 qubit: ", len(observables))
#     return observables

def get_default_observables(n_qubits) -> List[Observable]:
    observables = []
    for qubit in range(n_qubits):
        Xobs = Observable(n_qubits)
        Xobs.add_operator(1.0, "X {}".format(qubit))
        Yobs = Observable(n_qubits)
        Yobs.add_operator(1.0, "Y {}".format(qubit))
        Zobs = Observable(n_qubits)
        Zobs.add_operator(1.0, "Z {}".format(qubit))
        observables += [
            Xobs, Yobs, Zobs,
        ]
    return observables


def get_bell_state() -> np.ndarray:
    target = np.zeros(2**2, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target


def get_ghz_state(n_qubits: int = 3) -> np.ndarray:
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target
