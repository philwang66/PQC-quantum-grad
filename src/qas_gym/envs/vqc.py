from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.gate import CNOT, RX, RY, RZ

import os

import scipy
import time
import matplotlib.pyplot as plt
import torch
import pathlib

import numpy as np
from sys import argv, stdout

import argparse
# from mlflow import log_metric, log_param, set_experiment


# -----------------------------------------------------------------------------

class Parametric_Circuit:
    def __init__(self,n_qubits):
        self.n_qubits = n_qubits
        self.ansatz = ParametricQuantumCircuit(n_qubits)

    def construct_ansatz(self, action_gates, thetas):
        for i, gate in enumerate(action_gates):
                if gate.get_name() in ('X-rotation', 'Y-rotation', 'Z-rotation'): theta = thetas[i] #np.random.rand()
                if gate.get_name() =='X-rotation':
                    self.ansatz.add_parametric_RX_gate(gate.get_target_index_list()[0], theta)
                elif gate.get_name() =='Y-rotation':
                    self.ansatz.add_parametric_RY_gate(gate.get_target_index_list()[0], theta)
                elif gate.get_name() =='Z-rotation':
                    self.ansatz.add_parametric_RZ_gate(gate.get_target_index_list()[0], theta)
                elif gate.get_name() =='CNOT':
                    self.ansatz.add_CNOT_gate(gate.get_control_index_list()[0],gate.get_target_index_list()[0])
                elif gate.get_name() =='CZ':
                    self.ansatz.add_CZ_gate(gate.get_control_index_list()[0],gate.get_target_index_list()[0])
                elif gate.get_name() =='H':
                    self.ansatz.add_H_gate(gate.get_target_index_list()[0])
                else:
                    raise TypeError("Not implemented")
        # assert self.ansatz.get_gate_count() <= state.shape[1], "Wrong circuit construction, too many gates!!!"
        return self.ansatz
    

# def get_energy_qulacs(angles, observable, circuit, n_qubits, energy_shift, which_angles=[]):
#     """"
#     Function for Qiskit energy minimization using Qulacs
    
#     Input:
#     angles                [array]      : list of trial angles for ansatz
#     observable            [Observable] : Qulacs observable (Hamiltonian)
#     circuit               [circuit]    : ansatz circuit
#     n_qubits              [int]        : number of qubits
#     energy_shift          [float]      : energy shift for Qiskit Hamiltonian after freezing+removing orbitals
    
#     Output:
#     expval [float] : expectation value 
    
#     """
        
#     parameter_count_qulacs = circuit.get_parameter_count()
#     param_qulacs = [circuit.get_parameter(ind) for ind in range(parameter_count_qulacs)]    
#     if not list(which_angles):
#             which_angles = np.arange(parameter_count_qulacs)
    
#     for i, j in enumerate(which_angles):
#         circuit.set_parameter(j, angles[i])
        
#     state = QuantumState(n_qubits)
#     circuit.update_quantum_state(state)   
#     v = state.get_vector()
#     return np.real(np.vdot(v,np.dot(observable,v))) + energy_shift

def calculate_fidelity(state, target):
    inner = np.inner(np.conj(state.get_vector()), target)
    fidelity = np.conj(inner) * inner
    return fidelity.real  

def get_fidelity_pc(angles, circuit, n_qubits, target, initial=None, which_angles=[]):

    parameter_count_qulacs = circuit.get_parameter_count()
    param_qulacs = [circuit.get_parameter(ind) for ind in range(parameter_count_qulacs)]    
    if not list(which_angles):
            which_angles = np.arange(parameter_count_qulacs)
    
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])

    state = QuantumState(n_qubits)
    if initial is not None:
        state.load(initial)   
    
    circuit.update_quantum_state(state)   
    fidelity = calculate_fidelity(state, target)
    return -fidelity



if __name__ == "__main__":
    pass