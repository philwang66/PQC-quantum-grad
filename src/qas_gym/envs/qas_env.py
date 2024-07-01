import sys
from contextlib import closing
from io import StringIO
from typing import Dict, List, Optional, Union

import cirq
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
# from qulacs_core import QuantumGateBase
from qulacs.gate import BitFlipNoise,DepolarizingNoise
from qulacs import QuantumGateBase, QuantumState, QuantumCircuit, QuantumCircuitSimulator, Observable
from copy import deepcopy

class QuantumArchSearchEnv(gym.Env):
    metadata = {'render.modes': ['ansi', 'human']}

    def __init__(
        self,
        target: np.ndarray,
        # qubits: List[int],
        n_qubits: int,
        state_observables: List[Observable],
        action_gates: List[QuantumGateBase],
        fidelity_threshold: float,
        reward_penalty: float,
        max_timesteps: int,
        error_observables: Optional[float] = None,
        error_gates: Optional[float] = None,
    ):
        super(QuantumArchSearchEnv, self).__init__()

        # set parameters
        self.target = target
        self.n_qubits = n_qubits
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.error_observables = error_observables
        self.error_gates = error_gates
        # set environment
        self.target_density = target * np.conj(target).T
        # self.simulator = cirq.Simulator()
        # set spaces
        self.observation_space = spaces.Box(low=-1.,
                                            high=1.,
                                            shape=(len(state_observables), ))
        self.action_space = spaces.Discrete(n=len(action_gates))
        self.seed()

    def __str__(self):
        desc = 'QuantumArchSearch-v0('
        desc += '{}={}, '.format('Qubits', len(self.qubits))
        desc += '{}={}, '.format('Target', self.target)
        desc += '{}=[{}], '.format(
            'Gates', ', '.join(gate.__str__() for gate in self.action_gates))
        desc += '{}=[{}])'.format(
            'Observables',
            ', '.join(gate.__str__() for gate in self.state_observables))
        return desc

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.circuit_gates = []
        return self._get_obs()

    def _get_cirq(self, maybe_add_noise=False):
        circuit = QuantumCircuit(self.n_qubits)
        for gate in self.circuit_gates:
            circuit.add_gate(gate)
            if maybe_add_noise and (self.error_gates is not None):
                noise_gate = DepolarizingNoise(
                    self.error_gates, *gate._qubits)
                circuit.append(noise_gate)
        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = BitFlipNoise(
                self.error_observables, *self.qubits)
            circuit.append(noise_observable)
        return circuit

    def _get_cirq_for_vis(self, maybe_add_noise=False):
        qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in qubits)
        for gate in self.circuit_gates:
            # print(gate.get_name())
            if gate.get_name() == 'X':
               cir_gate = cirq.X(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Y':
                cir_gate = cirq.Y(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Z':
                cir_gate = cirq.Z(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'X-rotation':
                cir_gate = cirq.rx(np.pi/2)(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Y-rotation':
                cir_gate = cirq.ry(np.pi/2)(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Z-rotation':
                cir_gate = cirq.rz(np.pi/2)(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'CNOT':
                cir_gate = cirq.CNOT(qubits[gate.get_control_index_list()[0]], qubits[gate.get_target_index_list()[0]])
            else:
                raise TypeError("Wrong gate type")


            circuit.append(cir_gate)
            if maybe_add_noise and (self.error_gates is not None):
                noise_gate = cirq.depolarize(
                    self.error_gates).on_each(*gate._qubits)
                circuit.append(noise_gate)
        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = cirq.bit_flip(
                self.error_observables).on_each(*self.qubits)
            circuit.append(noise_observable)
        return circuit


    def _get_obs(self):
        state = QuantumState(self.n_qubits) # deepcopy(self.state)
        circuit = self._get_cirq(maybe_add_noise=False)
        simulator = QuantumCircuitSimulator(circuit, state)
        # obs = [simulator.get_expectation_value(o) for o in self.state_observables]
        simulator.simulate()
        obs = [o.get_expectation_value(state) for o in self.state_observables]
        return np.array(obs).real

    def _get_fidelity(self):
        state =  QuantumState(self.n_qubits) # deepcopy(self.state)
        circuit = self._get_cirq(maybe_add_noise=False)
        simulator = QuantumCircuitSimulator(circuit, state)
        simulator.simulate()
        print(state.get_vector())
        inner = np.inner(np.conj(state.get_vector()), self.target)
        fidelity = np.conj(inner) * inner
        return fidelity.real

    # def _get_fidelity_estimate(self):
    #     circuit = self._get_cirq(maybe_add_noise=False)
    #     print(circuit)
    #     simulator = QuantumCircuitSimulator(circuit, self.state)

    #     pred = self.circuit.update_quantum_state(state)
    #     print(pred)
    #     numq = len(self.qubits)
    #     pauli_exp_vals = np.zeros(len(self.pauli_observables))
    #     for i in range(len(self.pauli_observables)):
    #         pauli_exp_vals[i] = self.pauli_observables[i].expectation_from_state_vector(self.target, qubit_map={self.qubits[0]: 0, self.qubits[1]: 1}).real

    #     # print("Expected value pauli observables: ", pauli_exp_vals**2/np.sqrt(len(self.pauli_observables)))
    #     K=min(16, len(self.pauli_observables))
    #     sample_pauli_idx = np.random.choice(len(self.pauli_observables), K, p = pauli_exp_vals**2/np.sqrt(len(self.pauli_observables))) #np.arange(K) #
    #     sample_pauli = [self.pauli_observables[i] for i in sample_pauli_idx]
    #     fidelity_est = 0
    #     result = measure_observables(
    #         circuit, sample_pauli, cirq.Simulator(), stopping_criteria=RepetitionsStoppingCriteria(1000))

    #     for i in range(K):
    #         # sample_pauli_obs = self.pauli_observables[sample_pauli_idx[i]]
    #         # mcircuit = circuit.copy()
    #         # mcircuit.append(cirq.measure_single_paulistring(sample_pauli[i], key='m'))
    #         # sim = cirq.Simulator()
    #         # result2 = sim.run(mcircuit, repetitions=100).measurements['m']
    #         # print(sample_pauli[i], result2.mean(), result2.var(), sample_pauli[i].expectation_from_state_vector(pred, qubit_map={self.qubits[0]: 0, self.qubits[1]: 1}).real)
    #         fidelity_est += sample_pauli[i].expectation_from_state_vector(pred, qubit_map={self.qubits[0]: 0, self.qubits[1]: 1}).real /pauli_exp_vals[sample_pauli_idx[i]]
    #     fidelity_est = fidelity_est/K  #sqrt(len(self.pauli_observables))    
    #     print("Estimated fidelity: ", fidelity_est)
    #     return fidelity_est
    
    def step(self, action):

        # update circuit
        action_gate = self.action_gates[action]
        self.circuit_gates.append(action_gate)

        # compute observation
        observation = self._get_obs()

        # compute fidelity
        fidelity = self._get_fidelity()
        # fidelity = self._get_fidelity_estimate()

        # compute reward
        if fidelity > self.fidelity_threshold:
            reward = fidelity - self.reward_penalty
        else:
            reward = -self.reward_penalty

        # check if terminal
        terminal = (reward > 0.) or (len(self.circuit_gates) >=
                                     self.max_timesteps)

        # return info
        info = {'fidelity': fidelity, 'circuit': self._get_cirq()}

        return observation, reward, terminal, info

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n' + self._get_cirq_for_vis(False).__str__() + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
