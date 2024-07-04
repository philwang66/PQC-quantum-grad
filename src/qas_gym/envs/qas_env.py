import sys
sys.path.append("C:/Users/Mac/Documents/Code/quantum-arch-search/src/qas_gym/envs")
from contextlib import closing
from io import StringIO
from typing import Dict, List, Optional, Union

import cirq
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from qulacs.gate import BitFlipNoise,DepolarizingNoise
from qulacs import QuantumGateBase, QuantumState, QuantumCircuit, QuantumCircuitSimulator, Observable, ParametricQuantumCircuit
from copy import deepcopy
from vqc import Parametric_Circuit, get_fidelity_pc, calculate_fidelity
from scipy import optimize

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
        initial: Optional[np.ndarray] = None,
    ):
        super(QuantumArchSearchEnv, self).__init__()

        # set parameters
        self.target = target
        self.num_qubits = n_qubits
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.error_observables = error_observables
        self.error_gates = error_gates
        self.initial = initial
        self.ansatz = ParametricQuantumCircuit(n_qubits)
        # set environment
        self.target_density = target * np.conj(target).T
        # self.simulator = cirq.Simulator()
        # set spaces
        self.observation_space = spaces.Box(low=-1.,
                                            high=1.,
                                            shape=(len(state_observables), ))
        self.action_space = spaces.Discrete(n=len(action_gates))
        self.optim_alg = 'Nelder-Mead'
        self.global_iters = 1000
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
        self.ansatz = ParametricQuantumCircuit(self.num_qubits)
        return self._get_obs()

    def _get_cirq(self, maybe_add_noise=False):
        circuit = QuantumCircuit(self.num_qubits)
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
        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in qubits)
        for idx, gate in enumerate(self.circuit_gates):
            if gate.get_name() == 'X':
               cir_gate = cirq.X(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Y':
                cir_gate = cirq.Y(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Z':
                cir_gate = cirq.Z(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'X-rotation':
                cir_gate = cirq.rx(np.pi/4)(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Y-rotation':
                cir_gate = cirq.ry(np.pi/4)(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Z-rotation':
                cir_gate = cirq.rz(np.pi/4)(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'H':
                cir_gate = cirq.H(qubits[gate.get_target_index_list()[0]])
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


    # def _get_obs(self):
    #     state = QuantumState(self.num_qubits) # deepcopy(self.state)
    #     if self.initial:
    #         state.load(self.initial)  
    #     circuit = self._get_cirq(maybe_add_noise=False)
    #     simulator = QuantumCircuitSimulator(circuit, state)
    #     # obs = [simulator.get_expectation_value(o) for o in self.state_observables]
    #     simulator.simulate()
    #     obs = [o.get_expectation_value(state) for o in self.state_observables]
    #     return np.array(obs).real
    
    def _get_obs(self):
        state = QuantumState(self.num_qubits) # deepcopy(self.state)
        if self.initial:
            state.load(self.initial)  
        simulator = QuantumCircuitSimulator(self.ansatz, state)
        simulator.simulate()
        print("state: ", state.get_vector())
        obs = [o.get_expectation_value(state) for o in self.state_observables]
        return np.array(obs).real       

    # def _get_fidelity(self):
    #     state =  QuantumState(self.num_qubits) # deepcopy(self.state)
    #     if self.initial:
    #         state.load(self.initial)  

    #     circuit = self._get_cirq(maybe_add_noise=False)
    #     simulator = QuantumCircuitSimulator(circuit, state)
    #     simulator.simulate()
    #     print(state.get_vector())
    #     inner = np.inner(np.conj(state.get_vector()), self.target)
    #     fidelity = np.conj(inner) * inner
    #     return fidelity.real

    def _get_fidelity_qulacs(self, circuit):
        state =  QuantumState(self.num_qubits) # deepcopy(self.state)
        if self.initial:
            state.load(self.initial)          
        simulator = QuantumCircuitSimulator(circuit, state)
        simulator.simulate()
        return calculate_fidelity(state, self.target)

    def scipy_optim(self, method, which_angles=[]):
        """
        if only optimize the latest parameter, set which_angles=[-1]
        """
        # qulacs_inst = Parametric_Circuit(n_qubits=self.num_qubits)
        # circuit = qulacs_inst.construct_ansatz(self.circuit_gates)
        circuit = self.ansatz
        parameter_count_qulacs = circuit.get_parameter_count()
        print(r"number of gates:{},  number of param gates:{}".format(circuit.get_gate_count(), parameter_count_qulacs))
        if parameter_count_qulacs > 0:

            thetas = [circuit.get_parameter(ind) for ind in range(parameter_count_qulacs)] 
            x0 = np.asarray(thetas)
            if list(which_angles):
                # print(which_angles)
                # print(x0)
                result_min_qulacs = optimize.minimize(get_fidelity_pc, x0=x0[which_angles],
                                                                args=(circuit,
                                                                    self.num_qubits,
                                                                    self.target,
                                                                    self.initial,
                                                                    which_angles),
                                                                method=method,
                                                                options={'maxiter':self.global_iters})
                # print(result_min_qulacs)
                x0[which_angles] = result_min_qulacs['x']
                # state[-1][state[2]!=self.num_qubits] = torch.tensor(x0, dtype=torch.float)
            else:
                result_min_qulacs = optimize.minimize(get_fidelity_pc, x0=x0,
                                                            args=(circuit,
                                                                self.num_qubits,
                                                                self.target,
                                                                self.initial),
                                                            method=method,
                                                            options={'maxiter':self.global_iters})
                # state[-1][state[2]!=self.num_qubits] = torch.tensor(result_min_qulacs['x'], dtype=torch.float)
                x0 = result_min_qulacs['x']

                # print(-result_min_qulacs['fun'])
                if  result_min_qulacs['success']:
                    result = -result_min_qulacs['fun']
                else:
                    result = -get_fidelity_pc(x0, circuit, self.num_qubits, self.target)
        else:
            result = self._get_fidelity_qulacs(circuit)
            x0 = None

        return result, x0
    
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
        self.circuit_gates.append(action_gate) # 后面不再使用
        ##
        # self.circuit_gates.append(action_gate)
        if action_gate.get_name() in ('X-rotation', 'Y-rotation', 'Z-rotation'): 
            theta = np.random.rand()
            if action_gate.get_name() =='X-rotation':
                self.ansatz.add_parametric_RX_gate(action_gate.get_target_index_list()[0], theta)
            elif action_gate.get_name() =='Y-rotation':
                self.ansatz.add_parametric_RY_gate(action_gate.get_target_index_list()[0], theta)
            elif action_gate.get_name() =='Z-rotation':
                self.ansatz.add_parametric_RZ_gate(action_gate.get_target_index_list()[0], theta)
        elif action_gate.get_name() =='CNOT':
            self.ansatz.add_CNOT_gate(action_gate.get_control_index_list()[0],action_gate.get_target_index_list()[0])
        elif action_gate.get_name() =='H':
            self.ansatz.add_H_gate(action_gate.get_target_index_list()[0])
        else:
            raise TypeError("Not implemented")
        # compute observation
        observation = self._get_obs()
        # print(observation)

        # compute fidelity
        # fidelity = self._get_fidelity()
        # fidelity = self._get_fidelity_estimate()
        fidelity, thetas = self.scipy_optim(self.optim_alg)
        print("Parameters: ", thetas)

        # compute reward
        if fidelity > self.fidelity_threshold:
            reward = fidelity - self.reward_penalty
        else:
            reward = -self.reward_penalty

        # check if terminal
        terminal = (reward > 0.) or (self.ansatz.get_gate_count() >=
                                     self.max_timesteps)

        # return info
        info = {'fidelity': fidelity, 'circuit': self._get_cirq()} #self._get_cirq_with_params()

        return observation, reward, terminal, info

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n' + self._get_cirq_for_vis(False).__str__() + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

