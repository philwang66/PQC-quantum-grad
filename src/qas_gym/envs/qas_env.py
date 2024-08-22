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
from vqc import Parametric_Circuit, get_fidelity_pc, calculate_fidelity
from scipy import optimize
import json

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
        initial: Optional[np.ndarray] = None,
        error_observables: Optional[float] = None,
        error_gates: Optional[float] = None,
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
        self.initial = initial
        self.error_observables = error_observables
        self.error_gates = error_gates
        self.ansatz = ParametricQuantumCircuit(n_qubits)
        # set spaces
        self.observation_space = spaces.Box(low=-1.,
                                            high=1.,
                                            shape=(len(state_observables), ))
        self.action_space = spaces.Discrete(n=len(action_gates))
        self.optim_alg = 'COBYLA' #'Nelder-Mead'
        self.global_iters = 1000
        self.seed()

    def __str__(self):
        desc = 'QuantumArchSearch-v0('
        # desc += '{}={}, '.format('Qubits', len(self.qubits))
        desc += '{}={}, '.format('Initial', self.initial)
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
        seed=None
        self.circuit_gates = []
        self.ansatz = ParametricQuantumCircuit(self.num_qubits)
        return self._get_obs_initial(seed)

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
        i = 0

        for idx, gate in enumerate(self.circuit_gates):
            if gate.get_name() in ('X-rotation', 'Y-rotation' ,'Z-rotation','Pauli-rotation'): 
                angle = self.ansatz.get_parameter(i)
                i += 1

            if gate.get_name() == 'X':
               cir_gate = cirq.X(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Y':
                cir_gate = cirq.Y(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Z':
                cir_gate = cirq.Z(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'X-rotation':
                cir_gate = cirq.rx(angle)(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Y-rotation':
                cir_gate = cirq.ry(angle)(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Z-rotation':
                cir_gate = cirq.rz(angle)(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'H':
                cir_gate = cirq.H(qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'CNOT':
                cir_gate = cirq.CNOT(qubits[gate.get_control_index_list()[0]], qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'CZ':
                cir_gate = cirq.CZ(qubits[gate.get_control_index_list()[0]], qubits[gate.get_target_index_list()[0]])
            # elif gate.get_name() == 'SWAP':
            #     cir_gate = cirq.SWAP(qubits[gate.get_control_index_list()[0]], qubits[gate.get_target_index_list()[0]])
            elif gate.get_name() == 'Pauli-rotation':
                # qlist = gate.get_target_index_list()
                # cir_gate = cirq.SWAP(qubits[qlist[0]], qubits[qlist[1]])
                qlist = gate.get_target_index_list()
                a = json.loads(gate.to_json())
                pauli_ids = [int(_tmp['pauli_id']) for _tmp in a['pauli']['pauli_list']]
                if pauli_ids == [1,1]:
                    cir_gate = cirq.XX(qubits[qlist[0]], qubits[qlist[1]])
                elif pauli_ids ==[2,2]:
                    cir_gate = cirq.YY(qubits[qlist[0]], qubits[qlist[1]])
                elif pauli_ids ==[3,3]:
                    cir_gate = cirq.ZZ(qubits[qlist[0]], qubits[qlist[1]])
                else:
                    print("No correpsonding Pauli-rotation type")
                # cir_gate.on(qubits[qlist[0]], qubits[qlist[1]])
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

    def _get_obs_initial(self, seed=None):
        # print(self.initial)
        state = QuantumState(self.num_qubits)
        if self.initial is not None and seed is None:
            state.load(self.initial)
        else:
            state = QuantumState(self.num_qubits)
            state.set_Haar_random_state(seed)
        obs = [o.get_expectation_value(state) for o in self.state_observables]
        return np.array(obs).real  

    def _get_obs(self):
        state = QuantumState(self.num_qubits)
        if self.initial is not None:
            state.load(self.initial)  
        simulator = QuantumCircuitSimulator(self.ansatz, state)
        simulator.simulate()
        # print("state: ", state.get_vector())
        obs = [o.get_expectation_value(state) for o in self.state_observables]
        return np.array(obs).real       

    def _get_fidelity_qulacs(self, circuit):
        state =  QuantumState(self.num_qubits) # deepcopy(self.state)
        if self.initial is not None:
            state.load(self.initial)          
        simulator = QuantumCircuitSimulator(circuit, state)
        simulator.simulate()
        return calculate_fidelity(state, self.target)

    def scipy_optim(self, method, which_angles=[]):
        """
        if only optimize the latest parameter, set which_angles=[-1]
        """
        circuit = self.ansatz
        parameter_count_qulacs = circuit.get_parameter_count()
        # print(r"number of gates:{},  number of param gates:{}".format(circuit.get_gate_count(), parameter_count_qulacs))
        if parameter_count_qulacs > 0:

            thetas = [circuit.get_parameter(ind) for ind in range(parameter_count_qulacs)] 
            x0 = np.asarray(thetas)
            if list(which_angles):
                # print(which_angles)
                # print(x0)
                bnds = [(0, 2*np.pi) for i in len(which_angles)]
                 
                result_min_qulacs = optimize.minimize(get_fidelity_pc, x0=x0[which_angles],
                                                                args=(circuit,
                                                                    self.num_qubits,
                                                                    self.target,
                                                                    self.initial,
                                                                    which_angles),
                                                                method=method,
                                                                bounds=bnds,
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
        if action_gate.get_name() in ('X-rotation', 'Y-rotation', 'Z-rotation', 'Pauli-rotation'): 
            theta = np.random.rand() * 2*np.pi
            if action_gate.get_name() =='X-rotation':
                self.ansatz.add_parametric_RX_gate(action_gate.get_target_index_list()[0], theta)
            elif action_gate.get_name() =='Y-rotation':
                self.ansatz.add_parametric_RY_gate(action_gate.get_target_index_list()[0], theta)
            elif action_gate.get_name() =='Z-rotation':
                self.ansatz.add_parametric_RZ_gate(action_gate.get_target_index_list()[0], theta)
            elif action_gate.get_name() =='Pauli-rotation':
                a = json.loads(action_gate.to_json())
                pauli_ids = [int(_tmp['pauli_id']) for _tmp in a['pauli']['pauli_list']]
                self.ansatz.add_parametric_multi_Pauli_rotation_gate(action_gate.get_target_index_list(), pauli_ids, theta)


        elif action_gate.get_name() =='CNOT':
            self.ansatz.add_CNOT_gate(action_gate.get_control_index_list()[0],action_gate.get_target_index_list()[0])
        elif action_gate.get_name() =='CZ':
            self.ansatz.add_CZ_gate(action_gate.get_control_index_list()[0],action_gate.get_target_index_list()[0])
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
        # print("Parameters: ", thetas)
        # print("Fidelity:", fidelity)

        # compute reward
        if fidelity > self.fidelity_threshold:
            reward = fidelity - self.fidelity_threshold - self.reward_penalty
        else:
            reward = -self.reward_penalty

        # check if terminal
        terminal = (reward > 0.) or (self.ansatz.get_gate_count() >=
                                     self.max_timesteps)
        _state_in =  QuantumState(self.num_qubits)
        _state_in.load(self.initial)
        simulator = QuantumCircuitSimulator(self.ansatz, _state_in)
        simulator.simulate()     
        state = _state_in.get_vector()

        # return info
        info = {'fidelity': fidelity, 'circuit': self.ansatz, 'state': state} #self._get_cirq_with_params()

        return observation, reward, terminal, info

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n' + self._get_cirq_for_vis(False).__str__() + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

