from typing import Dict, List, Optional, Union

import qulacs
import numpy as np
from qulacs.gate import X, Y, Z, CNOT, CZ, RX, RY, RZ, H, PauliRotation
from qulacs_core import QuantumGateBase
from qulacs import Observable
import gym
import torch.optim as optim
from stable_baselines3 import PPO


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


def fidelity(state1, state2):

    state1 = np.asarray(state1)
    state2 = np.asarray(state2)
    
#     print(np.vdot(state1, state1))
    # Normalize the states
    state1 = state1 / np.linalg.norm(state1)
    state2 = state2 / np.linalg.norm(state2)
    
    # Calculate the fidelity
    fidelity_value = np.abs(np.vdot(state1, state2))**2
    
    return fidelity_value


def PQC_RL(env, learn_steps, state_in, f_in):
    """
    Compute parameterized quantum circuit that tranform state_in to state_out.

    Parameters:
        env (environment): the defined qas_env environment with corresponding initial and target state.
        learn_steps: number of episodes to learn policy
        state_in: initial state
        f_in: initial fidelity  (the logic is if the best fidelity value after learning is no better that the initial value, then return the initial state)

    Returns:
        max_fstate: state with max fidelity
        max_fidelity: max fidelity with learned policy
    """


    # Parameters
    gamma = 0.99
    n_epochs = 4
    clip_range = 0.2
    learning_rate = 0.0001
    policy_kwargs = dict(optimizer_class=optim.Adam)

    # Agent
    ppo_model = PPO("MlpPolicy",
                    env,
                    gamma=gamma,
                    n_epochs=n_epochs,
                    clip_range=clip_range,
                    learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log='logs/')
    
    ppo_model.learn(total_timesteps=learn_steps)
    
    # simulate to get fidelity
    state = env.reset()

    # set max_fidelity, max_fstate to the initial value
    max_fidelity = f_in
    max_fstate = state_in

    done = False
    while not done:
        action = ppo_model.predict(state)
        state, reward, done, info = env.step(action[0])
        #展示当前的线路 和 state
        # env.render()
        # print(state)
        # print(info['fidelity'])
        if info['fidelity'] > max_fidelity:
            max_fidelity = info['fidelity']
            max_fstate = info['state']
    return max_fstate, max_fidelity


def get_PQC_state(state_in, state_out, env_name='BasicFourQubit-v0',reward_penalty = 0.01,max_depth = 10,fidelity_threshold=0.99,train_steps=10000):

    env = gym.make(env_name, target = state_out,
            fidelity_threshold=fidelity_threshold,
            reward_penalty=reward_penalty,
            max_timesteps=max_depth,
            initial = state_in)
    f_ini = fidelity(state_out, state_in)
    
    a, b = PQC_RL(env, train_steps, state_in, f_ini)
    return a, b 