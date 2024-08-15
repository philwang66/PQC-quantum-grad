import cirq
import numpy as np
from qas_gym.envs.qas_env import QuantumArchSearchEnv
from qas_gym.utils import *


class BasicNQubitEnv(QuantumArchSearchEnv):
    def __init__(self,
                 initial: np.ndarray,
                 target: np.ndarray,
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        n_qubits = int(np.log2(len(target)))
        state_observables = get_default_observables(n_qubits)
        action_gates = get_default_gates(n_qubits)
        super(BasicNQubitEnv,
              self).__init__(target, n_qubits, state_observables, action_gates,
                             fidelity_threshold, reward_penalty, max_timesteps, initial)


class BasicTwoQubitEnv(BasicNQubitEnv):
    def __init__(self,
                 initial: np.ndarray,
                 target: np.ndarray = get_bell_state(),
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        assert len(target) == 4, 'Target must be of size 4'
        super(BasicTwoQubitEnv, self).__init__(initial, target, fidelity_threshold,
                                               reward_penalty, max_timesteps)
        # self.pauli_observables = get_pauli_observables_twoqubits(self.qubits)



class BasicThreeQubitEnv(BasicNQubitEnv):
    def __init__(self,
                initial: np.ndarray,
                 target: np.ndarray = get_ghz_state(3),
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        assert len(target) == 8, 'Target must be of size 8'
        super(BasicThreeQubitEnv, self).__init__(initial, target, fidelity_threshold,
                                                 reward_penalty, max_timesteps)


class BasicFourQubitEnv(BasicNQubitEnv):
    def __init__(self,
                 initial: np.ndarray,
                 target: np.ndarray = get_ghz_state(4),
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        assert len(target) == 16, 'Target must be of size 16'
        super(BasicFourQubitEnv, self).__init__(initial, target, fidelity_threshold,
                                                 reward_penalty, max_timesteps)