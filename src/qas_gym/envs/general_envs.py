import cirq
import numpy as np
from qas_gym.envs.qas_env import QuantumArchSearchEnv, QuantumArchSearchGeneralEnv
from qas_gym.utils import *


class GeneralNQubitEnv(QuantumArchSearchGeneralEnv): #QuantumArchSearchEnv
    def __init__(self,
                 n_qubits,
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20,
                 initial: np.ndarray = None,
                 target: np.ndarray = None,
                 gateset_option: str='default',
                 obs_option: str='default',
                 open_fermion_str: str = None):
        # n_qubits = int(np.log2(len(target)))
        state_observables = get_default_observables(n_qubits, obs_option, open_fermion_str)
        action_gates = get_default_gates(n_qubits, gateset_option)
        super(GeneralNQubitEnv,
              self).__init__(n_qubits, state_observables, action_gates,
                             fidelity_threshold, reward_penalty, max_timesteps, initial, target)




# class GeneralTwoQubitEnv(QuantumArchSearchGeneralEnv):
#     def __init__(self,
#                  fidelity_threshold: float = 0.95,
#                  reward_penalty: float = 0.01,
#                  max_timesteps: int = 20):
#         # assert len(target) == 4, 'Target must be of size 4'
#         n_qubits = 2
#         state_observables = get_default_observables(n_qubits)
#         action_gates = get_default_gates(n_qubits)
#         super(GeneralTwoQubitEnv,
#               self).__init__(n_qubits, state_observables, action_gates,
#                              fidelity_threshold, reward_penalty, max_timesteps)
#         # self.pauli_observables = get_pauli_observables_twoqubits(self.qubits)



# class GeneralThreeQubitEnv(QuantumArchSearchGeneralEnv):
#     def __init__(self,
#                  fidelity_threshold: float = 0.95,
#                  reward_penalty: float = 0.01,
#                  max_timesteps: int = 20):
#         # assert len(target) == 8, 'Target must be of size 8'
#         n_qubits = 3
#         state_observables = get_default_observables(n_qubits)
#         action_gates = get_default_gates(n_qubits)
#         super(GeneralThreeQubitEnv, self).__init__(n_qubits, state_observables, action_gates,
#                              fidelity_threshold, reward_penalty, max_timesteps)


# class GeneralFourQubitEnv(QuantumArchSearchGeneralEnv):
#     def __init__(self,
#                  fidelity_threshold: float = 0.95,
#                  reward_penalty: float = 0.01,
#                  max_timesteps: int = 20):
#         # assert len(target) == 16, 'Target must be of size 16'
#         super(GeneralFourQubitEnv, self).__init__(fidelity_threshold,
#                                                  reward_penalty, max_timesteps)