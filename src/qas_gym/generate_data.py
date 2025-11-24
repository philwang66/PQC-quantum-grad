import numpy as np
import scipy as sp
from scipy.linalg import expm
from utils import generate_random_hermitian

def GHZ_derived(nqubit, state, dt=0.2):
    obs = generate_random_hermitian(2**nqubit)
    # target = QuantumState(n)
    target = expm(1j *obs * dt) @ state
    return(target)

def calculate_fidelity(initial, target):
    inner = np.inner(np.conj(initial), target)
    return np.conj(inner) * inner

# fidelity = calculate_fidelity(initial, target)

if __name__ == "__main__":
    nqubit = 3
    ndata = 100
    rstate = np.zeros((ndata,2**nqubit), dtype=complex)
    fs = np.zeros(ndata)
    # define GHZ state
    ghz_state = np.zeros(2**nqubit, dtype=complex)
    ghz_state[0] = 1/np.sqrt(2)
    ghz_state[2**nqubit-1] = 1/np.sqrt(2)
    # generate data for dt=0.2
    for i in range(ndata):
        rstate[i,:]= GHZ_derived(nqubit, ghz_state)
        fs[i] = calculate_fidelity(rstate[i,:], ghz_state)
    # print(fs)
    print('Mean fidelity: ', np.mean(fs),   ' Std fidelity: ', np.std(fs))
    rstate02 = rstate

    # generate data for dt=0.1
    for i in range(ndata):
        rstate[i,:]= GHZ_derived(nqubit, ghz_state, 0.1)
        fs[i] = calculate_fidelity(rstate[i,:], ghz_state)
    # print(fs)
    print('Mean fidelity: ', np.mean(fs),   ' Std fidelity: ', np.std(fs))
    rstate01 = rstate

    # generate data for dt=0.5
    rstate = np.zeros((ndata,2**nqubit))
    for i in range(ndata):
        rstate[i,:]= GHZ_derived(nqubit, ghz_state, 0.5)
        fs[i] = calculate_fidelity(rstate[i,:], ghz_state)
    # print(fs)
    print('Mean fidelity: ', np.mean(fs),   ' Std fidelity: ', np.std(fs))
    rstate05 = rstate
    #
    np.savez_compressed('../../data/3q_test.npy', rstate01=rstate01, rstate02=rstate02, rstate05= rstate05)