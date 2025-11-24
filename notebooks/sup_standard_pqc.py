# %%
import numpy as np
import matplotlib.pyplot as plt
import cirq
import sympy
import pandas as pd
from matplotlib.lines import Line2D
import gymnasium as gym
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import qas_gym
import warnings
from qas_gym.utils import fidelity
import re
import dill

# =============================================================================
# 0) 工具
# =============================================================================
def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

# =============================================================================
# 1) 随机 QNN 电路（同前）
# =============================================================================
def generate_random_qnn(qubits, symbol, depth, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    circuit = cirq.Circuit()
    for q in qubits:
        circuit += cirq.ry(np.pi / 4.0)(q)
    for d in range(depth):
        for i, q in enumerate(qubits):
            r = rng.uniform()
            angle = (symbol if (i == 0 and d == 0) else rng.uniform(0.0, 2.0*np.pi))
            if r > 2.0/3.0:
                circuit += cirq.rz(angle)(q)
            elif r > 1.0/3.0:
                circuit += cirq.ry(angle)(q)
            else:
                circuit += cirq.rx(angle)(q)
        for src, dest in zip(qubits, qubits[1:]):
            circuit += cirq.CZ(src, dest)
    return circuit

# =============================================================================
# 2) 模拟工具与目标
# =============================================================================
def simulate_state(simulator, circuit, symbol, theta):
    resolved = cirq.resolve_parameters(circuit, {symbol: float(theta)})
    res = simulator.simulate(resolved)
    psi = np.asarray(res.final_state_vector, dtype=complex)
    nrm = np.linalg.norm(psi)
    return psi if nrm == 0 else (psi / nrm)

# ---- A) 局域 Z 可观测：C_locZ = (1 - target * <Z_k>)/2 (0–1) ----
def build_local_Zop(n, qidx=0):
    I = np.array([[1,0],[0,1]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    ops = [I]*n
    ops[qidx] = Z
    return kron_all(ops)

def locZ_expectation_from_psi(psi, Zk):
    return float(np.real(np.vdot(psi, Zk @ psi)))

def locZ_expectation(simulator, circuit, symbol, theta, Zk):
    psi = simulate_state(simulator, circuit, symbol, theta)
    return locZ_expectation_from_psi(psi, Zk)

def cost_grad_locZ(simulator, circuit, symbol, theta, Zk, target=1.0, shift=np.pi/2):
    Ez = locZ_expectation(simulator, circuit, symbol, theta, Zk)
    C  = 0.5 * (1.0 - target * Ez)
    Ezp = locZ_expectation(simulator, circuit, symbol, theta + shift, Zk)
    Ezm = locZ_expectation(simulator, circuit, symbol, theta - shift, Zk)
    dEz = 0.5 * (Ezp - Ezm)
    g   = -(target / 2.0) * dEz          # dC/dθ
    return C, g, Ez

# ---- B) 环形 ZZ 能量：C_eng = (⟨H⟩-E0)/(Emax-E0) (0–1) ----
def build_ring_zz_hamiltonian(n, coeff=-0.5):
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        ops = [I] * n
        ops[i] = Z
        ops[(i + 1) % n] = Z
        H += kron_all(ops)
    return coeff * H

def energy_expectation_from_psi(psi, H):
    return float(np.real(np.vdot(psi, H @ psi)))

def energy_expectation(simulator, circuit, symbol, theta, H):
    psi = simulate_state(simulator, circuit, symbol, theta)
    return energy_expectation_from_psi(psi, H)

def cost_grad_energy_normalized(simulator, circuit, symbol, theta, H, E0, Emax, shift=np.pi/2):
    E = energy_expectation(simulator, circuit, symbol, theta, H)
    C = (E - E0) / (Emax - E0)
    Ep = energy_expectation(simulator, circuit, symbol, theta + shift, H)
    Em = energy_expectation(simulator, circuit, symbol, theta - shift, H)
    dE = 0.5 * (Ep - Em)
    g = dE / (Emax - E0)
    return C, g, E

# ---- C) Z^{⊗n}：C_Zn = (1 - target * ⟨Z^{⊗n}⟩)/2 (0–1) ----
def build_Z_tensor_n(n):
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return kron_all([Z] * n)

def Zn_expectation_from_psi(psi, Zn):
    return float(np.real(np.vdot(psi, Zn @ psi)))

def Zn_expectation(simulator, circuit, symbol, theta, Zn):
    psi = simulate_state(simulator, circuit, symbol, theta)
    return Zn_expectation_from_psi(psi, Zn)

def cost_grad_Zn(simulator, circuit, symbol, theta, Zn, target=1.0, shift=np.pi/2):
    Ez = Zn_expectation(simulator, circuit, symbol, theta, Zn)
    C  = 0.5 * (1.0 - target * Ez)
    Ezp = Zn_expectation(simulator, circuit, symbol, theta + shift, Zn)
    Ezm = Zn_expectation(simulator, circuit, symbol, theta - shift, Zn)
    dEz = 0.5 * (Ezp - Ezm)
    g   = -(target / 2.0) * dEz
    return C, g, Ez

# =============================================================================
# 3) 训练器 —— 方法 1：参数移位（param）【记录在更新之前，t=0 即初始点】
# =============================================================================
def train_locZ_param(circuit, symbol, theta0, Zk, target=1.0, iters=200, lr=0.1, simulator=None, shift=np.pi/2):
    if simulator is None:
        simulator = cirq.Simulator()
    theta = float(theta0)
    hist = []
    for t in range(iters):
        C, g, Ez = cost_grad_locZ(simulator, circuit, symbol, theta, Zk, target=target, shift=shift)
        hist.append((t, C, g, theta, Ez))   # 先记录，再更新
        theta -= lr * g
    return hist

def train_energy_param(circuit, symbol, theta0, H, E0, Emax, iters=200, lr=0.1, simulator=None, shift=np.pi/2):
    if simulator is None:
        simulator = cirq.Simulator()
    theta = float(theta0)
    hist = []
    for t in range(iters):
        C, g, E = cost_grad_energy_normalized(simulator, circuit, symbol, theta, H, E0, Emax, shift=shift)
        hist.append((t, C, g, theta, E))
        theta -= lr * g
    return hist

def train_Zn_param(circuit, symbol, theta0, Zn, target=1.0, iters=200, lr=0.1, simulator=None, shift=np.pi/2):
    if simulator is None:
        simulator = cirq.Simulator()
    theta = float(theta0)
    hist = []
    for t in range(iters):
        C, g, Ez = cost_grad_Zn(simulator, circuit, symbol, theta, Zn, target=target, shift=shift)
        hist.append((t, C, g, theta, Ez))
        theta -= lr * g
    return hist

# =============================================================================
# 4) 训练器 —— 方法 2：量子梯度（qgrad）【共享同一 psi0；记录在更新之前】
# =============================================================================
# def train_locZ_qgrad(psi0, Zk, target=1.0, iters=200, lr=0.1):
#     """A: H = -target * Z_k；g = ||lr * H psi||_2；aux = <Z_k>"""
#     psi = psi0.copy()
#     H = -target * Zk
#     hist = []
#     for t in range(iters):
#         Ez = locZ_expectation_from_psi(psi, Zk)
#         C  = 0.5 * (1.0 - target * Ez)
#         hpsi = H @ psi
#         g = np.linalg.norm(lr * hpsi)
#         hist.append((t, C, g, np.nan, Ez))
#         psi = psi - lr * hpsi
#         nrm = np.linalg.norm(psi); psi = psi / (nrm if nrm > 0 else 1.0)
#     return hist

def PQC_RL(env, learn_steps, state_in, state_out, model_weight = None):
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
    policy_kwargs = dict(optimizer_class=optim.Adam, net_arch=dict(pi=[256, 128], vf=[256, 128]))

    # Agent
    ppo_model = PPO("MlpPolicy",
                    env,
                    gamma=gamma,
                    n_epochs=n_epochs,
                    clip_range=clip_range,
                    learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log='logs/')
    if model_weight is not None:
        ppo_model.policy.load_state_dict(model_weight)
    
    ppo_model.learn(total_timesteps=learn_steps)
    
    # simulate to get fidelity
    f_ini = fidelity(state_out, state_in)

    # set max_fidelity, max_fstate to the initial value
    max_fidelity = f_ini
    max_fstate = state_in
    max_obs = None
    circuit = None

    for i in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            action = ppo_model.predict(state)
            state, reward, done, _, info = env.step(action[0])
            #展示当前的线路 和 state
            # env.render()
            # print(state)
            # print(info['fidelity'])
            if info['fidelity'] > max_fidelity:
                max_fidelity = info['fidelity']
                max_fstate = info['state']
                max_obs = state
                circuit = info['circuit']
                # print(circuit)
                # depth = get_circuit_depth(circuit)
                # print("Circuit depth:", depth)
                # print(ppo_model.policy.state_dict())
    return max_fstate, max_fidelity, ppo_model.policy.state_dict(), circuit


def get_PQC_state(state_in, state_out, env_name='GeneralNQubit-v0',reward_penalty = 0.01,max_depth = 10,fidelity_threshold=0.99,
                  train_steps=10000,state_dict=None,pool='default',observable='default',mol=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opf_str = None
        if observable =='OpenFermion':
            try:
                opf_str = str(mol['Hq'])
            except:
                print("Wrong molecule input") 
            
        env = gym.make(env_name, n_qubits = int(np.log2(len(state_in))), 
                fidelity_threshold=fidelity_threshold,
                reward_penalty=reward_penalty,
                max_timesteps=max_depth,
                initial = state_in, 
                target = state_out,
                gateset_option=pool,
                obs_option=observable,
                open_fermion_str = opf_str
                )
        
        a, b, state_dict, s = PQC_RL(env, train_steps, state_in, state_out, model_weight = state_dict)
    return a, b, state_dict, s

def get_twoqubit_gate_num(circuit):
    number = 0
    if circuit:
        cinfo = circuit.to_string()
        match = re.search(r'# of 2 qubit gate:\s*(\d+)', cinfo)
        if match:
            number = int(match.group(1))
            # print(number)  # Output: 1
    return number

def train_locZ_qgrad_pqc(psi0, Zk, target=1.0, iters=200, lr=0.1, maxdepth = 10, max_2qg=100, pool='default'):
    """A: H = -target * Z_k；g = ||lr * H psi||_2；aux = <Z_k>"""
    psi = psi0.copy()
    H = -target * Zk
    state_dict = None
    hist = []
    circuit_depth = 0
    for t in range(iters):
        print("Iter {}: ".format(t))
        psi_init = psi
        Ez = locZ_expectation_from_psi(psi, Zk)
        C  = 0.5 * (1.0 - target * Ez)
        hpsi = H @ psi
        g = np.linalg.norm(lr * hpsi)
        hist.append((t, C, g, np.nan, Ez))
        psi = psi - lr * hpsi
        nrm = np.linalg.norm(psi); psi = psi / (nrm if nrm > 0 else 1.0)

        _f = fidelity(psi_init, psi)
        print("Initial fidelity : {:.4f}: ".format(_f))
        # oneminusf.append(1-_f)
        # 动态调整fidelity_threshold
        ths = max(1-(1-_f)/3., 0.99)
        psi, fidelity_j, state_dict, circuit = get_PQC_state(psi_init, psi, max_depth=maxdepth, fidelity_threshold=ths,train_steps=10000,pool=pool)
        print(" fidelity after PQC : {:.4f}".format(fidelity_j))
        # if circuit:
        circuit_depth += get_twoqubit_gate_num(circuit)
        if circuit_depth>max_2qg:
            break

    return hist

# def train_energy_qgrad(psi0, H, E0, Emax, iters=200, lr=0.1):
#     """B: H = ring ZZ；g = ||lr * H psi||_2；aux = E"""
#     psi = psi0.copy()
#     hist = []
#     for t in range(iters):
#         E = energy_expectation_from_psi(psi, H)
#         C = (E - E0) / (Emax - E0)
#         hpsi = H @ psi
#         g = np.linalg.norm(lr * hpsi)
#         hist.append((t, C, g, np.nan, E))
#         psi = psi - lr * hpsi
#         nrm = np.linalg.norm(psi); psi = psi / (nrm if nrm > 0 else 1.0)
#     return hist

def train_energy_qgrad_pqc(psi0, H, E0, Emax, iters=200, lr=0.1, maxdepth = 10, max_2qg=100, pool='default'):
    """B: H = ring ZZ；g = ||lr * H psi||_2；aux = E"""
    psi = psi0.copy()
    hist = []
    circuit_depth = 0
    for t in range(iters):
        print("Iter {}: ".format(t))
        psi_init = psi       
        E = energy_expectation_from_psi(psi, H)
        C = (E - E0) / (Emax - E0)
        hpsi = H @ psi
        g = np.linalg.norm(lr * hpsi)
        hist.append((t, C, g, np.nan, E))
        psi = psi - lr * hpsi
        nrm = np.linalg.norm(psi); psi = psi / (nrm if nrm > 0 else 1.0)

        _f = fidelity(psi_init, psi)
        print("Initial fidelity : {:.4f}: ".format(_f))
        # oneminusf.append(1-_f)
        # 动态调整fidelity_threshold
        ths = max(1-(1-_f)/3., 0.99)
        psi, fidelity_j, state_dict, circuit = get_PQC_state(psi_init, psi, max_depth=maxdepth, fidelity_threshold=ths,train_steps=10000,pool=pool)
        print(" fidelity after PQC : {:.4f}".format(fidelity_j))
        # if circuit:
        circuit_depth += get_twoqubit_gate_num(circuit)
        if circuit_depth>max_2qg:
            break
    return hist

# def train_Zn_qgrad(psi0, Zn, target=1.0, iters=200, lr=0.1):
#     """C: H = -target * Z^{⊗n}；g = ||lr * H psi||_2；aux = <Z^{⊗n}>"""
#     psi = psi0.copy()
#     H = -target * Zn
#     hist = [] 
#     for t in range(iters):
#         Ez = Zn_expectation_from_psi(psi, Zn)
#         C  = 0.5 * (1.0 - target * Ez)
#         hpsi = H @ psi
#         g = np.linalg.norm(lr * hpsi)
#         hist.append((t, C, g, np.nan, Ez))
#         psi = psi - lr * hpsi
#         nrm = np.linalg.norm(psi); psi = psi / (nrm if nrm > 0 else 1.0)
#     return hist

def train_Zn_qgrad_pqc(psi0, Zn, target=1.0, iters=200, lr=0.1, maxdepth = 10, max_2qg=100, pool='default'):
    """C: H = -target * Z^{⊗n}；g = ||lr * H psi||_2；aux = <Z^{⊗n}>"""
    psi = psi0.copy()
    H = -target * Zn
    hist = [] 
    circuit_depth = 0
    for t in range(iters):
        print("Iter {}: ".format(t))
        psi_init = psi 
        Ez = Zn_expectation_from_psi(psi, Zn)
        C  = 0.5 * (1.0 - target * Ez)
        hpsi = H @ psi
        g = np.linalg.norm(lr * hpsi)
        hist.append((t, C, g, np.nan, Ez))
        psi = psi - lr * hpsi
        nrm = np.linalg.norm(psi); psi = psi / (nrm if nrm > 0 else 1.0)

        _f = fidelity(psi_init, psi)
        print("Initial fidelity : {:.4f}: ".format(_f))
        # oneminusf.append(1-_f)
        # 动态调整fidelity_threshold
        ths = max(1-(1-_f)/3., 0.99)
        psi, fidelity_j, state_dict, circuit = get_PQC_state(psi_init, psi, max_depth=maxdepth, fidelity_threshold=ths,train_steps=10000,pool=pool)
        print(" fidelity after PQC : {:.4f}".format(fidelity_j))
        # if circuit:
        circuit_depth += get_twoqubit_gate_num(circuit)
        if circuit_depth>max_2qg:
            break
    return hist

import seaborn as sns
def plot_case(df, ylabel, title, max_plot_iter=100):

    tmp1 = df #.drop(df[(df.n_qubits==8) & (df.method=="qgrad") & (df.iter>=4)].index)
    gfg = sns.lineplot(data=tmp1, x="iter", y="cost", errorbar ='se', style="method", hue="n_qubits") #

    # sns.xlabel("Iteration"); sns.ylabel(ylabel); sns.title(title)
    gfg.set(xlabel ="Iteration", ylabel = ylabel, title = title)
    fig = gfg.get_figure()
    fig.savefig("_".join(title.split(' ')[:1])+".pdf",  format='pdf', bbox_inches='tight')

# %%
# =============================================================================
# 5) 主流程：三目标（param vs qgrad）
# =============================================================================
def run_experiment(n_qubits_list, depth=10, iters=100, lr=0.1, seed=2025,
                   qidx=0,                 # 局域 Z 的比特索引
                   csv_locZ="bp_locZ_zero.csv",
                   csv_eng ="bp_energy_zero.csv",
                   csv_zn  ="bp_Zn_zero.csv",
                   zn_target=1.0,
                   locZ_target=1.0,
                   numrun=10):
    rng = np.random.default_rng(seed)
    sim = cirq.Simulator()

    rows_locZ, rows_eng, rows_zn = [], [], []

    for n in n_qubits_list:
        print("Qubits: ", n)
        qubits = cirq.GridQubit.rect(1, n)
        symbol = sympy.Symbol('theta')

        # average over n runs for random circuit
        for i in range(numrun):
            local_rng = np.random.default_rng(seed + n +i)

            circuit = generate_random_qnn(qubits, symbol, depth, rng=local_rng)
            theta0  = local_rng.uniform(0.0, 2.0*np.pi)
            psi0    = simulate_state(sim, circuit, symbol, theta0)  # 共享初始态

            # ---------- A: 局域 Z ----------
            Zk = build_local_Zop(n, qidx=min(qidx, n-1))
            for (t, C, g, th, Ez) in train_locZ_param(circuit, symbol, theta0, Zk, target=locZ_target, iters=iters, lr=lr, simulator=sim):
                rows_locZ.append(("param", n, t, C, g, th, Ez, locZ_target, qidx))

            # ---------- B: 能量 ----------
            H = build_ring_zz_hamiltonian(n, coeff=-0.5)
            evals = np.linalg.eigvalsh(H)
            E0, Emax = float(np.min(evals)), float(np.max(evals))
            for (t, C, g, th, E) in train_energy_param(circuit, symbol, theta0, H, E0, Emax, iters=iters, lr=lr, simulator=sim):
                rows_eng.append(("param", n, t, C, g, th, E, E0, Emax))

            # ---------- C: Z^{⊗n} ----------
            Zn = build_Z_tensor_n(n)
            for (t, C, g, th, Ez) in train_Zn_param(circuit, symbol, theta0, Zn, target=zn_target, iters=iters, lr=lr, simulator=sim):
                rows_zn.append(("param", n, t, C, g, th, Ez, zn_target))

            ################## used for terminitions on quantum graident for pqc training 
            n_gate= (depth-1) * n
            for (t, C, g, th, Ez) in train_locZ_qgrad_pqc(psi0, Zk, target=locZ_target, iters=iters, lr=lr, max_2qg = n_gate, pool='PauliAll2'):
                rows_locZ.append(("qgrad", n, t, C, g, th, Ez, locZ_target, qidx))

            for (t, C, g, th, E) in train_energy_qgrad_pqc(psi0, H, E0, Emax, iters=iters, lr=lr, max_2qg = n_gate, pool='PauliAll2'):
                rows_eng.append(("qgrad", n, t, C, g, th, E, E0, Emax))

            for (t, C, g, th, Ez) in train_Zn_qgrad_pqc(psi0, Zn, target=zn_target, iters=iters, lr=lr, max_2qg = n_gate, pool='PauliAll2'):
                rows_zn.append(("qgrad", n, t, C, g, th, Ez, zn_target))

    df_locZ = pd.DataFrame(rows_locZ, columns=["method","n_qubits","iter","cost","grad","theta","Ez","target","qidx"])
    df_eng  = pd.DataFrame(rows_eng,  columns=["method","n_qubits","iter","cost","grad","theta","E","E0","Emax"])
    df_zn   = pd.DataFrame(rows_zn,   columns=["method","n_qubits","iter","cost","grad","theta","Ez","target"])

    if csv_locZ: df_locZ.to_csv(csv_locZ, index=False); print(f"[INFO] saved: {csv_locZ}")
    if csv_eng:  df_eng.to_csv(csv_eng,   index=False);  print(f"[INFO] saved: {csv_eng}")
    if csv_zn:   df_zn.to_csv(csv_zn,     index=False);  print(f"[INFO] saved: {csv_zn}")


    return df_locZ, df_eng, df_zn

# =============================================================================
# 6) 入口
# =============================================================================
if __name__ == "__main__":
    n_qubits_list = [4, 6, 8, 10]
    depth = 20
    iters = 20
    lr = 0.5
    qidx=0
    df_locZ, df_eng, df_zn = run_experiment(
        n_qubits_list, depth=depth, iters=iters, lr=lr, seed=2025,
        qidx=qidx,                # 选观测 Z 的比特索引
        csv_locZ="bp_locZ_zero.csv",
        csv_eng ="bp_energy_zero.csv",
        csv_zn  ="bp_Zn_zero.csv",
        zn_target=1.0,
        locZ_target=1.0,
        numrun=10
    )
    dill.dump_session(f"result/example_Ramdom_PQC_lr05_256_128_Pauli2All_depth20_10repeat.db")


    plot_case(df_locZ, r"$C_{\rm locZ}=\frac{1-\mathrm{target}\,\langle Z_{q}\rangle}{2}$ (0–1)",
                f"Local Z objective (q={qidx}): param vs qgrad",20)
    plot_case(df_eng,  r"$C_{\rm eng}=(\langle H\rangle-E_0)/(E_{\max}-E_0)$ (0–1)",
                "Ring ZZ energy objective: param vs qgrad",20)
    plot_case(df_zn,   r"$C_{Z^n}=\frac{1-\mathrm{target}\,\langle Z^{\otimes n}\rangle}{2}$ (0–1)",
                r"Global $Z^{\otimes n}$ objective: param vs qgrad", 25)

