# chem_lih_h2o_active_openfermion.py
# Build 6-qubit LiH and 8-qubit H2O Hamiltonians (STO-3G) via
# freeze-core + remove-highest-virtuals, JW mapping, return SciPy CSR.

from typing import List, Tuple, Dict
import numpy as np
from scipy.sparse import csr_matrix

from openfermion import MolecularData, get_fermion_operator, normal_ordered, count_qubits
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion.linalg import jw_hartree_fock_state

def _active_from_freeze_remove(n_orb: int, freeze_list: List[int], remove_list: List[int]) -> Tuple[List[int], List[int]]:
    """Given spatial-orbital freeze/remove (0..n_orb-1; negatives allowed),
       return (occupied_indices, active_indices) for OpenFermion."""
    # normalize negatives to [0, n_orb)
    freeze = sorted({i % n_orb for i in freeze_list})
    remove = sorted({i % n_orb for i in remove_list})
    # active = remaining after freeze & remove
    all_idx = set(range(n_orb))
    active = sorted(list(all_idx - set(freeze) - set(remove)))
    return freeze, active

def _molecule_to_qubit_csr(geom, basis: str, mult: int, charge: int,
                           freeze_list: List[int], remove_list: List[int],
                           description: str = "") -> Dict:
    """Build qubit Hamiltonian (JW, no taper) with given freeze/remove."""
    # 1) Mean-field & integrals
    mol = MolecularData(geom, basis, mult, charge, description=description)
    mol = run_pyscf(mol, run_scf=True, run_fci=False)
    n_orb = mol.n_orbitals

    # 2) active space selection (spatial-orbital indices)
    occ_idx, act_idx = _active_from_freeze_remove(n_orb, freeze_list, remove_list)
    if len(act_idx) == 0:
        raise ValueError("Active space empty; adjust freeze/remove lists.")

    # 3) 2nd-quantized Hamiltonian with core energy folded in
    H2 = mol.get_molecular_hamiltonian(occupied_indices=occ_idx, active_indices=act_idx)
    Hf = normal_ordered(get_fermion_operator(H2))

    # 4) JW mapping → sparse
    Hq = jordan_wigner(Hf)
    n_qubits = count_qubits(Hq)
    Hs = get_sparse_operator(Hq).tocsr()

    const_shift = float(H2.constant)  # includes nuclear repulsion + frozen-core contrib.
    return dict(Hf=Hf, Hq=Hq, Hs=Hs, n_qubits=n_qubits, const_shift=const_shift,
                occ_idx=occ_idx, act_idx=act_idx, n_orb=n_orb, mol=mol)

# ---------------------------
# Public builders (default geometries from常用设置/论文)
# ---------------------------

def build_LiH_6q(geometry=None):
    """
    LiH (STO-3G), target 6 qubits:
    - freeze 1 lowest (≈ Li 1s)
    - remove 2 highest virtuals (≈ Li 2p_x, 2p_y)
    """
    geom = geometry or [('Li',(0.0,0.0,0.0)), ('H',(0.0,0.0,2.2))]  # JW 6q geometry
    # We'll take n_orb at runtime; here specify relative lists:
    freeze_list = [0]          # lowest-energy spatial orbital
    remove_list = [-3, -2]     # two highest virtual spatial orbitals
    out = _molecule_to_qubit_csr(geom, 'sto-3g', mult=1, charge=0,
                                 freeze_list=freeze_list, remove_list=remove_list,
                                 description="LiH_6q")
    assert out['n_qubits'] == 6, f"Expected 6 qubits, got {out['n_qubits']}"
    return out

def build_H2O_8q(geometry=None):
    """
    H2O (STO-3G), target 8 qubits:
    - freeze 1 lowest (≈ O 1s)
    - remove 2 highest virtuals
    Geometry defaults to a common planar structure (Å)；你也可替换成论文的坐标。
    """
    geom = geometry or [
        ('O', (0.000000, 0.000000, 0.000000)),
        ('H', (0.000000, 0.757160, 0.586260)),
        ('H', (0.000000, -0.757160, 0.586260)),
    ]
    freeze_list = [0]
    remove_list = [-2, -1]
    out = _molecule_to_qubit_csr(geom, 'sto-3g', mult=1, charge=0,
                                 freeze_list=freeze_list, remove_list=remove_list,
                                 description="H2O_8q")
    assert out['n_qubits'] == 8, f"Expected 8 qubits, got {out['n_qubits']}"
    return out

# ---------------------------
# Quick test / print
# ---------------------------
if __name__ == "__main__":
    lih = build_LiH_6q()
    n_e_total   = lih['mol'].n_electrons
    # n_e_active_lih  = n_e_total - 2*len(lih['occ_idx'])
    n_e_active_lih  = n_e_total-2
    print(f"[LiH] n_orb={lih['n_orb']}, occ={lih['occ_idx']}, act={lih['act_idx']}, n_qubits={lih['n_qubits']}")
    print(f"[LiH] constant shift (Ha): {lih['const_shift']:.12f}")

    h2o = build_H2O_8q()
    n_e_total   = h2o['mol'].n_electrons
    # n_e_active_h2o  = n_e_total - 2*len(h2o['occ_idx'])
    n_e_active_h2o  = n_e_total -2

    print(f"[H2O] n_orb={h2o['n_orb']}, occ={h2o['occ_idx']}, act={h2o['act_idx']}, n_qubits={h2o['n_qubits']}")
    print(f"[H2O] constant shift (Ha): {h2o['const_shift']:.12f}")

    # Optional: exact ground energy of the qubit Hamiltonians (small sizes)
    from scipy.sparse.linalg import eigsh
    e_lih, _ = eigsh(lih['Hs'], k=1, which='SA'); e_lih = float(e_lih[0])
    e_h2o, _ = eigsh(h2o['Hs'], k=1, which='SA'); e_h2o = float(e_h2o[0])
    print(f"[LiH 6q] ED ground energy (Ha): {e_lih:.12f}   (+ shift={lih['const_shift']:.12f})")
    print(f"[H2O 8q] ED ground energy (Ha): {e_h2o:.12f}   (+ shift={h2o['const_shift']:.12f})")


# %% [markdown]
# #  2. adapt 算法需要的子函数

# %%
# demo_breadapt_maxcut_pauli_pool_logging.py
# Max-Cut with full 1- and 2-qubit Pauli generator pool.
# Capture breadapt() logs without modifying the library, export CSV, and plot convergence.
import os, io, re, csv, contextlib
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, identity, kron
from adapt.driver import Xiphos, t_ucc_state, t_ucc_E


# =========================
# Linear algebra helpers
# =========================
def I2(): return csr_matrix(np.eye(2, dtype=complex))
def X2(): return csr_matrix(np.array([[0,1],[1,0]], dtype=complex))
def Y2(): return csr_matrix(np.array([[0,-1j],[1j,0]], dtype=complex))
def Z2(): return csr_matrix(np.array([[1,0],[0,-1]], dtype=complex))
PAULI_MAP = {'I': I2, 'X': X2, 'Y': Y2, 'Z': Z2}
def H2():
    return (1/np.sqrt(2)) * csr_matrix(np.array([[1, 1],
                                                 [1,-1]], dtype=complex))

def identity_n(n):
    return identity(2**n, dtype=complex, format='csr')


def kron_n(ops):
    out = ops[0]
    for op in ops[1:]:
        out = kron(out, op, format="csr")
    return out

def tensor_product(*labels):
    """tensor_product('Z','Z','I','I') → ZZII (CSR)"""
    ops = [PAULI_MAP[L]() for L in labels]  # noqa: E999
    # ↑ 注意：你的编辑器可能把这行里的反引号标红；请把 ` 去掉。正确写法如下：
    # ops = [PAULI_MAP[L]() for L in labels]
    return kron_n(ops)

def one_qubit_op(n, i, base):
    ops = [base if q == i else I2() for q in range(n)]
    return kron_n(ops)

def two_qubit_op(n, i, j, base_i, base_j):
    ops = []
    for q in range(n):
        if q == i: ops.append(base_i)
        elif q == j: ops.append(base_j)
        else: ops.append(I2())
    return kron_n(ops)

def two_qubit_ZZ(n, i, j):
    return one_qubit_op(n, i, Z2()) @ one_qubit_op(n, j, Z2())

def zero_state(n):
    vec = np.zeros((2**n, 1), dtype=complex); vec[0,0] = 1.0
    return csc_matrix(vec)

def hadamard_layer_state(n, start='zero'):
    """
    H^{⊗n} applied to |0...0> (start='zero') -> |+>^{⊗n}
    or to |1...1> (start='one') -> |->^{⊗n}
    """
    Hn = kron_n([H2()] * n)
    base = zero_state(n) if start == 'zero' else one_state(n)
    return csc_matrix(Hn @ base)

# =========================
# Pauli generator pool: all 1- and 2-qubit
# =========================
def one_two_qubit_pauli_pool(n):
    """
    Return (pool, v_pool) where each generator is -i/2 * P
    with P in {X,Y,Z} on 1 or 2 sites (no identity-only).
    Count: 3n + 9*C(n,2)
    """
    pool, v_pool = [], []
    # 1-qubit
    for i in range(n):
        for label, base in (('X', X2()), ('Y', Y2()), ('Z', Z2())):
            P = one_qubit_op(n, i, base)
            K = (-0.5j) * P
            pool.append(K.tocsr())
            v_pool.append(f"-i/2 * {label}{i}")
    # 2-qubit
    for i in range(n):
        for j in range(i+1, n):
            for li, bi in (('X', X2()), ('Y', Y2()), ('Z', Z2())):
                for lj, bj in (('X', X2()), ('Y', Y2()), ('Z', Z2())):
                    P = two_qubit_op(n, i, j, bi, bj)
                    K = (-0.5j) * P
                    pool.append(K.tocsr())
                    v_pool.append(f"-i/2 * {li}{i}{lj}{j}")
    return pool, v_pool

def chem_gate_pool(n):
    """
    Return (pool, v_pool) for the parameterized gate set:
      {RX, RY, RZ, RZZ, RYXXY, RXXYY, Controlled-RYXXY, Controlled-RXXYY}.
    Each entry is an anti-Hermitian generator K = (-i/2) * H_gen (CSR matrix).
    Labels are human-readable.

    RXXYY(i,j):  H_gen = X_i X_j + Y_i Y_j
    RYXXY(i,j):  H_gen = Y_i X_j + X_i Y_j
    Controlled-* : H_gen = [(I - Z_c)/2] ⊗ H_target(i,j)
    """
    pool, v_pool = [], []
    I_n = identity_n(n)

    # --- 1) Single-qubit rotations RX/RY/RZ ---
    for i in range(n):
        for name, base in (("RX", X2()), ("RY", Y2()), ("RZ", Z2())):
            H_gen = one_qubit_op(n, i, base)
            K = (-0.5j) * H_gen
            pool.append(K.tocsr())
            v_pool.append(f"{name} q{i} (K=-i/2·{name[-1]}_{i})")

    # --- 2) Two-qubit rotations: RZZ, RXXYY, RYXXY ---
    for i in range(n):
        for j in range(i+1, n):
            # RZZ
            H_gen_zz = two_qubit_op(n, i, j, Z2(), Z2())
            pool.append((-0.5j) * H_gen_zz)
            v_pool.append(f"RZZ q{i},q{j} (K=-i/2·Z_{i}Z_{j})")

            # RXXYY: X_i X_j + Y_i Y_j
            H_xx = two_qubit_op(n, i, j, X2(), X2())
            H_yy = two_qubit_op(n, i, j, Y2(), Y2())
            H_gen_xxyy = H_xx + H_yy
            pool.append((-0.5j) * H_gen_xxyy)
            v_pool.append(f"RXXYY q{i},q{j} (K=-i/2·(X_{i}X_{j}+Y_{i}Y_{j}))")

            # RYXXY: Y_i X_j + X_i Y_j
            H_yx = two_qubit_op(n, i, j, Y2(), X2())
            H_xy = two_qubit_op(n, i, j, X2(), Y2())
            H_gen_yxxy = H_yx + H_xy
            pool.append((-0.5j) * H_gen_yxxy)
            v_pool.append(f"RYXXY q{i},q{j} (K=-i/2·(Y_{i}X_{j}+X_{i}Y_{j}))")

    # --- 3) Controlled versions: c ∈ {0..n-1} \ {i,j} ---
    # Projector onto |1> on control: P1(c) = (I - Z_c)/2
    for c in range(n):
        Zc = one_qubit_op(n, c, Z2())
        P1c = 0.5 * (I_n - Zc)  # CSR

        for i in range(n):
            if i == c: continue
            for j in range(i+1, n):
                if j == c: continue

                # Controlled-RXXYY
                H_xx = two_qubit_op(n, i, j, X2(), X2())
                H_yy = two_qubit_op(n, i, j, Y2(), Y2())
                H_target = H_xx + H_yy
                H_gen_ctrl = P1c @ H_target  # commutes → Hermitian
                pool.append((-0.5j) * H_gen_ctrl)
                v_pool.append(f"Controlled-RXXYY ctrl q{c} | targets q{i},q{j} (K=-i/2·Π1_{c}·(X_{i}X_{j}+Y_{i}Y_{j}))")

                # Controlled-RYXXY
                H_yx = two_qubit_op(n, i, j, Y2(), X2())
                H_xy = two_qubit_op(n, i, j, X2(), Y2())
                H_target = H_yx + H_xy
                H_gen_ctrl = P1c @ H_target
                pool.append((-0.5j) * H_gen_ctrl)
                v_pool.append(f"Controlled-RYXXY ctrl q{c} | targets q{i},q{j} (K=-i/2·Π1_{c}·(Y_{i}X_{j}+X_{i}Y_{j}))")

    return pool, v_pool

def bits_str(bits): return ''.join(str(b) for b in bits)


def run_breadapt_with_logging(xiphos_obj, system_tag, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        err = xiphos_obj.breadapt(**kwargs)
    text = buf.getvalue()

    # # 可选：把原始日志回放到屏幕（便于查看）
    # print(text)

    # 解析
    cur_iter = None
    rows = []
    cur_rec = {}

    def flush_cur():
        nonlocal cur_rec
        if "iter" in cur_rec:
            rows.append(cur_rec)
        cur_rec = {}

    for line in text.splitlines():
        m = LOG_PATTERNS["iter"].search(line)
        if m:
            flush_cur()
            cur_iter = int(m.group(1))
            cur_rec["iter"] = cur_iter
            continue
        m = LOG_PATTERNS["next_op"].search(line)
        if m and cur_iter is not None:
            cur_rec["op"] = m.group(1).strip()
            continue
        m = LOG_PATTERNS["assoc_grad"].search(line)
        if m and cur_iter is not None:
            cur_rec["assoc_grad"] = float(m.group(1))
            continue
        m = LOG_PATTERNS["gnorm"].search(line)
        if m and cur_iter is not None:
            cur_rec["gnorm"] = float(m.group(1))
            continue
        m = LOG_PATTERNS["Cos"].search(line)
        if m and cur_iter is not None:
            cur_rec["Cos"] = float(m.group(1))
            continue

    # flush_cur()

    # conv = {}
    # for key in ("conv_E", "conv_err", "conv_gnorm", "conv_fid"):
    #     m = LOG_PATTERNS[key].search(text)
    #     if m:
    #         conv[key] = float(m.group(1))

    # CSV
    csv_path = f"{system_tag}/adapt_metrics.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"[Saved] {csv_path}  （列：iter, op, assoc_grad, gnorm, Cos）")
    else:
        print("[Warn] 没解析到迭代记录（也许很快就收敛了？）")

    # # 图
    # if rows:
    #     df = pd.DataFrame(rows).sort_values("iter")

    #     plt.figure()
    #     plt.plot(df["iter"], df["Cos"], marker="o")
    #     plt.xlabel("Iteration"); plt.ylabel("Cost function")
    #     plt.title("ADAPT: Cost convergence")
    #     plt.grid(True); plt.tight_layout()
    #     plt.savefig(f"{system_tag}/Cos_curve.png", dpi=180)

    #     plt.figure()
    #     plt.semilogy(df["iter"], df["gnorm"], marker="o")
    #     plt.xlabel("Iteration"); plt.ylabel("Gradient norm")
    #     plt.title("ADAPT: Gradient-norm convergence")
    #     plt.grid(True, which="both"); plt.tight_layout()
    #     plt.savefig(f"{system_tag}/gnorm_curve.png", dpi=180)

    #     print(f"[Saved] {system_tag}/Cos_curve.png, {system_tag}/gnorm_curve.png")

    return err, rows, text

# %% [markdown]
# # 3. 分别构造两类优化问题 LiH 和 H2O
# 

# %%
# =========================
# Config
# =========================
SYSTEM_TAG = "MAXCUT_4Q_PauliPool_logging"
os.makedirs(SYSTEM_TAG, exist_ok=True)


# ADAPT/breadapt params
ADAPT_KW = dict(
    params=[], ansatz=[], ref=None,  # ref will be filled later
    Etol=1e-8, gtol=None, max_depth=80,
    guesses=0, hf=False, n=1, threads=1, seed=123,
    criteria="grad"
)


# =========================
# Log capture → CSV → plots (no library changes)
# =========================
LOG_PATTERNS = {
    "iter": re.compile(r"ADAPT Iteration\s+(\d+)"),
    "next_op": re.compile(r"^Next operator to be added:\s*(.+)$"),
    "assoc_grad": re.compile(r"^Associated gradient:\s*([-\dEe\.+]+)"),
    "gnorm": re.compile(r"^Gradient norm:\s*([-\dEe\.+]+)"),
    "Cos": re.compile(r"^Cost function:\s*([-\dEe\.+]+)"),
    "conv_E": re.compile(r"^Converged ADAPT energy:\s*([-\dEe\.+]+)"),
    "conv_err": re.compile(r"^Converged ADAPT error:\s*([-\dEe\.+]+)"),
    "conv_gnorm": re.compile(r"^Converged ADAPT gnorm:\s*([-\dEe\.+]+)"),
    "conv_fid": re.compile(r"^Converged ADAPT fidelity:\s*([-\dEe\.+]+)"),
}

# =====================================================================
# === 添加：量子梯度下降（QGrad）与 b-ADAPT 能量曲线对比 ===============
# =====================================================================
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

def _to_dense_ket(vec):
    """把 ref（可能是稀疏列向量/数组）转成归一化的一维 ket（np.ndarray, complex）。"""
    if sp.issparse(vec):
        arr = vec.toarray()
    else:
        arr = np.asarray(vec)
    arr = arr.reshape(-1)  # 展成 1D
    nrm = np.linalg.norm(arr)
    if nrm == 0:
        raise ValueError("Reference state has zero norm.")
    return (arr / nrm).astype(np.complex128)

def _matvec(H, psi):
    """支持稀疏/致密 H 乘以致密向量 psi。"""
    if sp.issparse(H):
        return H.dot(psi)
    return H @ psi

def estimate_op_norm_power(H, n_dim, iters=30, seed=1):
    """幂迭代估算 ||H||_2（最大奇异值/谱半径，H 厄米时等于最大特征值绝对值）。"""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n_dim) + 1j * rng.normal(size=n_dim)
    v = v / np.linalg.norm(v)
    for _ in range(iters):
        w = _matvec(H, v)
        nrm = np.linalg.norm(w)
        if nrm == 0:
            return 1.0
        v = w / nrm
    # Rayleigh quotient 近似谱范数
    w = _matvec(H, v)
    rq = np.vdot(v, w).real
    return max(abs(rq), 1.0)

def qgrad_descent_energy(H, ref_ket, steps, lr=None, norm_iters=30):
    """
    虚时间式 QGrad：psi_{t+1} = psi_t - lr * H psi_t（每步归一化）
    返回 energies（长度=steps），以及最终 psi。
    """
    d = ref_ket.size
    psi = ref_ket.astype(np.complex128, copy=True)
    # 学习率自适应：lr = 0.1 / ||H||_2
    if lr is None:
        Hnorm = estimate_op_norm_power(H, d, iters=norm_iters)
        lr = 0.1 / Hnorm
        print(f"[QGrad] Using lr = 0.1 / ||H||_2 ≈ {lr:.6g} (||H||_2≈{Hnorm:.6g})")

    energies = []
    for t in range(steps):
        Hpsi = _matvec(H, psi)
        E = float(np.real(np.vdot(psi, Hpsi)))
        energies.append(E)
        # 梯度步进 + 归一化
        psi = psi - lr * Hpsi
        nrm = np.linalg.norm(psi)
        if nrm == 0:
            raise ValueError("State collapsed to zero vector during QGrad.")
        psi = psi / nrm
    return np.array(energies, dtype=float), psi, lr

def _rows_to_df(rows, raw_log=None):
    """
    尽量从 run_breadapt_with_logging 返回的 rows 构造成 DataFrame。
    自动猜测列名：迭代列（iter/step/depth）、能量列（E/energy/...）。
    """
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    elif isinstance(rows, list) and len(rows) > 0:
        if isinstance(rows[0], dict):
            df = pd.DataFrame(rows)
        else:
            # 如果是元组/列表，尝试直接转 DF；用户可手动改列名
            df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame()

    if df.empty and isinstance(raw_log, str):
        # 简易兜底：从原始日志里抽取 "iter=, E=" 之类字段（可按你的日志格式调整）
        import re
        iters, Es = [], []
        for line in raw_log.splitlines():
            m_iter = re.search(r'(?:iter|depth|step)\s*=\s*(\d+)', line, re.I)
            m_E    = re.search(r'(?:E|energy)\s*=\s*([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)', line)
            if m_iter and m_E:
                iters.append(int(m_iter.group(1)))
                Es.append(float(m_E.group(1)))
        if iters:
            df = pd.DataFrame({"iter": iters, "E": Es})

    # 规范列名为小写，便于匹配
    df.columns = [str(c).strip() for c in df.columns]
    lowmap = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lowmap)

    # 猜测迭代列
    iter_col = None
    for key in ["iter", "depth", "step", "k", "t"]:
        if key in df.columns:
            iter_col = key
            break
    if iter_col is None:
        # 如果没有迭代列，就补一个 0..n-1
        df["iter"] = np.arange(len(df))
        iter_col = "iter"

    # 猜测能量列
    energy_col = None
    candidates = [c for c in df.columns if c in ("e","energy","<h>","en","e_total","E","E_total")]
    if not candidates:
        # 放宽：包含 'e' 或 'ener' 的列名
        candidates = [c for c in df.columns if "ener" in c or c == "e"]
    energy_col = candidates[0] if candidates else None

    # 若还没找到，尝试从数值列里找一列最像能量（包含负数，波动递减）
    if energy_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            energy_col = num_cols[0]  # 兜底：第一列数值
            print(f"[WARN] energy column not found. Using '{energy_col}' as energy.")

    return df[[iter_col, energy_col]].rename(columns={iter_col: "iter", energy_col: "E"})





# -----------------------
# 2) 量子梯度下降（与 breadapt 相同步数）
# -----------------------
# ref → ket（1D 复向量）
N=6
# A) zero state
# ref = zero_state(N)
# # B) hartree fock state 
_ref = jw_hartree_fock_state(n_e_active_lih,N)
ref=csc_matrix(_ref).T.conj()
# equal weights state
# ref = hadamard_layer_state(N, start='zero')

ADAPT_KW["ref"] = ref
# Pauli 生成元池（1-、2-比特全体）
# pool, v_pool = one_two_qubit_pauli_pool(N)
pool, v_pool = chem_gate_pool(N)
H=lih['Hs']
# Xiphos 
x = Xiphos(H=H, ref=ref, system=SYSTEM_TAG, pool=pool, v_pool=v_pool,
           H_adapt=H, H_vqe=H, sym_ops=None)
# 跑 breadapt（不改库），捕获日志→CSV→曲线
err, rows, raw_log = run_breadapt_with_logging(
    x, system_tag=SYSTEM_TAG,
    params=ADAPT_KW["params"], ansatz=ADAPT_KW["ansatz"], ref=ADAPT_KW["ref"],
    Etol=ADAPT_KW["Etol"], gtol=ADAPT_KW["gtol"], max_depth=ADAPT_KW["max_depth"],
    guesses=ADAPT_KW["guesses"], hf=ADAPT_KW["hf"], n=ADAPT_KW["n"],
    threads=ADAPT_KW["threads"], seed=ADAPT_KW["seed"], criteria=ADAPT_KW["criteria"]
)

# -----------------------
# 1) 从 breadapt rows 中取出能量曲线
# -----------------------
df_bre = pd.DataFrame(rows)
df_bre = df_bre.sort_values("iter")
steps = len(df_bre)



psi0 = _to_dense_ket(ref)
E_qgrad, psi_fin, used_lr = qgrad_descent_energy(H, psi0, steps=steps, lr=None)  # lr=None => 自适应

# -----------------------
# 3) 画图
# -----------------------
plt.figure(figsize=(8,6))
plt.plot(df_bre["iter"], df_bre["Cos"], "-", lw=2.2, label=f"Adapt (η≈{used_lr:.3g})")
plt.plot(np.arange(steps), E_qgrad, "--", lw=2.2, label=f"QGrad (η≈{used_lr:.3g})")
plt.xlabel("Iteration")
plt.ylabel("Energy  ⟨H⟩  (Hartree)")
plt.title("LiH (6 qubits): Energy vs Iteration — b-ADAPT vs Quantum Gradient")
plt.legend()
plt.tight_layout()
plt.show()

# （可选）打印两者最终能量
print(f"[b-ADAPT] final E = {df_bre['Cos'].iloc[-1]:.12f}")
print(f"[QGrad]   final E = {E_qgrad[-1]:.12f}")



##########################################################################H2O

# ref → ket（1D 复向量）
N=8
# A) zero state
# ref = zero_state(N)
# # B) hartree fock state 
# _ref = jw_hartree_fock_state(n_e_active_lih,N)
# ref=csc_matrix(_ref).T.conj()
# equal weights state
ref = hadamard_layer_state(N, start='zero')

ADAPT_KW["ref"] = ref
# Pauli 生成元池（1-、2-比特全体）
# pool, v_pool = one_two_qubit_pauli_pool(N)
pool, v_pool = chem_gate_pool(N)
H=h2o['Hs']
# Xiphos 
x = Xiphos(H=H, ref=ref, system=SYSTEM_TAG, pool=pool, v_pool=v_pool,
           H_adapt=H, H_vqe=H, sym_ops=None)
# 跑 breadapt（不改库），捕获日志→CSV→曲线
err, rows, raw_log = run_breadapt_with_logging(
    x, system_tag=SYSTEM_TAG,
    params=ADAPT_KW["params"], ansatz=ADAPT_KW["ansatz"], ref=ADAPT_KW["ref"],
    Etol=ADAPT_KW["Etol"], gtol=ADAPT_KW["gtol"], max_depth=ADAPT_KW["max_depth"],
    guesses=ADAPT_KW["guesses"], hf=ADAPT_KW["hf"], n=ADAPT_KW["n"],
    threads=ADAPT_KW["threads"], seed=ADAPT_KW["seed"], criteria=ADAPT_KW["criteria"]
)

# -----------------------
# 1) 从 breadapt rows 中取出能量曲线
# -----------------------
df_bre = pd.DataFrame(rows)
df_bre = df_bre.sort_values("iter")
steps = len(df_bre)



psi0 = _to_dense_ket(ref)
E_qgrad, psi_fin, used_lr = qgrad_descent_energy(H, psi0, steps=steps, lr=None)  # lr=None => 自适应

# -----------------------
# 3) 画图
# -----------------------
plt.figure(figsize=(8,6))
plt.plot(df_bre["iter"], df_bre["Cos"], "-", lw=2.2, label=f"Adapt (η≈{used_lr:.3g})")
plt.plot(np.arange(steps), E_qgrad, "--", lw=2.2, label=f"QGrad (η≈{used_lr:.3g})")
plt.xlabel("Iteration")
plt.ylabel("Energy  ⟨H⟩  (Hartree)")
plt.title("H2O (8 qubits): Energy vs Iteration — b-ADAPT vs Quantum Gradient")
plt.legend()
plt.tight_layout()
plt.show()

# （可选）打印两者最终能量
print(f"[b-ADAPT] final E = {df_bre['Cos'].iloc[-1]:.12f}")
print(f"[QGrad]   final E = {E_qgrad[-1]:.12f}")






# %%
# ===================== 帮助函数：子池选择、QGrad ======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import gymnasium as gym
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import qas_gym
import warnings
import random
from qas_gym.utils import fidelity

def subset_pool_by_stride(pool, v_pool, stride):
    """按步长挑选生成元：stride=3≈1/3池，2≈1/2池，1=全池。"""
    idxs = list(range(0, len(pool), stride))
    sub_pool  = [pool[i] for i in idxs]
    sub_vpool = [v_pool[i] for i in idxs]
    if len(sub_pool) == 0:
        raise ValueError(f"subset_pool_by_stride: stride={stride} 结果为空（pool 太短？）")
    return sub_pool, sub_vpool, idxs

def subset_pool_by_sample(pool, v_pool, stride):
    """按ratio挑选生成元：stride=3≈1/3池，2≈1/2池，1=全池。"""
    idxs = random.sample(range(len(pool)), int(len(pool)/stride))
    sub_pool  = [pool[i] for i in idxs]
    sub_vpool = [v_pool[i] for i in idxs]
    if len(sub_pool) == 0:
        raise ValueError(f"subset_pool_by_sample: ratio=1/{stride} 结果为空（pool 太短？）")
    return sub_pool, sub_vpool, idxs

def _to_dense_ket(vec):
    """把 ref（可能是稀疏列向量/数组）转 1D 归一化 ket。"""
    if sp.issparse(vec):
        arr = vec.toarray().reshape(-1)
    else:
        arr = np.asarray(vec).reshape(-1)
    nrm = np.linalg.norm(arr)
    if nrm == 0:
        raise ValueError("Reference state has zero norm.")
    return (arr / nrm).astype(np.complex128)

def _matvec(H, psi):
    return H.dot(psi) if sp.issparse(H) else (H @ psi)

def _estimate_op_norm_power(H, n_dim, iters=30, seed=1):
    """幂迭代估 ||H||_2（H 厄米时近似谱半径）。"""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n_dim) + 1j * rng.normal(size=n_dim)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w = _matvec(H, v)
        nrm = np.linalg.norm(w)
        if nrm == 0:
            return 1.0
        v = w / nrm
    rq = np.vdot(v, _matvec(H, v)).real
    return max(abs(rq), 1.0)

def qgrad_descent_energy(H, ref_ket, steps, lr=None, norm_iters=30):
    """虚时间步进：psi<-psi - lr*Hpsi（每步归一化），返回能量序列。"""
    d = ref_ket.size
    psi = ref_ket.astype(np.complex128, copy=True)
    if lr is None:
        Hn = _estimate_op_norm_power(H, d, iters=norm_iters)
        lr = 0.1 / Hn
        print(f"[QGrad] Using lr = 0.1/||H|| ≈ {lr:.6g} (||H||≈{Hn:.6g})")
 
    Es = []
    for _ in range(steps):
        Hpsi = _matvec(H, psi)
        E = float(np.real(np.vdot(psi, Hpsi)))
        Es.append(E)
        psi = psi - lr * Hpsi
        nrm = np.linalg.norm(psi); psi = psi / (nrm if nrm > 0 else 1.0)
  
    return np.array(Es, dtype=float), psi, lr

def _df_from_rows(rows):
    """将 run_breadapt_with_logging 的 rows 转成 DataFrame 并提取 (iter, Cos/E)。"""
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return pd.DataFrame(columns=["iter","y"])
    # 统一小写列名
    df.columns = [str(c).strip() for c in df.columns]
    # 选择迭代列
    iter_col = "iter" if "iter" in df.columns else None
    if iter_col is None:
        df["iter"] = np.arange(len(df))
        iter_col = "iter"
    # 选择纵轴列（优先 Cos，其次 E/energy）
    y_col = None
    for c in ["Cos","E","energy","e","E_total","<h>"]:
        if c in df.columns:
            y_col = c; break
    if y_col is None:
        # 兜底选第一列数值
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols: y_col = num_cols[0]
        else: y_col = iter_col
    return df[[iter_col, y_col]].rename(columns={iter_col:"iter", y_col:"y"}).sort_values("iter")


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
    return max_fstate, max_fidelity, ppo_model.policy.state_dict(), max_obs


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



def qgrad_descent_with_PQC(H, ref_ket, steps, lr=None, norm_iters=30, pool='default'):
    """虚时间步进：psi<-psi - lr*Hpsi（每步归一化），返回能量序列。"""
    d = ref_ket.size
    psi = ref_ket.astype(np.complex128, copy=True)
    if lr is None:
        Hn = _estimate_op_norm_power(H, d, iters=norm_iters)
        lr = 0.1 / Hn
        print(f"[QGrad] Using lr = 0.1/||H|| ≈ {lr:.6g} (||H||≈{Hn:.6g})")
 
    Es = []
    oneminusf = []
    state_dict = None
    ind = False
    for i in range(steps):
        print("Iter {}: ".format(i))
        psi_init = psi
        Hpsi = _matvec(H, psi)
        E = float(np.real(np.vdot(psi, Hpsi)))
        Es.append(E)

        psi = psi - lr * Hpsi
        nrm = np.linalg.norm(psi); psi = psi / (nrm if nrm > 0 else 1.0)
        _f = fidelity(psi, psi_init)
        print("Initial fidelity : {:.4f}: ".format(_f))
        oneminusf.append(1-_f)
        # 动态调整fidelity_threshold
        ths = max(1-(1-_f)/3., 0.999)
        # print(f"fidelity_threshold: {ths}")
        # psi, fidelity_j, state_dict, obs = get_PQC_state(psi_init, psi, max_depth=20, fidelity_threshold=ths,train_steps=20000, pool=pool, observable='OpenFermion',mol=lih)
        
        # Strategy 1: Warm-start, for every 5 steps re-initialize parameters
        if i % 5 == 0:
            psi, fidelity_j, state_dict, obs = get_PQC_state(psi_init, psi, max_depth=20, fidelity_threshold=ths,train_steps=10000, pool='PauliAll2', observable='OpenFermion',mol=lih) # state_dict=state_dict
        else:
            psi, fidelity_j, state_dict, obs = get_PQC_state(psi_init, psi, max_depth=20, fidelity_threshold=ths,train_steps=2000, pool='PauliAll2', state_dict=state_dict, observable='OpenFermion',mol=lih) # state_dict=state_dict 
        
        # # Strategy 2: If fidelity doesn't improve in last 5 steps, switch to net pool and/or observables
        # if ind:
        #     ind = False
        #     print("Switch to OpenFermion Observable") #Pauli2
        #     psi, fidelity_j, _, obs = get_PQC_state(psi_init, psi, max_depth=20, fidelity_threshold=ths,train_steps=20000, pool='PauliAll2', observable='OpenFermion',mol=lih) # state_dict=state_dict
        # else:
        #     psi, fidelity_j, state_dict, obs = get_PQC_state(psi_init, psi, max_depth=10, fidelity_threshold=ths,train_steps=10000, state_dict=state_dict) 

        # E = float(np.real(np.vdot(psi, _matvec(H, psi))))
        # if i>=5 and not ind: 
        #     avgElast5 =  np.mean(Es[-5:])
        #     ind = (E/avgElast5-1<0.001)
        print(" fidelity after PQC : {:.4f}".format(fidelity_j))


    return np.array(Es, dtype=float), psi, lr, oneminusf




# %%
# ADAPT/breadapt params
ADAPT_KW = dict(
    params=[], ansatz=[], ref=None,  # ref will be filled later
    Etol=1e-8, gtol=None, max_depth=150,
    guesses=0, hf=False, n=1, threads=1, seed=123,
    criteria="grad"
)


# # ===================== 主程序：三子池 + QGrad 对比 ======================
# # -------- LiH --------
# N = 6
# # A) HF 参考态
# # _ref = jw_hartree_fock_state(n_e_active_lih, N)
# # ref  = csc_matrix(_ref).T.conj()
# ref= hadamard_layer_state(N, start='zero')
# ADAPT_KW["ref"] = ref

# # chem gate 池
# pool_full, vpool_full = one_two_qubit_pauli_pool(N)
# H = lih["Hs"]

# # 三种子池（1/3, 1/2, 全）
# pool_cfgs = [
#     ("ADAPT, pool 1/3 (stride=3)", 3),
#     ("ADAPT, pool 1/2 (stride=2)", 2),
#     ("ADAPT, full pool", 1),
# ]

# results = []   # [(label, df_bre), ...]
# steps_list = []

# for label, stride in pool_cfgs:
#     sub_pool, sub_vpool, idxs = subset_pool_by_stride(pool_full, vpool_full, stride)
#     print(len(sub_pool))
#     tag = f"{SYSTEM_TAG}_LiH_s{stride}"
#     # 构建 Xiphos（仅替换 pool / v_pool）
#     x_sub = Xiphos(H=H, ref=ref, system=tag, pool=sub_pool, v_pool=sub_vpool,
#                    H_adapt=H, H_vqe=H, sym_ops=None)
#     # 跑 ADAPT
#     err, rows, raw_log = run_breadapt_with_logging(
#         x_sub, system_tag=tag,
#         params=ADAPT_KW["params"], ansatz=ADAPT_KW["ansatz"], ref=ADAPT_KW["ref"],
#         Etol=ADAPT_KW["Etol"],   gtol=ADAPT_KW["gtol"],   max_depth=ADAPT_KW["max_depth"],
#         guesses=ADAPT_KW["guesses"], hf=ADAPT_KW["hf"],   n=ADAPT_KW["n"],
#         threads=ADAPT_KW["threads"], seed=ADAPT_KW["seed"], criteria=ADAPT_KW["criteria"]
#     )
#     df_bre = _df_from_rows(rows)
#     results.append((label, df_bre))
#     steps_list.append(len(df_bre))

# # QGrad：用三次 ADAPT 的最大步数
# steps_q = max(steps_list) if steps_list else ADAPT_KW.get("max_depth", 20)
# lr = 1
# psi0 = _to_dense_ket(ref)
# E_qgrad, psi_fin, used_lr = qgrad_descent_energy(H, psi0, steps=steps_q, lr=lr)
# E_qgrad_pqc, psi_fin_pqc, used_lr_pqc, oneminusf = qgrad_descent_with_PQC(H, psi0, steps=steps_q, lr=lr)

# 画图：三条 ADAPT（实线不同色）+ 一条 QGrad（黑色虚线）
plt.figure(figsize=(8,6))
palette = plt.cm.tab10(np.linspace(0, 1, len(results)))
for (i, (label, df_bre)) in enumerate(results):
    if not df_bre.empty:
        plt.plot(df_bre["iter"], df_bre["y"], "-", lw=2.0, color=palette[i], label=label)
plt.plot(np.arange(steps_q), E_qgrad, "--", lw=2.2, color="k", label=f"QGrad (η≈{used_lr:.3g})")
plt.plot(np.arange(steps_q), E_qgrad_pqc, "--", lw=2.2, color="g", label=f"QGrad with PQC(η≈{used_lr:.3g})")
plt.xlabel("Iteration")
plt.ylabel("Cost / Energy")
plt.title("LiH (6 qubits): Hadamard state initialization")
plt.legend()
plt.tight_layout()
plt.savefig(f"example_LiH_HM_empty_PQC_lr{lr}_256_128_PauliAll2_OpenFermion_150_5try_S1_2.pdf", format='pdf', bbox_inches='tight')

# （可选）打印最终点
for (label, df_bre) in results:
    if not df_bre.empty:
        print(f"[{label}] final y = {df_bre['y'].iloc[-1]:.12f}")
print(f"[QGrad] final E = {E_qgrad[-1]:.12f}")
print(f"[QGrad with PQC] final E = {E_qgrad_pqc[-1]:.12f}")

# # -------- H2O --------
# N = 8
# # B) 均幅初态（按你给的示例）
# ref = hadamard_layer_state(N, start='zero')
# ADAPT_KW["ref"] = ref
# pool_full, vpool_full = chem_gate_pool(N)
# H = h2o["Hs"]

# results = []
# steps_list = []

# for label, stride in pool_cfgs:
#     sub_pool, sub_vpool, idxs = subset_pool_by_stride(pool_full, vpool_full, stride)
#     tag = f"{SYSTEM_TAG}_H2O_s{stride}"
#     x_sub = Xiphos(H=H, ref=ref, system=tag, pool=sub_pool, v_pool=sub_vpool,
#                    H_adapt=H, H_vqe=H, sym_ops=None)
#     err, rows, raw_log = run_breadapt_with_logging(
#         x_sub, system_tag=tag,
#         params=ADAPT_KW["params"], ansatz=ADAPT_KW["ansatz"], ref=ADAPT_KW["ref"],
#         Etol=ADAPT_KW["Etol"],   gtol=ADAPT_KW["gtol"],   max_depth=ADAPT_KW["max_depth"],
#         guesses=ADAPT_KW["guesses"], hf=ADAPT_KW["hf"],   n=ADAPT_KW["n"],
#         threads=ADAPT_KW["threads"], seed=ADAPT_KW["seed"], criteria=ADAPT_KW["criteria"]
#     )
#     df_bre = _df_from_rows(rows)
#     results.append((label, df_bre))
#     steps_list.append(len(df_bre))

# steps_q = max(steps_list) if steps_list else ADAPT_KW.get("max_depth", 20)
# psi0 = _to_dense_ket(ref)
# E_qgrad, psi_fin, used_lr = qgrad_descent_energy(H, psi0, steps=steps_q, lr=10)
# E_qgrad2, psi_fin2, used_lr2 = qgrad_descent_with_PQC(H, psi0, steps=steps_q, lr=10)

# plt.figure(figsize=(8,6))
# palette = plt.cm.tab10(np.linspace(0, 1, len(results)))
# for (i, (label, df_bre)) in enumerate(results):
#     if not df_bre.empty:
#         plt.plot(df_bre["iter"], df_bre["y"], "-", lw=2.0, color=palette[i], label=label)
# plt.plot(np.arange(steps_q), E_qgrad, "--", lw=2.2, color="k", label=f"QGrad (η≈{used_lr:.3g})")
# plt.plot(np.arange(steps_q), E_qgrad2, "--", lw=2.2, color="g", label=f"QGrad with PQC(η≈{used_lr:.3g})")
# plt.xlabel("Iteration")
# plt.ylabel("Cost / Energy")
# plt.title("H2O (8 qubits): ADAPT with different pools vs Quantum Gradient")
# plt.legend()
# plt.tight_layout()
# plt.show()

# for (label, df_bre) in results:
#     if not df_bre.empty:
#         print(f"[{label}] final y = {df_bre['y'].iloc[-1]:.12f}")
# print(f"[QGrad] final E = {E_qgrad[-1]:.12f}")

# plt.savefig(f"example_H2O_HF_empty_PQC_lr{lr}_tmpSwitch.pdf", format='pdf', bbox_inches='tight')

import dill
dill.load_session('result/example_LiH_HM_empty_PQC_lr{lr}_256_128_PauliAll2_OpenFermion_150.db')


