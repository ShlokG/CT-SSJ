"""Create a report for the Exact-Pi mixed 3%/10% HA runtime experiment.

This script is intentionally separate from HA_All.py, HA_Runtime.py, and the
model code.  It reads existing runtime CSVs, regenerates the 250k IRFs needed
for the report, and writes a standalone LaTeX document plus PDF figures.
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")

import numpy as np
from numba import njit

def make_time_grid(T_horizon, dt_min=0.25, dt_max=2.0, growth_rate=1.1, method='geometric', breakpoints=None):
    """Construct a non-uniform time grid covering [0, T_horizon].

    Args:
        T_horizon: Total time horizon (e.g., 300 years)
        dt_min: Minimum (initial) time step
        dt_max: Maximum time step
        growth_rate: Growth factor per step (geometric method only)
        method: 'geometric' or 'piecewise'
        breakpoints: List of (end_time, dt) tuples for piecewise method,
            e.g., [(10, 0.25), (100, 1.0), (300, 2.0)]

    Returns:
        t_vec: Array of time points (length T)
        dt_vec: Array of step sizes (length T-1)
    """
    if method == 'geometric':
        dt_list = []
        t = 0.0
        k = 0
        while t < T_horizon:
            dt_k = min(dt_min * growth_rate ** k, dt_max)
            # Don't overshoot the horizon
            dt_k = min(dt_k, T_horizon - t)
            dt_list.append(dt_k)
            t += dt_k
            k += 1
        dt_vec = np.array(dt_list)
        t_vec = np.concatenate(([0.0], np.cumsum(dt_vec)))

    elif method == 'piecewise':
        if breakpoints is None:
            raise ValueError("breakpoints must be provided for piecewise method")
        dt_list = []
        t = 0.0
        for end_time, dt_seg in breakpoints:
            while t < end_time - 1e-12:
                dt_k = min(dt_seg, end_time - t)
                dt_list.append(dt_k)
                t += dt_k
        dt_vec = np.array(dt_list)
        t_vec = np.concatenate(([0.0], np.cumsum(dt_vec)))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'geometric' or 'piecewise'.")

    return t_vec, dt_vec

def eig_pi_builder(ly: np.ndarray):
    vals, vecs = np.linalg.eig(ly)
    inv_vecs = np.linalg.inv(vecs)

    def build(dt: float) -> np.ndarray:
        Pi = (vecs @ (np.exp(float(dt) * vals)[:, None] * inv_vecs)).real
        return np.ascontiguousarray(Pi)

    return build

def eig_pi_ops(n: dict, dt_vec: np.ndarray) -> dict:
    build = eig_pi_builder(n["ly"])
    return {float(dt): build(float(dt)) for dt in np.unique(dt_vec)}

def exp_grid(horizon: float, dt0: float, growth: float, cap: float | None) -> tuple[np.ndarray, np.ndarray]:
    if cap is None:
        dt_list = []
        t = 0.0
        k = 0
        while t < horizon - 1e-12:
            dt_k = dt0 * growth**k
            dt_k = min(dt_k, horizon - t)
            dt_list.append(dt_k)
            t += dt_k
            k += 1
        dt_vec = np.array(dt_list, dtype=float)
        return np.concatenate(([0.0], np.cumsum(dt_vec))), dt_vec

    return make_time_grid(horizon, dt_min=dt0, dt_max=cap, growth_rate=growth, method="geometric")

@njit
def asset_explicit_step(D: np.ndarray, x_i: np.ndarray, S_npy: np.ndarray, dt: float) -> np.ndarray:
    Nz, Na = D.shape
    out = np.empty_like(D)
    for iz in range(Nz):
        for ia in range(Na):
            flat_i = iz * Na + ia
            left = x_i[iz, ia]
            drift = (
                S_npy[flat_i, 0] * D[iz, left]
                + S_npy[flat_i, 1] * D[iz, left + 1]
                + S_npy[flat_i, 2] * D[iz, ia]
            )
            out[iz, ia] = D[iz, ia] + dt * drift
    return out

def hybrid_transition_fast(D: np.ndarray, Pi: np.ndarray, ss: dict, dt: float) -> np.ndarray:
    return asset_explicit_step(np.ascontiguousarray(Pi @ D), ss["a_ind"], ss["S_npy"], float(dt))

def initial_phi(n: dict, ss: dict, store: dict, PRICES = ("r", "w")) -> dict:
    phi = {}
    rows = ss["consted_ind"]
    da0 = float(n["da"][0])
    for pr in PRICES:
        phi_pr = (store["inc_chg"][pr] * ss["up"]).copy()
        if rows.size:
            flat = phi_pr.ravel()
            flat[rows] = flat[rows + 1] + da0 * store["C_p"][pr]
            phi_pr = flat.reshape(n["Nz"], n["Na"])
        phi[pr] = phi_pr
    return phi

def policy_function_hybrid_fast_anticipate(
    p: dict,
    n: dict,
    ss: dict,
    store: dict,
    t_vec: np.ndarray,
    dt_vec: np.ndarray,
    pi_ops: dict,
    anticipate: bool,
    PRICES = ("r", "w")
):
    """Exact-Pi explicit-savings policy step with Model.policy_function's toggle."""
    T = len(t_vec)
    N = n["Ntot"]
    phi = initial_phi(n, ss, store, PRICES = PRICES)
    phi_hist = {pr: np.empty((N, T)) for pr in PRICES}
    for pr in PRICES:
        phi_hist[pr][:, 0] = phi[pr].ravel()

    rows = ss["consted_ind"]
    for t, dt_t in enumerate(dt_vec):
        Pi = pi_ops[float(dt_t)]
        for pr in PRICES:
            phi[pr] = hybrid_transition_fast(phi[pr], Pi, ss, float(dt_t))
            if rows.size:
                flat = phi[pr].ravel()
                flat[rows] = flat[rows + 1]
                phi[pr] = flat.reshape(n["Nz"], n["Na"])
            phi_hist[pr][:, t + 1] = phi[pr].ravel()

    dphi_da = {pr: ss["DA"] @ phi_hist[pr] for pr in PRICES}
    if not anticipate:
        up_a = ss["DA"] @ ss["up"].ravel()
        for pr in PRICES:
            dphi_da[pr][:, 0] = up_a * store["inc_chg"][pr].ravel()

    gU_inv = (ss["gm"] / store["U"]).ravel()[:, np.newaxis]
    rho_T = np.exp(-p["rho"] * t_vec)[np.newaxis, :]
    DA_T = ss["DA_T"]
    gm = ss["gm"].ravel()

    c_t = {}
    D = {}
    for pr in PRICES:
        c_pr = dphi_da[pr] * gU_inv * rho_T
        if pr == "r" and not anticipate:
            c_prime = ss["DA"] @ ss["c"].ravel()
            c_pr[:, 0] = (n["aa"] * ss["gm"]).ravel() * c_prime
        D_pr = DA_T @ c_pr
        D_pr[:, 0] += -DA_T @ (store["inc_chg"][pr].ravel() * gm)
        c_t[pr] = np.sum(c_pr, axis=0)
        D[pr] = D_pr

    return dphi_da, c_t, D

def expectation_vector_hybrid_fast(n: dict, ss: dict, store: dict, dt_vec: np.ndarray, pi_ops: dict, OUTPUTS = ("K", "C")) -> dict:
    T = len(dt_vec) + 1
    E_cur = {out: store["E"][out].copy() for out in OUTPUTS}
    E_t = {out: np.empty((T, n["Ntot"])) for out in OUTPUTS}
    for out in OUTPUTS:
        E_t[out][0, :] = E_cur[out].ravel()

    for t, dt_t in enumerate(dt_vec):
        Pi = pi_ops[float(dt_t)]
        for out in OUTPUTS:
            E_cur[out] = hybrid_transition_fast(E_cur[out], Pi, ss, float(dt_t))
            E_t[out][t + 1, :] = E_cur[out].ravel()
    return E_t

def fast_inversion_ha(p, n, ss, store, J, z_hat, T):
    """Direct HA inversion without SimpleSparse objects.

    This matches Model.inversion for HA when prices are r,w and household
    output is K, but avoids constructing sparse-displacement wrappers.
    """
    lab = float(np.dot(ss["gm"].ravel(), n["zz"].ravel()))
    K = float(np.dot(ss["gm"].ravel(), n["aa"].ravel()))

    jrK = p["alpha"] * p["AgZ"] * (p["alpha"] - 1) * (K / lab) ** (p["alpha"] - 2) / lab
    jwK = (1 - p["alpha"]) * p["alpha"] * p["AgZ"] * (K / lab) ** (p["alpha"] - 1) / lab
    jrZ = store["zeta"]["r"]
    jwZ = store["zeta"]["w"]

    H_K = J["r"]["K"] * jrK + J["w"]["K"] * jwK - np.eye(T)
    H_Z = (J["r"]["K"] * jrZ + J["w"]["K"] * jwZ) @ z_hat
    dK = -np.linalg.solve(H_K, H_Z)
    dr = jrZ * z_hat + jrK * dK
    dw = jwZ * z_hat + jwK * dK
    return {"r": dr, "w": dw}

def continuous_outputs(n, ss, J, dr, dw):
    dK = J["r"]["K"] @ dr + J["w"]["K"] @ dw
    dC = J["r"]["C"] @ dr + J["w"]["C"] @ dw
    K_ss = float(np.sum(ss["gm"] * n["aa"]))
    C_ss = float(np.sum(ss["gm"] * ss["c"]))
    return dK, dC, K_ss, C_ss

def solve_outputs(p: dict, n: dict, ss: dict, store: dict, J: dict, t_vec: np.ndarray) -> dict:
    z_hat = 0.01 * np.exp(-p["rho_Z"] * t_vec)
    prices = fast_inversion_ha(p, n, ss, store, J, z_hat, len(t_vec))
    dr, dw = prices["r"], prices["w"]
    dK, dC, K_ss, C_ss = continuous_outputs(n, ss, J, dr, dw)
    return {
        "t": t_vec,
        "r": np.asarray(dr),
        "w": np.asarray(dw),
        "K": np.asarray(dK),
        "C": np.asarray(dC),
        "K_ss": K_ss,
        "C_ss": C_ss,
    }