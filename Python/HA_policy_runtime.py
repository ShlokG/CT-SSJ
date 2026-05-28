# %%
"""Policy-function runtime decomposition.
The timed objects use IPython's `%timeit -o` machinery through `run_line_magic`,
matching the runtime protocol in `HA_Runtime.py` while still allowing `REPEAT`
and `NUMBER` to be ordinary Python variables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import HA_DT as ha
from Model import get_parameters, get_ss, step_0
from Jacobian_Helpers import calc_phi, calc_policy_fn
from Parameters import param_econ, param_num
from toolkit import aux_speed as aux_jac
from toolkit import utils


# %%
# User choices
NZ = 50
NA = 5000
MU = 1.0
T = 300
H = 1e-4

# Match HA_Runtime.py's repeated `%timeit` structure.  Use NUMBER=100 final.
REPEAT = 7
NUMBER = 10

CACHE_DIR = Path("Storage")
FIG_DIR = Path("Figures")
FIG_DIR.mkdir(exist_ok=True)

# %%
# Small helper: IPython timeit with Python variables.
def ipy_timeit(stmt):
    return get_ipython().run_line_magic("timeit", f"-o -r {REPEAT} -n {NUMBER} {stmt}")


def average_result(result):
    return float(result.average)


def dt_step_fake_1_reference(
    back_step_fun,
    shock_dict,
    ss,
    ssinput_dict,
    ssy_list,
    outcome_list,
    v_name,
    a_pol_i,
    a_space,
    horizon,
):
    """HA_Runtime.py policy-function reference: includes curlyD and curlyY."""
    return aux_jac.step_fake_1(
        back_step_fun,
        shock_dict,
        ss,
        ssinput_dict,
        ssy_list,
        outcome_list,
        v_name,
        a_pol_i,
        a_space,
        horizon,
    )


def dt_forward_difference_wrapper_actual(
    back_step_fun,
    shock_dict,
    ssinput_dict,
    ssy_list,
    v_name,
    horizon,
    h,
):
    """All actual `utils.numerical_diff` calls in the step_fake_1 recursion.

    This mirrors the shock recursion in `aux_speed.backward_iteration`, but
    drops the distribution response (`curlyD`) and aggregation (`curlyY`) work
    after the forward-difference wrapper returns.
    """
    for shock in shock_dict.values():
        cur_shock = shock
        for _ in range(horizon):
            curly_v, _da, _dc = utils.numerical_diff(
                back_step_fun,
                ssinput_dict,
                cur_shock,
                h,
                ssy_list,
            )
            cur_shock = {v_name + "_p": curly_v}


def build_dt_component_blocks_by_price(back_step_fun, ssinput_dict, shock_dict, h):
    """Construct one contemporaneous shocked block per price for isolated timing."""
    blocks_by_price = {}
    for price, shock in shock_dict.items():
        shocked_inputs = {
            **ssinput_dict,
            **{k: ssinput_dict[k] + h * value for k, value in shock.items()},
        }
        beta = shocked_inputs["beta"]
        Pi_p = shocked_inputs["Pi_p"]
        Va_p = shocked_inputs["Va_p"]
        eis = shocked_inputs["eis"]
        a_grid = shocked_inputs["a_grid"]
        e_grid = shocked_inputs["e_grid"]
        r = shocked_inputs["r"]
        w = shocked_inputs["w"]

        uc_nextgrid = (beta * Pi_p) @ Va_p
        c_nextgrid = uc_nextgrid ** (-eis)
        coh = (1.0 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
        a = utils.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
        utils.setmin(a, a_grid[0])
        c = coh - a

        blocks_by_price[price] = {
            "inputs": shocked_inputs,
            "outputs": back_step_fun(**shocked_inputs),
            "c_nextgrid": c_nextgrid,
            "c": c,
        }
    return blocks_by_price


def dt_euler_inversion_by_price(blocks_by_price, horizon):
    """Nonlinear Euler calculations inside `HA_DT.backward_iterate`, T times per price.

    Corresponds to:
        uc_nextgrid = (beta * Pi_p) @ Va_p
        c_nextgrid = uc_nextgrid ** (-eis)
        Va = (1 + r) * c ** (-1 / eis)
    """
    for block in blocks_by_price.values():
        x = block["inputs"]
        beta = x["beta"]
        Pi_p = x["Pi_p"]
        Va_p = x["Va_p"]
        eis = x["eis"]
        r = x["r"]
        c = block["c"]
        for _ in range(horizon):
            uc_nextgrid = (beta * Pi_p) @ Va_p
            c_nextgrid = uc_nextgrid ** (-eis)
            Va = (1.0 + r) * c ** (-1.0 / eis)
    return Va


def dt_interpolation_by_price(blocks_by_price, horizon):
    """Interpolation and remaining policy-grid mapping lines, T times per price.

    This includes cash-on-hand, endogenous-grid interpolation, the borrowing
    constraint, and c = coh - a. It excludes the nonlinear Euler/Va powers.
    """
    for block in blocks_by_price.values():
        x = block["inputs"]
        a_grid = x["a_grid"]
        e_grid = x["e_grid"]
        r = x["r"]
        w = x["w"]
        c_nextgrid = block["c_nextgrid"]

        for _ in range(horizon):
            coh = (1.0 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
            a = utils.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
            utils.setmin(a, a_grid[0])
            c = coh - a
    return c


def dt_finite_difference_algebra_by_price(blocks_by_price, ssy_list, horizon, h):
    """Subtraction/division after `backward_iterate`, T times per price."""
    for block in blocks_by_price.values():
        y_list = block["outputs"]
        for _ in range(horizon):
            dy_list = [(y - y_ss) / h for y, y_ss in zip(y_list, ssy_list)]
    return dy_list

def ct_calc_phi_all_prices(ss, n, store, horizon):
    """Run all calc_phi calls over prices."""
    phi_by_price = {}
    for pr in store["prices"]:
        phi_by_price[pr] = calc_phi(
            ss,
            n,
            horizon,
            dt=1.0,
            phi0=store["inc_chg"][pr] * ss["up"],
            L_p=store["L_p"][pr],
            Vss=store["Vss"],
            u_p=store["u_p"],
            C_p=store["C_p"][pr],
            C_V=store["C_V"],
            C_dV=store["C_dV"],
            price=True,
            iter_style="DT_loop",
        )
    return phi_by_price


def ct_dphi_da_all_prices(ss, store, phi_by_price):
    """Construct dphi_da for all prices, including the time-0 adjustment."""
    da_up = ss["DA"] @ ss["up"].flatten()
    dphi_da_by_price = {}
    for pr in store["prices"]:
        dphi_da = ss["DA"] @ phi_by_price[pr]
        dphi_da[:, 0] = da_up * store["inc_chg"][pr].flatten()
        dphi_da_by_price[pr] = dphi_da
    return dphi_da_by_price


def ct_c_pr_all_prices(p, n, ss, store, dphi_da_by_price, horizon):
    """Run calc_policy_fn for all prices and apply the time-0 c_pr adjustment."""
    c_pr_by_price = {}
    for pr in store["prices"]:
        c_pr = calc_policy_fn(
            store["U"],
            horizon,
            p["rho"],
            ss["gm"],
            dt=1.0,
            u_cp=store["u_cp"],
            u_cz=store["u_cz"],
            dphi_da=dphi_da_by_price[pr],
            L_c=None,
            phi_t=None,
        )
        if pr == "r":
            c_prime = ss["DA"] @ ss["c"].ravel()
            c_pr[:, 0] = (n["aa"] * ss["gm"]).ravel() * c_prime
        c_pr_by_price[pr] = c_pr
    return c_pr_by_price

def ct_marginal_value_all_prices(ss, n, store, horizon):
    """Run calc_phi and dphi_da construction for all prices."""
    phi_by_price = ct_calc_phi_all_prices(ss, n, store, horizon)
    return ct_dphi_da_all_prices(ss, store, phi_by_price)


# %%
# Discrete-time setup, not timed.  These are the exact objects passed to
# `aux_jac.step_fake_1` in HA_Runtime.py.
p_dt = param_econ()
n_dt = param_num(p_dt, Nz=NZ, Na=NA, mu=MU)
dt_ss = ha.ha_ss_r(Nz = NZ, Na = NA, eis = 1/p_dt['gamma'], delta = p_dt['d'],
                           alpha = p_dt['alpha'], rho = p_dt['rho_e'], sigma = p_dt['sigma_e'],
                           lb = n_dt['rmin'], ub = n_dt['rmax'], beta = np.exp(-p_dt['rho']),
                           maxiter = n_dt['Ir'], xtol = n_dt['crit_S'], back_tol = n_dt['back_tol'], amax = n_dt['amax'],
                           fwd_tol = n_dt['fwd_tol'], back_maxit = n_dt['back_maxit'], fwd_maxit = n_dt['fwd_maxit'])

dt_shock_dict = {"r": {"r": 1}, "w": {"w": 1}}
(
    dt_ssinput_dict,
    dt_ssy_list,
    dt_outcome_list,
    dt_v_name,
    dt_a_pol_i,
    dt_a_pol_pi,
    dt_a_space,
) = aux_jac.step_fake_0(ha.backward_iterate, dt_ss)

# One contemporaneous shocked input/output per price.  Component kernels below
# run T repetitions for each price rather than using one representative price
# shock repeated 2*T times.
dt_blocks_by_price = build_dt_component_blocks_by_price(
    ha.backward_iterate,
    dt_ssinput_dict,
    dt_shock_dict,
    H,
)


# Warm up numba-compiled kernels and allocate once before timing.
_ = dt_step_fake_1_reference(
    ha.backward_iterate,
    dt_shock_dict,
    dt_ss,
    dt_ssinput_dict,
    dt_ssy_list,
    dt_outcome_list,
    dt_v_name,
    dt_a_pol_i,
    dt_a_space,
    T,
)
_ = dt_forward_difference_wrapper_actual(
    ha.backward_iterate,
    dt_shock_dict,
    dt_ssinput_dict,
    dt_ssy_list,
    dt_v_name,
    T,
    H,
)
_ = dt_interpolation_by_price(dt_blocks_by_price, T)
_ = dt_euler_inversion_by_price(dt_blocks_by_price, T)
_ = dt_finite_difference_algebra_by_price(dt_blocks_by_price, dt_ssy_list, T, H)


# %%
# Discrete-time timings.
dt_interpolation = ipy_timeit("dt_interpolation_by_price(dt_blocks_by_price, T)")
dt_euler = ipy_timeit("dt_euler_inversion_by_price(dt_blocks_by_price, T)")
dt_finite_diff = ipy_timeit(
    "dt_finite_difference_algebra_by_price(dt_blocks_by_price, dt_ssy_list, T, H)"
)


# %%
# Continuous-time setup, not timed.  These are the objects used by
# `policy_function` inside `run_ha` in HA_Runtime.py.
p_ct, n_ct = get_parameters(Nz = NZ, Na = NA, mu = MU)    # loads parameters and grids
ss_ct      = get_ss(p_ct, n_ct, EGM = True, HANK = False) # gets steady state
ss_ct, store_ct = step_0(p_ct, n_ct, ss_ct, EGM=True, dt=1.0)

# Warm up and initialize phi/dphi state for downstream timed functions.
ct_phi_for_dphi = ct_calc_phi_all_prices(ss_ct, n_ct, store_ct, T)
ct_dphi_for_c = ct_dphi_da_all_prices(ss_ct, store_ct, ct_phi_for_dphi)
_ = ct_c_pr_all_prices(p_ct, n_ct, ss_ct, store_ct, ct_dphi_for_c, T)
_ = ct_marginal_value_all_prices(ss_ct, n_ct, store_ct, T)


# %%
# Continuous-time timings.
ct_marginal_value = ipy_timeit("ct_marginal_value_all_prices(ss_ct, n_ct, store_ct, T)")

ct_phi_for_dphi = ct_calc_phi_all_prices(ss_ct, n_ct, store_ct, T)
ct_dphi_for_c = ct_dphi_da_all_prices(ss_ct, store_ct, ct_phi_for_dphi)
ct_c_pr = ipy_timeit("ct_c_pr_all_prices(p_ct, n_ct, ss_ct, store_ct, ct_dphi_for_c, T)")

# %%
# Summaries.
dt_rows = [
    ("Interpolation", average_result(dt_interpolation)),
    ("Euler inversion", average_result(dt_euler)),
    ("Finite difference", average_result(dt_finite_diff)),
]

ct_rows = [
    ("Marginal Value", average_result(ct_marginal_value)),
    ("Consumption", average_result(ct_c_pr)),
]

summary_rows = []
for component, seconds in dt_rows:
    summary_rows.append(
        {
            "Type": "Discrete",
            "Nz": NZ,
            "Na": NA,
            "Gridpoints": NZ * NA,
            "Repeat": REPEAT,
            "Number": NUMBER,
            "Component": component,
            "Seconds": seconds,
        }
    )

for component, seconds in ct_rows:
    summary_rows.append(
        {
            "Type": "Continuous, EGM",
            "Nz": NZ,
            "Na": NA,
            "Gridpoints": NZ * NA,
            "Repeat": REPEAT,
            "Number": NUMBER,
            "Component": component,
            "Seconds": seconds,
        }
    )

summary = pd.DataFrame(summary_rows)

# %%
# Save table and a compact component figure.
out_base = FIG_DIR / f"HA_policy_function_only_runtime_decomp_interactive_{NZ * NA}_r{REPEAT}n{NUMBER}"
summary_path = out_base.with_suffix(".csv")
summary.to_csv(summary_path, index=False)

dt_plot = {
    "Interpolation": average_result(dt_interpolation),
    "Euler inversion": average_result(dt_euler),
    "Finite difference": average_result(dt_finite_diff),
}
ct_plot = {
    "Marginal Value": average_result(ct_marginal_value),
    "Consumption": average_result(ct_c_pr),
}

dt_colors = ["#0B3D5C", "#2A6F97", "#8FC5E8"]
ct_colors = ["#5C0B0B", "#E8A090"]
fig, ax = plt.subplots(figsize=(9.5, 6.8))
for x, parts, color_list in [(0, dt_plot, dt_colors), (1, ct_plot, ct_colors)]:
    bottom = 0.0
    for (label, value), color in zip(parts.items(), color_list):
        ax.bar(x, value, width=0.62, bottom=bottom, color=color, edgecolor="white", linewidth=0.9)
        bottom += value

ax.set_xticks([0, 1], ["Discrete", "Continuous"])
ax.set_ylabel("Seconds", fontsize=20)
ax.tick_params(axis="both", labelsize=18)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, max(sum(dt_plot.values()), sum(ct_plot.values())) * 1.20)

dt_handles = [Patch(facecolor=c, label=l) for c, l in zip(dt_colors, dt_plot.keys())]
ct_handles = [Patch(facecolor=c, label=l) for c, l in zip(ct_colors, ct_plot.keys())]
fig.tight_layout()
leg1 = ax.legend(handles=dt_handles, title="Discrete", loc="upper right",
                 bbox_to_anchor=(0.98, 0.98), fontsize=14, title_fontsize=16,
                 frameon=False)
leg1._legend_box.align = "left"
ax.add_artist(leg1)
fig.canvas.draw()
leg1_left_disp = leg1.get_window_extent().x0
leg1_left_axes = ax.transAxes.inverted().transform((leg1_left_disp, 0))[0]
leg2 = ax.legend(handles=ct_handles, title="Continuous", loc="upper left",
                 bbox_to_anchor=(leg1_left_axes, 0.70), fontsize=14, title_fontsize=16,
                 frameon=False)
leg2._legend_box.align = "left"

pdf_path = FIG_DIR / "HA_policy_function_only_runtime_decomp.pdf"
fig.savefig(pdf_path)
plt.show()
