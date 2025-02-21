# This file plots the Jacobian and IRFs for the HA model
from Model import * # import model-specific functions from general code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolkit import jacobian as jac      # discrete-time code to get GE jacobian
from toolkit import aux_speed as aux_jac # discrete-time code for each step
import HA_DT as ha               # discrete-time version of model

# User Inputs
dt  = 1.0              # time step
T   = int(300 / dt)    # number of time periods
Nz  = 25               # number of idiosyncratic productivity states
Na  = 2500             # number of asset gridpoints
EGM = True             # solve for steady state via endogenous gridpoint or implicit method
iter_style = 'DT_loop' # how to iterate forward in phi and E calculations
                       # 3 options: 'CT_matrix', 'CT_loop', 'DT_loop' where 'CT' uses ly and DT uses Pi and
                       # 'matrix' does a sparse matrix multiplication while 'loop' uses a for loop with numba

# Plotting Inputs
plot_T = 30 # Number of periods to plot for IRFs

# String to append to file names
str_append  = f"_Nz{Nz}_Na{Na}"
str_append += "_HJB" if not EGM else ""
str_append += f"_dt{dt}" if dt != 1.0 else ""

fig_dir = "Figures/HA_"

# share moving out of gridpt via savings
## can be 1 if using DT_loop; otherwise less than 1
mu = 1 if iter_style == 'DT_loop' else 0.5

# Get Jacobians in continuous time
p, n            = get_parameters(Nz = Nz, Na = Na, mu = mu) # loads parameters and grids
ss              = get_ss(p, n, EGM = EGM, HANK = False)     # gets steady state
ss, store       = step_0(p, n, ss, EGM = EGM, dt = dt)      # stores asset derivative, constrained agents, asset transition indices given savings rule
dphi_da, c_t, D = policy_function(p, n, ss, store, T, dt = dt, iter_style = iter_style)   # policy functions and distribution change from future shock
E_t             = expectation_vector(ss, store['E'], T, dt = dt, iter_style = iter_style) # expectation vector (for propagation of distribution)
F               = fake_news(store['prices'], store['outputs'], E_t, D, T, c_t)            # fake news operator
J               = jacobian(store['prices'], store['outputs'], F, dt = dt)                 # jacobian
# Get IRFs
z_hat           = 0.01 * np.exp(-p['rho_Z'] * np.linspace(0, (T - 1) * dt, T)) # 1% TFP shock
irfs            = inversion(p, n, ss, store, J, z_hat, T)                      # solve system to get prices and outputs

# Repeat but with anticipation of time-0 r shock
dphi_da2, c_t2, D2  = policy_function(p, n, ss, store, T, anticipate = True, dt = dt, iter_style = iter_style)
F2                  = fake_news(store['prices'], store['outputs'], E_t, D2, T, c_t2)
J2                  = jacobian(store['prices'], store['outputs'], F2, dt = dt)
irfs2               = inversion(p, n, ss, store, J2, z_hat, T)

# Get Jacobians in discrete time
## first, steady-state in discrete time
dt_ss = ha.ha_ss_r(Nz = Nz, Na = Na, eis = 1/p['gamma'], delta = p['d'],
                       alpha = p['alpha'], rho = p['rho_e'], sigma = p['sigma_e'],
                       lb = n['rmin'], ub = n['rmax'], beta = np.exp(-p['rho']),
                       maxiter = n['Ir'], xtol = n['crit_S'], back_tol = n['back_tol'], amax = n['amax'],
                       fwd_tol = n['fwd_tol'], back_maxit = n['back_maxit'], fwd_maxit = n['fwd_maxit'])

# jacobians in discrete time, step-by-step
shock_dict       = {'r': {'r': 1}, 'w': {'w': 1}}                                           # prices
ssinput_dict, ssy_list, outcome_list, V_name, a_pol_i, a_pol_pi, a_space = aux_jac.step_fake_0(ha.backward_iterate, dt_ss) # necessary objects for jacobian from steady state
curlyYs, curlyDs = aux_jac.step_fake_1(ha.backward_iterate, {'r': {'r': 1}, 'w': {'w': 1}}, dt_ss, ssinput_dict, ssy_list, outcome_list, V_name, a_pol_i, a_space, T) # change in distribution
curlyPs          = aux_jac.step_fake_2(dt_ss, outcome_list, ssy_list, a_pol_i, a_pol_pi, T) # expectation vector
curlyFs          = aux_jac.step_fake_3(shock_dict, outcome_list, curlyYs, curlyDs, curlyPs) # fake news matrix
curlyJs          = aux_jac.step_fake_4(curlyFs, shock_dict, outcome_list)                   # jacobian

# GE Jacobian in 1 line for impulse responses
G = jac.get_G(block_list = [ha.firm, ha.mkt_clearing, ha.household], exogenous = ['Z'], unknowns = ['K'],
              targets = ['asset_mkt'], T = T, ss = dt_ss)

# Plot Jacobians for Comparison
orig_cols       = [0, 100, 200] # time periods to plot
columns_to_plot = [int(orig_cols[i] / dt) for i in range(len(orig_cols))] # converted to columns when dt != 1
linestyles      = ['-', '--', '-.', ':', (0, (1, 10))]
default_colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_Jac(shock_name = 'K', price = 'r', fig_dir = 'Figures/', str_append = ''):
    if shock_name == 'K':
        DT_name = 'a'                        # name for capital in discrete-time jacobian
        CT_Val  = np.sum(ss['gm'] * n['aa']) # steady-state capital
    else:
        DT_name = 'c'                        # name for consumption in discrete-time jacobian
        CT_Val  = np.sum(ss['gm'] * ss['c']) # steady-state consumption

    DT_Jac = curlyJs[DT_name][price] / dt_ss[shock_name] # discrete-time jacobian in % deviation from steady state
    # discrete-time plots choice of assets each period instead of assets at start of period, so adjust to match
    if shock_name == 'K':
        top_row = np.zeros((1, DT_Jac.shape[1]))
        DT_Jac  = np.vstack([top_row, DT_Jac])

    for i, col in enumerate(columns_to_plot):
        plt.plot(np.arange(0, T * dt, dt), J[price][shock_name][:, col] / CT_Val,
                 label = f"Continuous (surprise)", linestyle = linestyles[0], linewidth = 3, color = default_colors[0])
        plt.plot(np.arange(0, T * dt, dt), J2[price][shock_name][:, col] / CT_Val,
                 label = f"Continuous (anticipation)", linestyle = linestyles[3], linewidth = 4, color = default_colors[0])
        plt.plot(range(int(T * dt)), DT_Jac[:int(T * dt), orig_cols[i]], label = f"Discrete",
                    linestyle = linestyles[1], linewidth = 3, color = default_colors[1])
        if col == 0 and shock_name == 'K' and price == 'w':
            plt.legend(framealpha=0)

    plt.xlabel("Year", fontsize=14)
    plt.ylabel("p.p. deviation from SS", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.savefig(fig_dir + "Jac_" + shock_name + price + str_append + ".pdf")
    plt.show()

plot_Jac(shock_name = 'K', price = 'r', str_append = str_append, fig_dir = fig_dir)
plot_Jac(shock_name = 'C', price = 'r', str_append = str_append, fig_dir = fig_dir)
plot_Jac(shock_name = 'K', price = 'w', str_append = str_append, fig_dir = fig_dir)
plot_Jac(shock_name = 'C', price = 'w', str_append = str_append, fig_dir = fig_dir)

# impulse responses of r, w, K, C to a Z (productivity) shock
DT_jac_rZ = G['r']['Z']
DT_jac_wZ = G['w']['Z']

z_hat_dt = 0.01 * np.exp([-p['rho_Z'] * t for t in range(T)]) # 1% TFP shock when dt = 1

# discrete-time impulse responses
DT_irf_r = DT_jac_rZ @ z_hat_dt
DT_irf_w = DT_jac_wZ @ z_hat_dt

# plot impulse responses of r and w to Z shock
plt.rc('legend', fontsize = 16)
plot_t = np.minimum(plot_T, int(T * dt))

def plot_IRF(DT_irf, CT_irf, CT_irf2, shock_name = "r", fig_dir = 'Figures/', str_append = ''):
    if shock_name == "K":
        DT_irf = np.insert(DT_irf, 0, 0)    # discrete-time plots choice of assets each period instead of assets at start of period, so adjust to match
        CT_val = np.sum(ss['gm'] * n['aa']) # steady-state capital
        DT_val = dt_ss['A']                 # steady-state capital in discrete time
    elif shock_name == "C":
        CT_val = np.sum(ss['gm'] * ss['c']) # steady-state consumption
        DT_val = dt_ss['C']                 # steady-state consumption in discrete time
    else:
        CT_val = 1  # no need to divide by steady state value for prices
        DT_val = 1

    plt.plot(np.arange(0, plot_t, dt), CT_irf[:int(plot_T / dt)] * 100 / CT_val, label = "Continuous (surprise)",
        linestyle = linestyles[0], color = default_colors[0],
        linewidth = 3)
    plt.plot(np.arange(0, plot_t, dt), CT_irf2[:int(plot_T / dt)] * 100 / CT_val, label = "Continuous (anticipation)",
        linestyle = linestyles[3], color = default_colors[0],
        linewidth = 4)
    plt.plot(DT_irf[:plot_t] * 100 / DT_val, label = "Discrete", linestyle = linestyles[1],
        color = default_colors[1], linewidth = 3)

    # Increase tick label sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("p.p. deviation from SS", fontsize=14)

    if shock_name == "w":
        plt.legend(framealpha=0)
    plt.tight_layout()
    plt.savefig(fig_dir + "IRF_" + shock_name + str_append + ".pdf")
    plt.show()

plot_IRF(DT_irf_r, irfs['r'], irfs2['r'], shock_name = "r", str_append = str_append, fig_dir = fig_dir)
plot_IRF(DT_irf_w, irfs['w'], irfs2['w'], shock_name = "w", str_append = str_append, fig_dir = fig_dir)

CT_K = J['r']['K'] @ irfs['r'] + J['w']['K'] @ irfs['w']
CT_C = J['r']['C'] @ irfs['r'] + J['w']['C'] @ irfs['w']

CT_K2 = J2['r']['K'] @ irfs2['r'] + J2['w']['K'] @ irfs2['w']
CT_C2 = J2['r']['C'] @ irfs2['r'] + J2['w']['C'] @ irfs2['w']

plot_IRF(G['K']['Z'] @ z_hat_dt, CT_K, CT_K2, shock_name = "K", str_append = str_append, fig_dir = fig_dir)
plot_IRF(G['C']['Z'] @ z_hat_dt, CT_C, CT_C2, shock_name = "C", str_append = str_append, fig_dir = fig_dir)

# Steady State and Parameter Tables

# Save parameter and steady-state objects to LaTeX table
p['beta'] = dt_ss['beta']
ct_keys = ['gamma', 'rho', 'beta', 'd', 'alpha', 'AgZ', 'rho_e', 'sigma_e', 'amin']
p2 = {key: p[key] for key in ct_keys}

ha_param = pd.DataFrame(list(p2.items()), columns=['Parameter', 'Value'])
ha_param['Description'] = [r"$\gamma$ in $u(c)=\frac{c^{1-\gamma}}{1-\gamma}$",
                           "Continuous Time Discount Rate","Discrete Time Discount Rate",
                           "Depreciation Rate","Capital Share",
                           "Aggregate Productivity","Idiosyncratic Productivity Persistence",
                           "Idiosyncratic Productivity Cross-Sectional Standard Deviation", "Min. Asset"]
ha_param['Parameter']   =  [r"$\gamma$",r"$\rho$",r"$\beta$",r"$\delta$",
                          r"$\alpha$",r"$\bar{Z}$",r"$\mu_e$",r"$\sigma_e$",
                          r"$\underline{a}$"]
ha_param['Value']       = ha_param['Value'].map(lambda x: f'{x:.2f}'.rstrip('0').rstrip('.'))

# Reorder and save
ha_param    = ha_param[['Parameter', 'Description', 'Value']]
latex_table = ha_param.to_latex(index=False, escape=False,
                             caption="HA Parameters",
                             label="tab:KS_Param",
                             column_format='llr')

# Add \centering to the LaTeX table string
latex_table = latex_table.replace("\\begin{tabular}", "\\centering\n\\begin{tabular}")

# Write the LaTeX table to a file
with open(fig_dir + "Parameters" + str_append + ".tex", 'w') as f:
    f.write(latex_table)

# Steady-state objects
ss_keys = ['r', 'w', 'A', 'Y', 'C', 'Share Constrained']
ss['A'] = np.sum(ss['gm'] * n['aa'])                           # Aggregate Capital
ss['C'] = np.sum(ss['gm'] * ss['c'])                           # Aggregate Consumption
Lab     = np.sum(ss['gm'] * n['zz'])                           # Aggregate Labor
ss['Y'] = ss['A']**p['alpha'] * Lab**(1-p['alpha']) * p['AgZ'] # Aggregate Output
ss['Share Constrained'] = np.sum(ss['gm'][ss['consted']])      # Share of agents constrained
ss2 = {key: ss[key] for key in ss_keys}

dt_ss['Share Constrained'] = np.sum(dt_ss['D'][ss['consted']]) # Share of agents constrained in discrete time
dt_ss2 = {key: dt_ss[key] for key in ss_keys}

ha_ss = pd.DataFrame(list(ss2.items()), columns=['SS Object', 'Cont Time Value'])
ha_ss['Discrete Time Value'] = [dt_ss2[key] for key in ss_keys]

ha_ss['Description'] = ["Real Interest Rate","Wage","Aggregate Capital",
                        "Aggregate Output","Aggregate Consumption",
                        "Share Constrained"]
ha_ss['SS Object'] =  [r"$r$",r"$w$",r"$K$",r"$Y$",
                          r"$C$","Share Constrained"]

# Reorder and save
ha_ss = ha_ss[['SS Object', 'Description', 'Cont Time Value', 'Discrete Time Value']]
latex_table = ha_ss.to_latex(index=False, escape=False,
                             float_format="%.3f",
                             caption="HA Steady State",
                             label="tab:KS_SS")

# Add \centering to the LaTeX table string
latex_table = latex_table.replace("\\begin{tabular}", "\\centering\n\\begin{tabular}")

# Write the LaTeX table to a file
with open(fig_dir + "SS" + str_append + ".tex", 'w') as f:
    f.write(latex_table)
