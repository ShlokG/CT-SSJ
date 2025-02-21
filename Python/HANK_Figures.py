# This file plots the Jacobian and IRFs and
# gets the steady-state and parameter tables
# for the HANK model
from Model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get Calibration
from Storage import calibration, models_heterogeneous
import json

from HANK_Helpers import *
from sequence_jacobian.blocks.support.het_support import CombinedTransition

# User Inputs
dt         = 1.0           # time step
T          = int(300 / dt) # number of time periods
Nz         = 25            # number of idiosyncratic productivity states
Na         = 2500          # number of asset gridpoints
EGM        = True          # solve for steady state via endogenous gridpoint or implicit method
iter_style = 'DT_loop'     # how to iterate forward in phi and E calculations
                           # 3 options: 'CT_matrix', 'CT_loop', 'DT_loop' where 'CT' uses ly and DT uses Pi and
                           # 'matrix' does a sparse matrix multiplication while 'loop' uses a for loop with numba

# Plotting Inputs
plot_T = 30 # Number of periods to plot for IRFs

# Directories
fig_dir = 'Figures/HANK_'

# String to append to file names
str_append  = f"_Nz{Nz}_Na{Na}"
str_append += "_HJB" if not EGM else ""
str_append += f"_dt{dt}" if dt != 1.0 else ""

# Load calibration
with open('Storage/solved_params.json', 'r') as f:
    params = json.load(f)

calib_ha_one, _ = calibration.get_ha_calibrations()
hh_het, ss_het  = models_heterogeneous.get_all(params)
m = 'HA-hi-liq'

# load model and parameters
hh_het = hh_het[m]
params = params[m]

## Update calibration
calib_ha_one['n_e']  = Nz                # idiosyncratic productivity gridpoints
calib_ha_one['n_a']  = Na                # asset gridpoints
calib_ha_one['eis']  = 0.5               # elasticity of intertemporal substitution
calib_ha_one['beta'] = ss_het[m]['beta'] # discount rate

# mu can be 1 if using DT_loop; otherwise less than 1
mu = 1 if iter_style == 'DT_loop' else 0.5

# Get Jacobians in continuous time
p, n            = get_parameters_hank(calib_ha_one, EGM = EGM, mu = mu)                   # loads parameters and grids
ss              = get_ss(p, n, EGM = EGM, HANK = True)                                    # gets steady state
ss, store       = step_0(p, n, ss, EGM = EGM, HANK = True, dt = dt)                       # stores asset derivative, constrained agents, asset transition indices given savings rule
dphi_da, c_t, D = policy_function(p, n, ss, store, T, dt = dt, iter_style = iter_style)   # policy functions and distribution change from future shock
E_t             = expectation_vector(ss, store['E'], T, dt = dt, iter_style = iter_style) # expectation vector (for propagation of distribution)
F               = fake_news(store['prices'], store['outputs'], E_t, D, T, c_t)            # fake news operator
J               = jacobian(store['prices'], store['outputs'], F, dt = dt)                 # jacobian

# Shocks
T2 = T - 10 # reduce T since matrix non-invertible otherwise
dG = np.array([np.exp(-p['rho_G'] * i * dt) for i in range(T2)]) # shock to govt spending
dr = np.array([p['rho_r'] ** (i * dt) for i in range(T2)])       # shock to interest rates

# calculate value for bonds to accommodate gov't spending shock
dB = np.empty(T2)
for t in range(T2):
    dB_lag = dB[t-1] if t > 0 else 0
    dB[t]  = (1 - p['rho_b'] * dt) * (dB_lag + dG[t] * dt) # dG annualized but dB is flow so multiplied by dt

# calculate corresponding tax process to balance budget
dB_lag = np.concatenate(([0], dB[:-1]))
dT     = dG + ((1 + ss['r']) * dB_lag - dB) / dt # divide bond change by dt to annualize

# Calculate aggregate consumption and capital
C       = np.sum(ss['gm'] * ss['c']) # aggregate consumption
Capital = np.sum(ss['gm'] * n['aa']) # aggregate capital

# Get curly M matrix and IRFs
M, dY_dG_ge, _  = GE_Jacs(J, dG, dT, np.zeros(T2), T, T2, p, ss['r'], C, Capital, M = None)
_, dY_dr_ge, _  = GE_Jacs(J, np.zeros(T2), np.zeros(T2), dr, T, T2, p, ss['r'], C, Capital, M = M)

# With Anticipation
dphi_da2, c_t2, D2 = policy_function(p, n, ss, store, T, anticipate = True, dt = dt, iter_style = iter_style)
F2                 = fake_news(store['prices'], store['outputs'], E_t, D2, T, c_t2)
J2                 = jacobian(store['prices'], store['outputs'], F2, dt = dt)

M2, dY_dG_ge2, _   = GE_Jacs(J2, dG, dT, np.zeros(T2), T, T2, p, ss['r'], C, Capital, M = None)
_, dY_dr_ge2, _    = GE_Jacs(J2, np.zeros(T2), np.zeros(T2), dr, T, T2, p, ss['r'], C, Capital, M = M2)

# Get Jacobians in discrete time

# Steady State Calculation in discrete time
dt_ss, dt_sol = hank_ss_r(hh_het, params, calib_ha_one, p, lb = n['rmin'], ub = n['rmax'],
                    maxiter = n['Ir'], xtol = n['crit_S'], backward_tol = n['back_tol'], backward_maxit = n['back_maxit'],
                    forward_tol = n['fwd_tol'], forward_maxit = n['fwd_maxit'])

# Set up for Discrete Time Jacobians
inputs     = ['r', 'Z'] # prices for discrete time
outputs_dt = ['C', 'A'] # outputs for discrete time
h          = 1E-4       # perturbation size for numerical differentiation
twosided   = False      # one-sided numerical differentiation

dt_ss      = hh_het.extract_ss_dict(dt_ss)
outputs_dt = hh_het.M_outputs.inv @ outputs_dt

# step 0: preliminary processing of steady state
exog           = hh_het.make_exog_law_of_motion(dt_ss)
endog          = hh_het.make_endog_law_of_motion(dt_ss)
differentiable_backward_fun, differentiable_hetinputs, differentiable_hetoutputs = hh_het.jac_backward_prelim(dt_ss, h, exog, twosided)
law_of_motion  = CombinedTransition([exog, endog]).forward_shockable(dt_ss['Dbeg'])
exog_by_output = {k: exog.expectation_shockable(dt_ss[k]) for k in outputs_dt | hh_het.backward}

# compute curlyY and curlyD (backward iteration) for each input i
curlyYs, curlyDs = hank_step1(hh_het, outputs_dt, T, differentiable_backward_fun,
    differentiable_hetinputs, differentiable_hetoutputs, law_of_motion, exog_by_output, inputs)

curlyPs = hank_step2(hh_het, dt_ss, T, law_of_motion, outputs_dt)   # expectation vector
curlyFs = hank_step3(curlyYs, curlyDs, curlyPs, outputs_dt, inputs) # Fake News Matrix
curlyJs = hank_step4(curlyFs, outputs_dt, inputs) # Jacobian

# shocks when time step is 1
dG_dt = np.array([np.exp(-p['rho_G'] * i * dt) for i in range(T2)]) # shock to govt spending when time step 1
dr_dt = np.array([p['rho_r'] ** i for i in range(T2)])              # shock to interest rates when time step 1

# calculate value for bonds
dB_dt = np.empty(T2)
for t in range(T2):
    dB_lag_dt = dB_dt[t-1] if t > 0 else 0
    dB_dt[t] = (1 - p['rho_b']) * (dB_lag_dt + dG_dt[t])

# calculate corresponding tax process to balance budget
dB_lag_dt = np.concatenate(([0], dB_dt[:-1]))
dT_dt     = dG_dt + (1 + dt_ss['r']) * dB_lag_dt - dB_dt

# calculate GE Jacobians
curlyMs, dY_dG_dt, _ = GE_Jacs_dt(curlyJs, dG_dt, dT_dt, np.zeros(T2), T, T2, p, dt_ss['r'], dt_ss['C'], dt_ss['A'], curlyMs = None)
_, dY_dr_dt, _       = GE_Jacs_dt(curlyJs, np.zeros(T2), np.zeros(T2), dr_dt, T, T2, p, dt_ss['r'], dt_ss['C'], dt_ss['A'], curlyMs = curlyMs)

# Plot MPC Jacobian
orig_cols       = [0, 100, 200] # time periods to plot
columns_to_plot = [int(orig_cols[i] / dt) for i in range(len(orig_cols))] # converted to columns when dt != 1
linestyles      = ['-', '--', '-.', ':', (0, (1, 10))]
default_colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, col in enumerate(columns_to_plot):
    plt.plot(np.arange(0, T * dt, dt), J['w']['C'][:, col] / C,
             label = f"Continuous (surprise)",
             linestyle = linestyles[0], color = default_colors[0], linewidth = 3)
    plt.plot(np.arange(0, T * dt, dt), J2['w']['C'][:, col] / C,
             label=f"Continuous (anticipated)",
             linestyle = linestyles[3], linewidth = 4, color = default_colors[0])
    plt.plot(np.arange(0, int(T * dt)), curlyJs['C']['Z'][:int(T * dt), orig_cols[i]] / dt_ss['C'],
             label = f"Discrete", linestyle = linestyles[1],
             color = default_colors[1], alpha = 0.8, linewidth = 3)
    if i == 0:
        plt.legend(framealpha=0)

plt.xlabel("Year", fontsize=14)
plt.ylabel("p.p. deviation from SS", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.tight_layout()
plt.savefig(fig_dir + "Jac_YZ" + str_append + ".pdf")
plt.show()

# Plot IRFs
plt.plot(dY_dG_ge[:plot_T], label = "Continuous (surprise)", linestyle = linestyles[0], linewidth = 3, color = default_colors[0])
plt.plot(dY_dG_ge2[:plot_T], label = "Continuous (anticipated)", linestyle = linestyles[3], linewidth = 4, color = default_colors[0])
plt.plot(dY_dG_dt[:plot_T], label = "Discrete", linestyle = linestyles[1], linewidth = 3, color = default_colors[1])
plt.legend(framealpha = 0)
plt.xlabel("Year", fontsize = 14)
plt.ylabel("p.p. deviation from SS", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.tight_layout()
plt.savefig(fig_dir + "IRF_YG" + str_append + ".pdf")
plt.show()

plt.plot(dY_dr_ge[:plot_T], label = "Continuous (surprise)", linestyle = linestyles[0], linewidth = 3, color = default_colors[0])
plt.plot(dY_dr_ge2[:plot_T], label = "Continuous (anticipated)", linestyle = linestyles[3], linewidth = 4, color = default_colors[0])
plt.plot(dY_dr_dt[:plot_T], label = "Discrete", linestyle = linestyles[1], linewidth = 3, color = default_colors[1])
plt.legend(framealpha = 0)
plt.xlabel("Year", fontsize = 14)
plt.ylabel("p.p. deviation from SS", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.tight_layout()
plt.savefig(fig_dir + "IRF_Yr" + str_append + ".pdf")
plt.show()

# Note: If IRFs have the opposite sign as expected in the first period, increase T

# Save parameters and steady state for reporting
p_store = {
    r'$\gamma$': p['gamma'],
    r'$\rho$': p['rho'],
    r'$\beta$': p['beta'],
    r'$\bar{Z}$': p['AgZ'],
    r'$\mu_w$': p['mu_w'],
    r'$\phi$': p['phi'],
    r'$\kappa_w$': p['kappa_w'],
    r'$\theta$': p['theta'],
    r'$\xi$': p['xi'],
    r'$N^{ss}$': p['Nss'],
    r'$Y^{ss}$': p['Yss'],
    r'$\frac{G}{Y}$': p['G_share'],
    r'$\frac{T}{Y}$': p['T_share'],
    r'$\rho_e$': p['rho_e'],
    r'$\sigma_e$': p['sigma_e'],
    r'$\underline{a}$': p['amin'],
}
ct_param_df = pd.DataFrame(list(p_store.items()), columns=['Parameter', 'Value'])
ct_param_df['Value'] = ct_param_df['Value'].map(lambda x: f'{x:.3f}'.rstrip('0').rstrip('.'))
ct_param_df['Description'] = [r"$\gamma$ in $u(c)=\frac{c^{1-\gamma}}{1-\gamma}$","Continuous Time Discount Rate","Discrete-Time Discount Rate",\
    "Aggregate Productivity","Wage Markup",\
    r"Taylor Rule Coefficient on $\pi$",\
    "Wage Flexibility", "Tax Parameter", "Labor Supply Elasticity",\
    "Steady State Labor", "Steady State Output",\
    "Government Share", "Transfer Share", "Idiosyncratic Productivity Persistence",\
    "Idiosyncratic Productivity Cross-Sectional Standard Deviation","Minimum Asset Constraint"]
ct_param_df = ct_param_df[['Parameter', 'Description', 'Value']]

# Convert to LaTeX
latex_table = ct_param_df.to_latex(index=False, escape=False,
                        caption="HANK Parameters",
                        label="tab:IKC_Param",
                        column_format='llr')

# Add \centering to the LaTeX table string
latex_table = latex_table.replace("\\begin{tabular}", "\\centering\n\\begin{tabular}")

# Write the LaTeX table to a file
with open(fig_dir + "ct_params_py" + str_append + ".tex", 'w') as f:
    f.write(latex_table)


# Save Steady State
ss_store = {
    r"$r$": ss['r'],
    r"$K$": Capital,
    r"$C$": C,
    "Share Constrained": np.sum(ss['gm'][ss['consted']])
}

ct_ss_df = pd.DataFrame(list(ss_store.items()), columns=['SS Object', 'Cont Time Value'])
ct_ss_df['Cont Time Value'] = ct_ss_df['Cont Time Value']#.map(lambda x: f'{x:.3f}'.rstrip('0').rstrip('.'))
ct_ss_df['Discrete Time Value'] = [dt_ss['r'], dt_ss['A'], dt_ss['C'], np.sum(dt_ss['D'][ss['consted']])]
# ct_ss_df['Discrete Time Value'] = ct_ss_df['Discrete Time Value'].map(lambda x: f'{x:.3f}').rstrip('0').rstrip('.'))
ct_ss_df['Description'] = ['Real Interest Rate', 'Aggregate Capital', 'Aggregate Consumption', 'Share Constrained']

ct_ss_df = ct_ss_df[['SS Object', 'Description', 'Cont Time Value', 'Discrete Time Value']]
# Convert to LaTeX
latex_table = ct_ss_df.to_latex(index=False, escape=False,
                        float_format="%.3f",
                        caption="HANK Steady State",
                        label="tab:IKC_SS")

# Add \centering to the LaTeX table string
latex_table = latex_table.replace("\\begin{tabular}", "\\centering\n\\begin{tabular}")

# Write the LaTeX table to a file
with open(fig_dir + "ct_ss_py" + str_append + ".tex", 'w') as f:
    f.write(latex_table)
