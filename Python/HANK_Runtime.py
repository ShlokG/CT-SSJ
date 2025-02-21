# This file gets the run times for the HANK model
from Model import *
import numpy as np
import pandas as pd

from Storage import calibration, models_heterogeneous
import json
import time

from Runtime_Plot_Fns import plot_fn, plot_cumulative_runtime
from HANK_Helpers import *
from sequence_jacobian.blocks.support.het_support import CombinedTransition

# User Inputs
T            = 300       # number of time periods
get_runtimes = True      # if False, just plot the runtimes
iter_style   = 'DT_loop' # how to iterate forward in phi and E calculations
                         # 3 options: 'CT_matrix', 'CT_loop', 'DT_loop' where 'CT' uses ly and DT uses Pi and
                         # 'matrix' does a sparse matrix multiplication while 'loop' uses a for loop with numba

# Create table to store runtimes
num_grids = 10 # Number of different gridpoints to test
n_type    = 3  # Number of different types of solution methods
n_tot     = num_grids * n_type

dim_size_list = [(2, 100), (5, 100), (11, 200), (15, 1500), (25, 2500), (30, 3000), (35, 3500), (40, 4000), (45, 4500), (50, 5000)]
df = {
    "Type": ["Discrete", "Continuous, EGM", "Continuous, Implicit"] * num_grids,
    "Gridpoints": [2*100] * n_type + [5*100] * n_type + [11*200] * n_type + \
                    [15*1500] * n_type + [25*2500] * n_type + \
                    [30*3000] * n_type + [35*3500] * n_type + \
                    [40*4000] * n_type + [45*4500] * n_type + [50*5000] * n_type,
    "Steady State": [np.nan] * n_tot,
    "Policy functions": [np.nan] * n_tot, # phi, cD, cY
    "Expectation vector": [np.nan] * n_tot, # cE
    "Fake news matrix": [np.nan] * n_tot, # Fake News
    "Jacobian": [np.nan] * n_tot, # Jacobian
    "Inversion": [np.nan] * n_tot, # IRF (solve)
}
runtime_df = pd.DataFrame(df)

# Load given calibration
with open('Storage/solved_params.json', 'r') as f:
    params = json.load(f)

calib_ha_one, _ = calibration.get_ha_calibrations()
hh_het, ss_het = models_heterogeneous.get_all(params)
m = 'HA-hi-liq'

## Update calibration
hh_het               = hh_het[m]
params               = params[m]
calib_ha_one['beta'] = ss_het[m]['beta']
calib_ha_one['eis']  = 0.5

# Shocks
T2 = T - 10               # reduce T since matrix non-invertible otherwise
dG = np.array([np.exp(np.log(0.8) * i) for i in range(T2)]) # shock to govt spending
dr = 0.8 ** np.arange(T2) # shock to interest rates

# bond process given G shock
dB = np.empty(T2)
for t in range(T2):
    dB_lag = dB[t - 1] if t > 0 else 0
    dB[t]  = (1 + np.log(0.5)) * (dB_lag + dG[t])

dB_lag = np.concatenate(([0], dB[:-1]))

# mu can be 1 if using DT_loop; otherwise less than 1
mu = 1 if iter_style == 'DT_loop' else 0.5

# Function to run HANK model and store runtimes for each step given gridpoint
def run_hank(Nz, Na, runtime_df = runtime_df, EGM = True, dt = 1.0):
    calib_ha_one['n_e'] = Nz
    calib_ha_one['n_a'] = Na

    # Get Jacobians in continuous time to get arguments to pass into runtime calls
    p, n            = get_parameters_hank(calib_ha_one, EGM = EGM, mu = mu)           # loads parameters and grids
    ss              = get_ss(p, n, EGM = EGM, HANK = True)                                    # gets steady state
    ss, store       = step_0(p, n, ss, EGM = EGM, HANK = True, dt = dt)                       # stores asset derivative, constrained agents, asset transition indices given savings rule
    dphi_da, c_t, D = policy_function(p, n, ss, store, T, dt = dt, iter_style = iter_style)   # policy functions and distribution change from future shock
    E_t             = expectation_vector(ss, store['E'], T, dt = dt, iter_style = iter_style) # expectation vector (for propagation of distribution)
    F               = fake_news(store['prices'], store['outputs'], E_t, D, T, c_t)            # fake news operator
    J               = jacobian(store['prices'], store['outputs'], F, dt = dt)                 # jacobian

    # Get times
    ct_egm0 = %timeit -o -n 25 ss               = get_ss(p, n, EGM = EGM, HANK = True)
    ct_egm1 = %timeit -o -n 100 dphi_da, c_t, D = policy_function(p, n, ss, store, T, dt = dt, iter_style = iter_style)
    ct_egm2 = %timeit -o -n 100 E_t             = expectation_vector(ss, store['E'], T, dt = dt, iter_style = iter_style)
    ct_egm3 = %timeit -o -n 100 F               = fake_news(store['prices'], store['outputs'], E_t, D, T, c_t)
    ct_egm4 = %timeit -o J                      = jacobian(store['prices'], store['outputs'], F, dt = dt) 

    dT = dG + (1 + ss['r']) * dB_lag - dB # tax process given G and B shocks

    # Get curly M matrix and IRFs
    C = np.sum(ss['gm'] * ss['c'])
    Capital = np.sum(ss['gm'] * n['aa'])

    T2 = T - 10
    M, dY_dG_ge, _                              = GE_Jacs(J, dG, dT, np.zeros(T2), T, T2, p, ss['r'], C, Capital) # Response to G, T shock
    ct_egm5_1 = %timeit -o -n 100 M, dY_dG_ge, _ = GE_Jacs(J, dG, dT, np.zeros(T2), T, T2, p, ss['r'], C, Capital)
    ct_egm5_2 = %timeit -o -n 100 _, dY_dr_ge, _ = GE_Jacs(J, np.zeros(T2), np.zeros(T2), dr, T, T2, p, ss['r'], C, Capital, M = M) # Response to r shock

    # Store Results
    types = "Continuous, EGM" if EGM else "Continuous, Implicit"
    ind_egm = (runtime_df['Gridpoints'] == n['Ntot']).values & (runtime_df['Type'] == types).values
    runtime_df.loc[ind_egm, "Steady State"]       = ct_egm0.average
    runtime_df.loc[ind_egm, "Policy functions"]   = ct_egm1.average
    runtime_df.loc[ind_egm, "Expectation vector"] = ct_egm2.average
    runtime_df.loc[ind_egm, "Fake news matrix"]   = ct_egm3.average
    runtime_df.loc[ind_egm, "Jacobian"]           = ct_egm4.average
    runtime_df.loc[ind_egm, "Inversion"]          = ct_egm5_1.average + ct_egm5_2.average

    return runtime_df

# Get runtimes in discrete time
def hank_dt(Nz, Na, runtime_df = runtime_df):
    calib_ha_one['n_e'] = Nz
    calib_ha_one['n_a'] = Na

    p, n = get_parameters_hank(calib_ha_one, EGM = True, mu = mu) # loads parameters and grids

    # Steady State Calculation for IKC Model
    dt_ss, _ = hank_ss_r(hh_het, params, calib_ha_one, p, lb=n['rmin'], ub=n['rmax'],
                        maxiter = n['Ir'], xtol = n['crit_S'], backward_tol=n['back_tol'], backward_maxit=n['back_maxit'],
                        forward_tol=n['fwd_tol'], forward_maxit=n['fwd_maxit'])
    
    dt_egm0 = %timeit -o -n 25 dt_ss, _ = hank_ss_r(hh_het, params, calib_ha_one, p, lb=n['rmin'], ub=n['rmax'], \
                                                     maxiter = n['Ir'], xtol = n['crit_S'], backward_tol=n['back_tol'], backward_maxit=n['back_maxit'], \
                                                     forward_tol=n['fwd_tol'], forward_maxit=n['fwd_maxit'])

    # Set up for Discrete Time Jacobians
    inputs     = ['r', 'Z'] # prices for discrete time
    outputs_dt = ['C', 'A'] # outputs of interest
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
    F       = hank_step3(curlyYs, curlyDs, curlyPs, outputs_dt, inputs) # Fake News Matrix
    J       = hank_step4(F, outputs_dt, inputs)                         # Jacobian

    dt_egm1 = %timeit -o -n 100 curlyYs, curlyDs = hank_step1(hh_het, outputs_dt, T, differentiable_backward_fun, \
        differentiable_hetinputs, differentiable_hetoutputs, law_of_motion, exog_by_output, inputs)
    dt_egm2 = %timeit -o -n 100 curlyPs = hank_step2(hh_het, dt_ss, T, law_of_motion, outputs_dt)
    dt_egm3 = %timeit -o -n 100 F = hank_step3(curlyYs, curlyDs, curlyPs, outputs_dt, inputs)
    dt_egm4 = %timeit -o J = hank_step4(F, outputs_dt, inputs)

    dT = dG + (1 + dt_ss['r']) * dB_lag - dB # tax process given G and B shocks

    # GE jacobians and impulse responses
    curlyMs, dY_dG_dt, _ = GE_Jacs_dt(J, dG, dT, np.zeros(T2), T, T2, p, dt_ss['r'], dt_ss['C'], dt_ss['A'], curlyMs = None)
    dt_egm5_1 = %timeit -o -n 100 curlyMs, dY_dG_dt, _ = GE_Jacs_dt(J, dG, dT, np.zeros(T2), T, T2, p, dt_ss['r'], dt_ss['C'], dt_ss['A'], curlyMs = None)
    dt_egm5_2 = %timeit -o -n 100 _, dY_dr_dt, _       = GE_Jacs_dt(J, np.zeros(T2), np.zeros(T2), dr, T, T2, p, dt_ss['r'], dt_ss['C'], dt_ss['A'], curlyMs = curlyMs)

    # Store Results
    ind_dt = (runtime_df['Gridpoints'] == n['Ntot']).values & (runtime_df['Type'] == "Discrete").values
    runtime_df.loc[ind_dt, "Steady State"]       = dt_egm0.average
    runtime_df.loc[ind_dt, "Policy functions"]   = dt_egm1.average
    runtime_df.loc[ind_dt, "Expectation vector"] = dt_egm2.average
    runtime_df.loc[ind_dt, "Fake news matrix"]   = dt_egm3.average
    runtime_df.loc[ind_dt, "Jacobian"]           = dt_egm4.average
    runtime_df.loc[ind_dt, "Inversion"]          = dt_egm5_1.average + dt_egm5_2.average

    return runtime_df

# Get the runtimes for each Nz, Na
runtime_df = pd.read_csv("Figures/HANK_runtimes.csv")
if get_runtimes:
    for i in dim_size_list:
        print(i)
        Nz = i[0]
        Na = i[1]

        runtime_df = run_hank(Nz, Na, runtime_df = runtime_df, EGM = True)
        runtime_df = run_hank(Nz, Na, runtime_df = runtime_df, EGM = False)
        runtime_df = hank_dt(Nz, Na, runtime_df = runtime_df)

        runtime_df.to_csv("Figures/HANK_runtimes.csv", index = False)
else:
    runtime_df = pd.read_csv("Figures/HANK_runtimes.csv")

# Plot the HANK data
# plot_fn(runtime_df, str_append = "_HANK_SS", no_legend = False)

# Plot cumulative runtime for HANK data
plot_cumulative_runtime(runtime_df, typed = "Continuous, EGM", str_append = "_HANK")
plot_cumulative_runtime(runtime_df, typed = "Continuous, Implicit", str_append = "_HANK_Imp")
plot_cumulative_runtime(runtime_df, typed = "Discrete", str_append = "_HANK_DT", no_legend = True)

# Get Table of Runtimes
runtime_df_tex = runtime_df.copy()
runtime_df_tex['Gridpoints'] = runtime_df_tex['Gridpoints'].apply(lambda x: f"{x:,}")

# Create new column names with line breaks
new_columns = {
    col: f"\\makecell{{{' '.join(col.split()[:1])}\\\\ {' '.join(col.split()[1:])}}}" 
    for col in runtime_df_tex.columns
}
new_columns['Fake news matrix'] = "\\makecell{Fake news\\\\ matrix}"

# Rename columns
runtime_df_tex = runtime_df_tex.rename(columns=new_columns)

# Create LaTeX table
latex_table = runtime_df_tex.to_latex(
    index=False, 
    caption="Runtime Data for HANK (in seconds)",
    label="tab:runtime_comp_hank", 
    longtable=True, 
    escape=False,
    header=True, 
    bold_rows=True,
    float_format="%.3f",
    column_format='l' + 'r' * (runtime_df_tex.shape[1] - 1)
)

# Add lines after every 3 rows
lines = latex_table.splitlines()
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if i > 18 and i % 3 == 1:
        new_lines.append(r'\hline')

latex_table_with_lines = '\n'.join(new_lines)

# Save to file
with open("Figures/HANK_comp_runtime.tex", "w") as file:
    file.write(latex_table_with_lines)