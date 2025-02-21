# This file produces the runtimes for each step of the solution method
from Model import *
import numpy as np
import pandas as pd
from toolkit import aux_speed as aux_jac # Discrete Time Code to get Jacobian
import HA_DT as ha # Discrete Time Version of Model
from Parameters import param_econ, param_num
from Runtime_Plot_Fns import plot_fn, plot_cumulative_runtime

# User Inputs
T            = 300       # Number of time periods
get_runtimes = True      # If False, just plots runtimes
iter_style   = 'DT_loop' # How to iterate forward in phi and E calculations
                         # 3 options: 'CT_matrix', 'CT_loop', 'DT_loop' where 'CT' uses ly and DT uses Pi and
                         # 'matrix' does a sparse matrix multiplication while 'loop' uses a for loop with numba

# Create Table to store runtimes
num_grids = 10 # Number of different gridpoints to test
n_type    = 3  # Number of different types of solution methods
n_tot     = num_grids * n_type

dim_size_list = [(2, 100), (5, 100), (7, 500), (15, 1500), (25, 2500), (30, 3000), (35, 3500), (40, 4000), (45, 4500), (50, 5000)]
df = {
    "Type": ["Discrete", "Continuous, EGM", "Continuous, Implicit"] * num_grids,
    "Gridpoints": [2*100] * n_type + [5*100] * n_type + [7*500] * n_type + \
                    [15*1500] * n_type + [25*2500] * n_type + \
                    [30*3000] * n_type + [35*3500] * n_type + \
                    [40*4000] * n_type + [45*4500] * n_type + [50*5000] * n_type,
    "Steady State": [np.nan] * n_tot,
    "Policy functions": [np.nan] * n_tot, # phi, D, cY
    "Expectation vector": [np.nan] * n_tot, # E
    "Fake news matrix": [np.nan] * n_tot, # Fake News
    "Jacobian": [np.nan] * n_tot, # Jacobian
    "Inversion": [np.nan] * n_tot, # IRF (solve)
}
runtime_df = pd.DataFrame(df)

# mu can be 1 if using DT_loop; otherwise less than 1
mu = 1 if iter_style == 'DT_loop' else 0.5

# Function to run HA model inc continuous time and store runtimes for each step given gridpoint
def run_ha(Nz, Na, runtime_df = runtime_df, EGM = True, dt = 1.0):
    # Parameters and Steady State
    p, n      = get_parameters(Nz = Nz, Na = Na, mu = mu)
    ss        = get_ss(p, n, EGM = EGM)
    ct_egm0   = %timeit -o -n 25 ss = get_ss(p, n, EGM = EGM)
    ss, store = step_0(p, n, ss, EGM = EGM, dt = dt)

    # Jacobians
    # run once to get all necessary values
    dphi_da, c_t, D = policy_function(p, n, ss, store, T, dt = dt, iter_style = iter_style)
    E_t             = expectation_vector(ss, store['E'], T, dt = dt, iter_style = iter_style)
    F               = fake_news(store['prices'], store['outputs'], E_t, D, T, c_t)
    J               = jacobian(store['prices'], store['outputs'], F)

    # get times for each step
    ct_egm1 = %timeit -o -n 100 dphi_da, c_t, D = policy_function(p, n, ss, store, T, dt = dt, iter_style = iter_style)
    ct_egm2 = %timeit -o -n 100 E_t = expectation_vector(ss, store['E'], T, dt = dt, iter_style = iter_style)
    ct_egm3 = %timeit -o -n 100 F = fake_news(store['prices'], store['outputs'], E_t, D, T, c_t)
    ct_egm4 = %timeit -o J = jacobian(store['prices'], store['outputs'], F, dt = dt)

    # get IRFs to a 1% TFP shock
    z_hat   = 0.01 * np.exp(-p['rho_Z'] * np.linspace(0, (T - 1) * dt, T))
    ct_egm5 = %timeit -o -n 100 irfs = inversion(p, n, ss, store, J, z_hat, T)

    # Store Results
    types = "Continuous, EGM" if EGM else "Continuous, Implicit"
    ind_egm = (runtime_df['Gridpoints'] == n['Ntot']).values & (runtime_df['Type'] == types).values
    runtime_df.loc[ind_egm, "Steady State"]       = ct_egm0.average
    runtime_df.loc[ind_egm, "Policy functions"]   = ct_egm1.average
    runtime_df.loc[ind_egm, "Expectation vector"] = ct_egm2.average
    runtime_df.loc[ind_egm, "Fake news matrix"]   = ct_egm3.average
    runtime_df.loc[ind_egm, "Jacobian"]           = ct_egm4.average
    runtime_df.loc[ind_egm, "Inversion"]          = ct_egm5.average

    return runtime_df

# Function to run HA model in discrete time and store runtimes for each step given gridpoint
def ha_dt(Nz, Na, runtime_df = runtime_df):
    ## Parameters and Steady State
    p  = param_econ()
    n  = param_num(p, Nz = Nz, Na = Na, mu = mu)

    # calculate steady state in discrete time
    dt_ss = ha.ha_ss_r(Nz = Nz, Na = Na, eis = 1/p['gamma'], delta = p['d'], alpha = p['alpha'],
                       rho = p['rho_e'], sigma = p['sigma_e'],
                       lb = 1e-5, ub = 0.9999*p['rho'], beta = np.exp(-p['rho']),
                       maxiter = n['Ir'], xtol = n['crit_S'], back_tol = n['back_tol'],
                       fwd_tol = n['fwd_tol'], back_maxit = n['back_maxit'], fwd_maxit = n['fwd_maxit'],
                       amax = n['amax'])

    dt_egm0 = %timeit -o -n 25 dt_ss = ha.ha_ss_r(Nz = Nz, Na = Na, eis = 1/p['gamma'], delta = p['d'], \
                                                  alpha = p['alpha'], rho = p['rho_e'], sigma = p['sigma_e'], \
                                                  lb = 1e-5, ub = 0.9999*p['rho'], beta = np.exp(-p['rho']), \
                                                  maxiter = n['Ir'], xtol = n['crit_S'], back_tol = n['back_tol'], \
                                                  fwd_tol = n['fwd_tol'], back_maxit = n['back_maxit'], fwd_maxit = n['fwd_maxit'], \
                                                  amax = n['amax'])

    shock_dict = {'r': {'r': 1}, 'w': {'w': 1}}
    ssinput_dict, ssy_list, outcome_list, V_name, a_pol_i, a_pol_pi, a_space = aux_jac.step_fake_0(ha.backward_iterate, dt_ss)

    # Jacobians
    curlyYs, curlyDs = aux_jac.step_fake_1(ha.backward_iterate, {'r': {'r': 1}, 'w': {'w': 1}}, dt_ss, ssinput_dict, ssy_list, outcome_list, V_name, a_pol_i, a_space, T)
    curlyPs = aux_jac.step_fake_2(dt_ss, outcome_list, ssy_list, a_pol_i, a_pol_pi, T)
    F = aux_jac.step_fake_3(shock_dict, outcome_list, curlyYs, curlyDs, curlyPs)
    J = aux_jac.step_fake_4(F, shock_dict, outcome_list)

    # Get runtimes
    dt_egm1 = %timeit -o -n 100 curlyYs, curlyDs = aux_jac.step_fake_1(ha.backward_iterate, {'r': {'r': 1}, 'w': {'w': 1}}, dt_ss, ssinput_dict, ssy_list, outcome_list, V_name, a_pol_i, a_space, T)
    dt_egm2 = %timeit -o -n 100 curlyPs = aux_jac.step_fake_2(dt_ss, outcome_list, ssy_list, a_pol_i, a_pol_pi, T)
    dt_egm3 = %timeit -o -n 100 F = aux_jac.step_fake_3(shock_dict, outcome_list, curlyYs, curlyDs, curlyPs)
    dt_egm4 = %timeit -o J = aux_jac.step_fake_4(F, shock_dict, outcome_list)

    # calculate impulse responses by getting firm and household jacobians
    z_hat_dt = 0.01 * np.exp([-p['rho_Z'] * t for t in range(T)]) # productivity shock w/ time step of 1
    dt_egm5  = %timeit -o -n 100 dt_irfs = ha.ha_J(dt_ss, z_hat_dt, J, T = T)

    # store results
    ind_dt = (runtime_df['Gridpoints'] == n['Ntot']).values & (runtime_df['Type'] == "Discrete").values
    runtime_df.loc[ind_dt, "Steady State"]       = dt_egm0.average
    runtime_df.loc[ind_dt, "Policy functions"]   = dt_egm1.average
    runtime_df.loc[ind_dt, "Expectation vector"] = dt_egm2.average
    runtime_df.loc[ind_dt, "Fake news matrix"]   = dt_egm3.average
    runtime_df.loc[ind_dt, "Jacobian"]           = dt_egm4.average
    runtime_df.loc[ind_dt, "Inversion"]          = dt_egm5.average

    return runtime_df

# Get the runtimes for each Nz, Na
runtime_df = pd.read_csv("Figures/HA_runtimes.csv")
if get_runtimes:
    for i in dim_size_list:
        print(i)
        Nz = i[0]
        Na = i[1]

        runtime_df = run_ha(Nz, Na, runtime_df = runtime_df, EGM = True)
        runtime_df = run_ha(Nz, Na, runtime_df = runtime_df, EGM = False)
        runtime_df = ha_dt(Nz, Na, runtime_df = runtime_df)

        runtime_df.to_csv("Figures/HA_runtimes.csv", index = False)
else:
    runtime_df = pd.read_csv("Figures/HA_runtimes.csv")

# Plot the HA data
plot_fn(runtime_df, str_append = "_HA_SS", no_legend = True)

# Plot cumulative runtime for HA data
plot_cumulative_runtime(runtime_df, typed = "Continuous, EGM", str_append = "_HA")
plot_cumulative_runtime(runtime_df, typed = "Continuous, Implicit", str_append = "_HA_Imp")
plot_cumulative_runtime(runtime_df, typed = "Discrete", str_append = "_HA_DT", no_legend = True)

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

# Save as LaTeX table
latex_table = runtime_df_tex.to_latex(index=False, caption="Runtime Data for HA (in seconds)",
                                label="tab:runtime_comp_ha", longtable=True, escape=False,
                                header=True, bold_rows=True,
                                float_format="%.3f",
                                column_format='l' + 'r' * (runtime_df_tex.shape[1] - 1))

# Add a line after every 3 rows
lines = latex_table.splitlines()
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if i > 18 and i % 3 == 1:  # Adjust the condition to add \hline after every 3 rows of data
        new_lines.append(r'\hline')

latex_table_with_lines = '\n'.join(new_lines)

with open("Figures/HA_comp_runtime.tex", "w") as file:
    file.write(latex_table_with_lines)
