# This file constructs the required functions to solve the HA and HANK Models

import numpy as np
from Parameters import param_econ, param_num, param_econ_hank, param_num_hank
from Jacobian_Helpers import *
from Steady_State import solve_steady_state, prelim_step
import sequence_jacobian as sj
from sequence_jacobian.classes import SimpleSparse

# Load Parameters
def get_parameters(Nz = 7, Na = 500, mu = 1.0):
    p = param_econ()                            # get economic parameters
    n = param_num(p, Nz = Nz, Na = Na, mu = mu) # get grid and convergence parameters
    return p, n

def get_parameters_hank(calib, mu = 1.0, EGM = True):
    p    = param_econ_hank(calib) # get parameters from file
    Na   = calib['n_a']           # use given dict for calibrating grid
    Nz   = calib['n_e']
    amax = calib['max_a']

    # if too few asset gridpts, lower amax to shrink asset step size
    if Na <= 100 and Nz <= 2 and not EGM:
        amax = np.minimum(400, amax)
    n, p = param_num_hank(p, Nz = Nz, Na = Na, amax = amax, mu = mu) # numerical parameters
    return p, n

# Utility Functions
def utility(c, gamma):
    if gamma != 1:
        return (c ** (1 - gamma)) / (1 - gamma)
    else:
        return np.log(c)

# u'(c)
def utility_prime(c, gamma):
    return c ** (-gamma)

# u'^{-1}(c)
def utility_prime_inv(c, gamma):
    return c ** (-1 / gamma)

# u"(c)
def utility_second_deriv(c, gamma):
    return -gamma * c ** (-gamma - 1)

# Calculate Steady State

# Need backward iteration for model if using EGM
def hh_CT(Va_p, a_grid, y, r, beta, gamma):
    uc_nextgrid       = beta * Va_p
    c_nextgrid        = utility_prime_inv(uc_nextgrid, gamma)
    coh               = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a                 = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    a[a < a_grid[0]]  = a_grid[0]
    c                 = coh - a
    Va                = (1 + r) * utility_prime(c, gamma)
    return Va, a, c

def get_ss(p, n, EGM = True, HANK = False):
    ss = solve_steady_state(p, n, fn = hh_CT, EGM = EGM, HANK = HANK)
    return ss

# STEP 0: Calculate necessary values
## store captures everything the user should provide
### (some matrices unnecessary if others provided)
def step_0(p, n, ss, EGM = True, HANK = False, dt = 1.0):
    ss        = prelim_step(ss, n, EGM = EGM, dt = dt)    # Calculate asset derivatives, which agents constrained, indices given savings rule
    ss['up']  = utility_prime(ss['c'], p['gamma'])        # u'(c) at steady state
    ss['upp'] = utility_second_deriv(ss['c'], p['gamma']) # u"(c) at steady state

    store            = {}                           # dictionary to store user-given values
    store['inc_chg'] = {'r': n['aa'], 'w': n['zz']} # direct income change from shock

    # Set up dicts and arrays for solution
    store['prices']       = ['r', 'w']      # prices in the model
    store['outputs']      = ['K', 'C']      # outputs of interest
    store['endog_states'] = ['a']           # endogenous idiosyncratic state variables
    store['DA_mat']       = {'a': ss['DA']} # derivatives w.r.t. endogenous state variables

    store['L_p']  = {'r': n['aa'].flatten() * ss['DA'], 'w': n['zz'].flatten() * ss['DA']} # Derivative of transition operator w.r.t prices
    store['LT_c'] = lambda x: ss['DA_T'] @ x # Derivative of distribution transition w.r.t. control. DA_T is for distribution changes
    store['LT_p'] = {'r': lambda g: -ss['DA_T'] @ (n['aa'].flatten() * g), 'w': lambda g: -ss['DA_T'] @ (n['zz'].flatten() * g)} # Derivative of distribution transition w.r.t. prices

    # Constraints
    store['U']    = calc_U(ss['upp']) # u"(c) at steady state
    store['C_p']  = {'r': (-ss['upp']* n['aa']).flatten()[ss['consted_ind']], 'w': (-ss['upp'] * n['zz']).flatten()[ss['consted_ind']]} # Derivative of constraints w.r.t prices
    store['C_V']  = 0 # Derivative of constraints w.r.t value function at steady state
    store['C_dV'] = 1 # Derivative of constraints w.r.t derivative of value function at steady state

    # Other values that in HA and HANK are unnecessary or zero
    store['Vss']  = None # steady-state value function
    store['u_p']  = 0    # derivative of utility with respect to prices
    store['u_z']  = 0    # derivative of utility with respect to aggregate shock
    store['u_cp'] = None # derivative of utility with respect to consumption and price
    store['u_cz'] = None # derivative of utility with respect to consumption and aggregate shock

    # Get steady-state values for labor and capital for HA
    if not HANK:
        Lab = np.sum(ss['gm'] * n['zz'])  # steady-state labor
        K   = np.sum(ss['gm'] * n['aa'])  # steady-state capital
        # immediate response of prices to aggregate productivity shock
        store['zeta'] = {'r': p['alpha'] * (Lab / K) ** (1 - p['alpha']), \
                         'w': (1 - p['alpha']) * (Lab / K) ** (-p['alpha'])}

    # Get the effect of a change in distribution at time 0 on outputs at time 0
    E = {}
    ## First, calculate E_0^o where o is the desired output
    if 'K' in store['outputs']:
        E['K'] = n['aa']
    if 'C' in store['outputs']:
        E['C'] = ss['c']

    if 'r' in store['outputs'] or 'w' in store['outputs']:
        Lab = np.sum(ss['gm'] * n['zz']) # steady-state labor
        K   = np.sum(ss['gm'] * n['aa']) # steady-state capital 
        E   = p['AgZ'] * p['alpha'] * (1 - p['alpha']) * (Lab / K) ** (1 - p['alpha']) * (n['zz'] / Lab - n['aa'] / K)
        if 'r' in store['outputs']:
            E['r'] = E
        if 'w' in store['outputs']:
            E['w'] = -E * K / Lab
    
    store['E'] = E

    return ss, store

# STEP 1: calculate change in value from future shock and consumption at time 0
def policy_function(p, n, ss, store, T, anticipate = False, dt = 1.0, iter_style = 'DT_loop'):
    # Dictionaries to store for each price shock, the
    # derivative of change in value function, consumption at time 0, and distribution at time 0
    dphi_da = {}
    c_t     = {}
    D       = {}

    for pr in store['prices']:
        # calculate change in value for each agent at each time
        phi_t = calc_phi(ss, n, T, dt = dt, phi0 = store['inc_chg'][pr] * ss['up'], L_p = store['L_p'][pr], \
                         Vss = store['Vss'], u_p = store['u_p'], C_p = store['C_p'][pr], \
                         C_V = store['C_V'], C_dV = store['C_dV'], price = True, iter_style = iter_style)
        # get derivative of change in value function w.r.t. assets
        dphi_da[pr] = ss['DA'] @ phi_t # use calc_dphi_dx when multiple endogenous states

        # adjust time-0 for ex-post income shock
        if not anticipate:
            dphi_da[pr][:, 0] = (ss['DA'] @ ss['up'].flatten()) * store['inc_chg'][pr].flatten()

        # calculate change in consumption at time 0
        c_pr = calc_policy_fn(store['U'], T, p['rho'], ss['gm'], dt = dt,
                                 u_cp = store['u_cp'], u_cz = store['u_cz'], dphi_da = dphi_da[pr],
                                 L_c = None, phi_t = None)

        # adjust consumption at time 0
        if pr == 'r' and not anticipate:
            c_prime    = ss['DA'] @ ss['c'].ravel()             # c'(a)
            c_pr[:, 0] = (n['aa'] * ss['gm']).ravel() * c_prime # adjusted time-0 consumption

        # change in distribution
        D[pr], P = calc_D(ss['gm'], p['rho'], store['LT_c'], store['U'], phi_t, store['LT_p'][pr],
                           T, dt = dt, L_c = None, dphi_da = dphi_da[pr],
                           u_cp = store['u_cp'], u_cz = store['u_cz'], c_t = c_pr, price = True)

        # aggregate change in consumption
        c_t[pr] = np.sum(c_pr, axis=0)

    return dphi_da, c_t, D

# STEP 2: E_t = Change in output at time t just from change in distribution at time 0
def expectation_vector(ss, E, T, dt = 1.0, iter_style = 'DT_loop'):
    # Iterate forward to get E_t for t > 0
    E_t = calc_E(E, T, ss, dt = dt, iter_style = iter_style)
    return E_t

# STEP 3: Calculate the Fake News Matrix
def fake_news(prices, outputs, E_t, D, T, c_t):
    F = {}
    for pr in prices:
        F[pr] = calc_F(E_t, D[pr], outputs, T, c_t = {'C': c_t[pr]})
    
    return F

# STEP 4: Calculate the Jacobian
def jacobian(prices, outputs, F, dt = 1.0):
    J = {}
    for pr in prices:
        J[pr] = calc_J(F[pr], outputs, dt = dt)
    return J

# STEP 5: Calculate IRFs (in General Equilibrium)
## HA Model-specific IRFs
def IRF_DT(J_firm, J_ha, T, z_hat):
    # effect of Z and K on K via changing prices
    J_curlyK_K = J_ha['A']['r'] @ J_firm['r']['K'] + J_ha['A']['w'] @ J_firm['w']['K']
    J_curlyK_Z = J_ha['A']['r'] @ J_firm['r']['Z'] + J_ha['A']['w'] @ J_firm['w']['Z']

    J_curlyK = {'K': J_curlyK_K, 'Z' : J_curlyK_Z}

    # set up system of equations
    H_K = J_curlyK['K'] - np.eye(T)
    H_Z = J_curlyK['Z']

    G = {'K': -np.linalg.solve(H_K, H_Z)} # solve system of equations

    # get impulse responses for prices based on implied solution of K
    G['r'] = J_firm['r']['Z'] + J_firm['r']['K'] @ G['K']
    G['w'] = J_firm['w']['Z'] + J_firm['w']['K'] @ G['K']

    # direct effect of Z on prices
    J_rz   = G['r'] @ z_hat
    J_wz   = G['w'] @ z_hat

    return J_rz, J_wz

def inversion(p, n, ss, store, J, z_hat, T):
    # if Jacobians for prices calculated, can automatically solve system
    if set(store['prices']).issubset(store['outputs']):
        return calc_irfs(J, z_hat, store['zeta'], store['prices'])
    
    # if not, apply model-specific knowledge
    Lab = np.dot(ss['gm'].ravel(), n['zz'].ravel()) # steady-state labor
    K   = np.dot(ss['gm'].ravel(), n['aa'].ravel()) # steady-state capital

    ## get firm Jacobians w.r.t. K first like in discrete time
    J_firm = {'r': {}, 'w': {}}

    # differentiate firm FOC w.r.t. K
    J_firm['r']['K'] = SimpleSparse({(0, 0): p['alpha'] * p['AgZ'] * (p['alpha'] - 1) * (K / Lab) ** (p['alpha'] - 2) / Lab})
    J_firm['w']['K'] = SimpleSparse({(0, 0): (1 - p['alpha']) * p['alpha'] * p['AgZ'] * (K / Lab) ** (p['alpha'] - 1) / Lab})

    # direct effect of Z shock on prices
    J_firm['r']['Z'] = SimpleSparse({(0, 0): store['zeta']['r']})
    J_firm['w']['Z'] = SimpleSparse({(0, 0): store['zeta']['w']})

    # rename to apply to impulse responses function
    J_ha = {'A': {}}
    J_ha['A']['r'] = J['r']['K']
    J_ha['A']['w'] = J['w']['K']

    # get and store impulse responses
    irf_tup = IRF_DT(J_firm, J_ha, T, z_hat)
    irfs    = {'r': irf_tup[0], 'w': irf_tup[1]}

    return irfs


