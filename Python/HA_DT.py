"""
HA model

Model DAG:
blocks     = [household, firm, mkt_clearing]
unknowns   = [K]
targets    = [asset_mkt]
exogenous  = [Z]
"""

import numpy as np
import scipy.optimize as opt
from toolkit import utils, het_block as het
from toolkit.simple_block import simple
from sequence_jacobian.classes import SimpleSparse
from Model import IRF_DT

'''Part 1: Het block'''


def backward_iterate(Va_p, Pi_p, a_grid, e_grid, r, w, beta, eis):
    """Single backward iteration step using endogenous gridpoint method for households with CRRA utility.

    Order of returns matters! backward_var, assets, others

    Parameters
    ----------
    Va_p : np.ndarray
        marginal value of assets tomorrow
    Pi_p : np.ndarray
        Markov transition matrix for skills tomorrow
    a_grid : np.ndarray
        asset grid
    e_grid : np.ndarray
        skill grid
    r : float
        ex-post interest rate
    w : float
        wage
    beta : float
        discount rate today
    eis : float
        elasticity of intertemporal substitution

    Returns
    ----------
    Va : np.ndarray, shape(nS, nA)
        marginal value of assets today
    a : np.ndarray, shape(nS, nA)
        asset policy today
    c : np.ndarray, shape(nS, nA)
        consumption policy today
    """
    uc_nextgrid = (beta * Pi_p) @ Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    a = utils.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    utils.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c


household = het.HetBlock(backward_iterate, exogenous='Pi', policy='a', backward='Va')

'''Part 2: Simple Blocks'''


@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y

@simple
def mkt_clearing(K, A):
    asset_mkt = K - A
    return asset_mkt


'''Part 3: Steady state'''

def ha_ss_r(lb=0.98, ub=0.999, beta = 0.95, eis=1, delta=0.025, alpha=0.11, rho=0.966, sigma=0.5, Nz=7, Na=500, amax=200, maxiter = 1000, xtol = 1e-6, back_tol = 1e-8, fwd_tol = 1e-10, back_maxit = 5000, fwd_maxit = 100_000):
    """Solve steady state of full GE model when r is unknown (calibrated to clear asset markets)"""
    # set up grid
    a_grid = np.linspace(0, amax, Na)
    e_grid, _, Pi = utils.markov_rouwenhorst(rho=rho, sigma=sigma, N=Nz)

    L = 1
    Z = 1

    def f(r):
        rk = r + delta
        new_w = (1 - alpha) * Z * (alpha * Z / rk) ** (alpha / (1 - alpha))
        Va = (new_w * e_grid[:, np.newaxis] + r * a_grid[np.newaxis, :])**(- 1/eis) * r / (1 - beta)

        KS = household.ss(Pi=Pi, a_grid=a_grid, e_grid=e_grid, r=r, w=new_w, beta=beta, eis=eis,
                                                    Va=Va, backward_tol=back_tol, backward_maxit=back_maxit,
                                                    forward_tol=fwd_tol, forward_maxit=fwd_maxit)['A']

        KD = (alpha * Z / rk) ** (1 / (1 - alpha)) * Z / L

        return KS - KD

    r, sol = opt.brentq(f, lb, ub, full_output=True, maxiter=maxiter, xtol=xtol)

    rk = r + delta
    w = (1 - alpha) * Z * (alpha * Z / rk) ** (alpha / (1 - alpha))
    Va = (w * e_grid[:, np.newaxis] + r * a_grid[np.newaxis, :])**(- 1/eis)
    
    K = (alpha * Z / rk) ** (1 / (1 - alpha))
    Y = Z * K ** alpha

    if not sol.converged:
        raise ValueError('Steady-state solver did not converge.')

    # extra evaluation to report variables
    ss = household.ss(Pi=Pi, a_grid=a_grid, e_grid=e_grid, r=r, w=w, beta=beta, eis=eis, Va=Va,
                      backward_tol=back_tol, backward_maxit=back_maxit, forward_tol=fwd_tol, forward_maxit=fwd_maxit)
    ss.update({'Z': Z, 'K': K, 'L': 1, 'Y': Y, 'alpha': alpha, 'delta': delta, 'goods_mkt': Y - ss['C'] - delta * K})

    return ss

# Get IRFs for discrete time HA model
def ha_J(dt_ss, z_hat, J_ha, T = 300):
    # firm Jacobian: r and w as functions of K and Z
    ## can also get by firm.jacobian(dt_ss, inputs = ['K', 'Z'])
    ## if dt_ss is a SteadyStateDict object and simple block imported from sequence_jacobian
    J_firm = {'r': {}, 'w': {}}
    J_firm['r']['K'] = SimpleSparse({(-1, 0): dt_ss['alpha'] * dt_ss['Z'] * (dt_ss['alpha'] - 1) * (dt_ss['K'] / dt_ss['L']) ** (dt_ss['alpha'] - 2) / dt_ss['L']})
    J_firm['w']['K'] = SimpleSparse({(-1, 0): (1 - dt_ss['alpha']) * dt_ss['alpha'] * dt_ss['Z'] * (dt_ss['K'] / dt_ss['L']) ** (dt_ss['alpha'] - 1) / dt_ss['L']})

    J_firm['r']['Z'] = SimpleSparse({(0, 0): dt_ss['alpha'] * (dt_ss['K'] / dt_ss['L']) ** (dt_ss['alpha'] - 1)})
    J_firm['w']['Z'] = SimpleSparse({(0, 0): (1 - dt_ss['alpha']) * (dt_ss['K'] / dt_ss['L']) ** dt_ss['alpha']})

    # rename household Jacobian to use in IRF_DT
    J_ha['A'] = J_ha['a']

    # Get IRFs
    return IRF_DT(J_firm, J_ha, T, z_hat)