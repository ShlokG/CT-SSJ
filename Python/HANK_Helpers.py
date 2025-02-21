# Get Matrices relevant for HANK Model

import numpy as np
from scipy.linalg import toeplitz
import scipy.optimize as opt

from sequence_jacobian.blocks.het_block import HetBlock

def Kmat(r,T):
    """ Build TxT K matrix with interest r """ 
    q = (1 + r) ** (-np.arange(T))
    K = np.triu(toeplitz(-q), 1) 
    return K

def getM(r, T, T2, J_CY, A = None):
    """ Calculate curlyM for GE model by pre-multiplying by K """
    K = Kmat(r, T)
    if A is None:
        A = K @ (np.eye(T) - J_CY) # Asset Jacobian
    curlyMs = np.linalg.solve(A[:T2,:T2], K[:T2,:T2])
    return curlyMs

# Calculate all the GE Jacobians and IRFs
def GE_Jacs(J, dG, dT, dr, T, T2, p, r, C, Capital, M = None):
    if M is None:
        q = (1 + r) ** (-np.arange(T))
        K = np.triu(toeplitz(-q), 1)

    J_Cw = J['w']['C'][:T2,:T2]
    J_Cr = J['r']['C'][:T2, :T2]

    # expressions simple if passive monetary policy
    if p['phi'] == 1.0:
        if M is None:
            A = K @ (np.eye(T) - J['w']['C']) # Asset Jacobian
            M = np.linalg.solve(A[:T2,:T2], K[:T2,:T2])

        J_YG   = M         # jacobian of Y wrt G
        J_YT   = -M @ J_Cw # jacobian of Y wrt T
        J_Yeps = M @ J_Cr  # jacobian of Y wrt r

        dY_dG_ge = M @ (dG - J_Cw @ dT)            # GE IRF to G shock
        dT_bal   = dr * Capital                     # when r changes, T must too to balance budget
        dY_dr_ge = M @ (J_Cr @ dr - J_Cw @ dT_bal) # GE IRF to r shock

        dY_ge = dY_dG_ge + dY_dr_ge # total GE IRF to G, T, and r shocks
        dr_ge = np.ones(T2) * dr    # effect of r shock on r

    # values differ if active monetary policy
    else:
        rho_disc = np.triu(toeplitz(np.exp(-p['rho'])**(np.arange(T))), 0) # discounting future
        Fmat     = np.eye(T, k=1) # forward matrix to match discrete time
        phiIF    = (p['phi'] * np.eye(T) - Fmat) # effect of inflation on interest rates

        # partial equilibrium inflation jacobians to contemporaneous shocks
        # shocks (in order): output, productivity, taxes, government spending, interest rate
        J_piY = rho_disc * p['kappa_w'] * ((1 + p['xi']) / p['Yss'] + p['gamma'] / C - 1/(p['Yss'] * (1 - p['T_share'])))
        J_piZ = -rho_disc * p['kappa_w'] * (1 + p['xi']) / p['AgZ']
        J_piT = rho_disc * p['kappa_w'] * 1/(p['Yss'] * (1 - p['T_share']))
        J_piG = -rho_disc * p['kappa_w'] * p['gamma'] / C
        J_pir = np.zeros(T)

        # intermediate jacobians used in GE jacobians later
        JrYT = np.linalg.inv(np.eye(T) - phiIF @ J_pir) @ phiIF
        JrYG = JrYT @ J_piY
        J_Tr = np.eye(T) * Capital # taxes change to interest rate shock to keep assets constant

        # if not passed in curlyM, calculate accounting for effect on interest rates
        if M is None:
            A = K @ (np.eye(T) - J['w']['C'] - J['r']['C'] @ JrYG + J['w']['C'] @ J_Tr @ JrYG) # Asset Jacobian
            M = np.linalg.solve(A[:T2,:T2], K[:T2,:T2])

        # GE Jacobians of output and interest rates to government spending shock
        J_YG = M @ (np.eye(T) + J['r']['C'] @ JrYT @ J_piG - J['w']['C'] @ J_Tr @ JrYT @ J_piG)[:T2,:T2]
        J_rG = JrYG[:T2,:T2] @ J_YG + JrYT[:T2,:T2] @ J_piG[:T2,:T2]

        # GE Jacobians of output and interest rates to tax shock
        J_YT = M @ (J['r']['C'] @ JrYT @ J_piT - J['w']['C'] - J['w']['C'] @ J_Tr @ JrYT @ J_piT)[:T2,:T2]
        J_rT = JrYT[:T2,:T2] @ (J_piY[:T2,:T2] @ J_YT + J_piT[:T2,:T2])

        # GE Jacobians of output and interest rates to productivity shock
        J_YZ = M @ (J['r']['C'] @ JrYT @ J_piZ)[:T2,:T2]
        J_rZ = JrYT[:T2,:T2] @ (J_piY[:T2,:T2] @ J_YZ + J_piZ[:T2,:T2])
        
        # GE Jacobians of output and interest rates to interest rate shock
        JrYe   = np.linalg.inv(np.eye(T) - phiIF @ J_pir)
        J_Yeps = M @ (J['r']['C'] @ JrYe)[:T2,:T2]
        J_reps = JrYe[:T2,:T2] @ ((p['phi'] - 1) * J_piY[:T2,:T2] @ J_Yeps + np.eye(T2))

        # GE IRFs of output and interest rates to G, T, and r shocks
        dY_ge = J_YG @ dG + J_YT @ dT + J_Yeps @ dr
        dr_ge = J_rG @ dG + J_rT @ dT + J_reps @ dr

    return M, dY_ge, dr_ge

# Functions for Discrete Time
def hank_ss_r(ha_one, params, calib_ha_one, p, lb=0.0, ub=0.1, maxiter = 1000, xtol = 1e-6, backward_tol=1E-8, backward_maxit=5000, forward_tol=1E-10, forward_maxit=100_000):
    """Solve steady state of full GE model. Calibrate beta to hit target for interest rate."""
    
    calib = calib_ha_one
    calib['beta'] = params['beta']
    calib['Z'] = params['Z']

    def f(r):
        calib['r'] = r
        new_ss = ha_one.steady_state(calib, backward_tol=backward_tol, backward_maxit=backward_maxit,
                                     forward_tol=forward_tol, forward_maxit=forward_maxit)
        return new_ss['A'] - (p['T_share'] - p['G_share']) * p['Yss'] / new_ss['r']

    r, sol = opt.brentq(f, lb, ub, full_output=True, xtol=xtol, maxiter=maxiter)
    
    if not sol.converged:
        raise ValueError('Steady-state solver did not converge.')
    
    calib['r'] = r
    return ha_one.steady_state(calib), sol

# compute curlyY and curlyD (backward iteration) for each input i
def hank_step1(hh_het, outputs_dt, T, differentiable_backward_fun,
    differentiable_hetinputs, differentiable_hetoutputs, law_of_motion, exog_by_output, inputs):
    curlyYs, curlyDs = {}, {}
    for i in inputs:
        curlyYs[i], curlyDs[i] = hh_het.backward_fakenews(i, outputs_dt, T, differentiable_backward_fun, \
            differentiable_hetinputs, differentiable_hetoutputs, \
            law_of_motion, exog_by_output)

    return curlyYs, curlyDs

def hank_step2(hh_het, ss, T, law_of_motion, outputs_dt):
    curlyPs = {}
    for o in outputs_dt:
        curlyPs[o] = hh_het.expectation_vectors(ss[o], T-1, law_of_motion)
    return curlyPs

# steps 3-4 of fake news algorithm
# make fake news matrix and Jacobian for each outcome-input pair
def hank_step3(curlyYs, curlyDs, curlyPs, outputs_dt, inputs):
    F = {}
    for o in outputs_dt:
        for i in inputs:
            if o.upper() not in F:
                F[o.upper()] = {}
            F[o.upper()][i] = HetBlock.build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
    return F

def hank_step4(F, outputs_dt, inputs):
    J = {}
    for o in outputs_dt:
        for i in inputs:
            if o.upper() not in J:
                J[o.upper()] = {}
            J[o.upper()][i] = HetBlock.J_from_F(F[o.upper()][i])
    return J

def getM_dt(r, T, T2, A):
    K = Kmat(r, T)
    A = K @ A
    curlyMs = np.linalg.solve(A[:T2,:T2], K[:T2,:T2])
    return curlyMs

def GE_Jacs_dt(J, dG, dT, dr, T, T2, p, r, C, Capital, curlyMs = None):
    if p['phi'] == 1:
        if curlyMs is None:
            curlyMs = getM(r, T, T, J['A']['Z'], A = J['A']['Z'])

        # Get GE IRFs to G,T and r shock
        dY_dG_DT = curlyMs[:T2,:T2] @ (dG - J['C']['Z'][:T2,:T2] @ dT)

        dT_bal = dr * Capital
        dY_dr_DT = curlyMs[:T2,:T2] @ (J['C']['r'][:T2,:T2] @ dr - J['C']['Z'][:T2,:T2] @ dT_bal)

        dY = dY_dG_DT + dY_dr_DT
        dr = np.ones(T2) * dr

    # Get correct discrete-time formulas if phi > 1
    ## Not set up for r shock
    else:
        Fmat = np.eye(T, k=1)
        phiIF = (p['phi'] * np.eye(T) - Fmat)

        kap_denom = (1 + p['Yss'] * p['gamma'] / (p['xi'] * C) - p['T_share'] / (p['xi'] * (1 - p['T_share'])))
        kap_tilde = p['kappa_w'] / (p['Yss'] / p['xi']) * kap_denom
        Kw = kap_tilde * np.triu(toeplitz(p['beta']**(np.arange(T))), 0)
        J_Tr = np.eye(T) * Capital # Capital is Bss here
        dY_inv = np.eye(T) - J['C']['Z'] - (J['C']['r'] - J['C']['Z'] @ J_Tr) @ phiIF @ Kw

        if curlyMs is None:
            curlyMs = getM_dt(r, T, T2, dY_inv)
        dYn_ge_DT = (p['Yss']/C * p['gamma'] / p['xi'] / kap_denom * dG - dT / (kap_denom * (1 - p['T_share']) * p['xi']))
        dY_dG_DT = curlyMs[:T2,:T2] @ (dG - J['C']['Z'][:T2,:T2] @ dT - ((J['C']['r'] - J['C']['Z'] @ J_Tr) @ phiIF @ Kw)[:T2,:T2] @ dYn_ge_DT)
        dr_dG_DT = (phiIF @ Kw)[:T2,:T2] @ (dY_dG_DT - dYn_ge_DT[:T2])

        dY = dY_dG_DT
        dr = dr_dG_DT

    return curlyMs, dY, dr


