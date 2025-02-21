# Thie file includes the functions to calculate stady-state in continuous time
import numpy as np
from toolkit import utils
import scipy.optimize as opt
import scipy.sparse as sp
from sequence_jacobian import utilities as ut
import scipy.sparse.linalg as spla

def solve_steady_state(p, n, fn = None, EGM = False, HANK = False):
    """Solve for the steady state.

    Args:
        p: Dict giving the parameters of the model.
        n: Dict giving the numerical parameters (eg. grid size, etc.)
        fn: Function that iterates backwards for EGM
        EGM: False for implicit method. True for EGM
        HANK: True for HANK, False for HA
    Returns:
        ss: A dictionary of the steady state values.
    """
    if EGM:
        return steady_state_EGM(p, n, fn, HANK = HANK)
    elif HANK:
        return steady_state_HJB_HANK(p, n, HANK = HANK)
    else:
        return steady_state_HJB_HA(p, n, HANK = HANK)

def steady_state_EGM(p, n, fn = None, HANK = False):
    """Solve for the steady state via endogenous gridpoint method.

    Args:
        p: Dict giving the parameters of the model.
        n: Dict giving the numerical parameters (eg. grid size, etc.)
        fn: Function that iterates backwards for EGM
        HANK: True for HANK, False for HA
    Returns:
        ss: A dictionary of the steady state values.
    """

    g_exog = utils.stationary(n['Pi'])                    # stationary distribution of exogenous states
    endog_uniform = np.full(n['Na'], 1/n['Na'])           # uniform distribution of endogenous states
    Dbeg = ut.multidim.outer([g_exog] + [endog_uniform])  # combine to get initial distribution
    Pi_T = n['Pi_T']                                      # transpose of prod. transition matrix, used in forward iteration

    # Function that optimizer will call for each guess of r to check market clearing
    def r_iteration_helper(r):
        # express wage as a function of current interest rate guess for HA
        if not HANK:
            alpha = p['alpha']
            w = alpha**(alpha / (1 - alpha)) * (1 - alpha) * p['AgZ']**(1 / (1 - alpha)) * (r + p['d'])**(-alpha / (1 - alpha)) * p['N']
        else:
            w = (1 - p['T_share']) * p['Yss']

        Ra = r * n['aa'] # interest income

        # test borrowing constraint binds
        if w * p['z'][0] + r * p['amin'] < 0:
            print('CAREFUL: borrowing constraint too loose')

        income              = w * n['zz'] + Ra            # total income
        dV_Upwind           = backward_init(p, n, income) # initialize value function derivative
        a_ind, sav_wt, c, a = backward_steady_state(dV_Upwind, p, n, w, r, fn)   # iterate backward to get policy functions
        # iterate forward to get distribution
        # discrete-time HANK code checks convergence on endogenous part only so
        ## for consistency in runtimes, we do the same here for HANK
        ## for HA, we check convergence on the full distribution as in discrete-time code
        if HANK:
            D = forward_steady_state_endog(Dbeg, n, Pi_T, a_ind, sav_wt)
        else:
            D = forward_steady_state(Dbeg, n, Pi_T, a_ind, sav_wt)

        # capital supply
        KS = np.sum(D * n['aa'])

        # capital demand
        if HANK:
            KD = (p['T_share'] - p['G_share']) * p['Yss'] / r
        else:
            KD = (alpha * p['AgZ'] / (r + p['d']))**(1 / (1 - alpha)) * p['AgZ'] * p['N']

        # net savings
        Sav = KS - KD

        return Sav, D, c, a_ind, sav_wt, w, a

    def r_iteration(r):
        Sav, _, _, _, _, _, _ = r_iteration_helper(r)
        return Sav

    # Brent's method to solve for r that clears the asset market
    r, sol = opt.brentq(r_iteration, n['rmin'], n['rmax'], full_output = True, xtol = n['crit_S'], maxiter = n['Ir'])

    if not sol.converged:
        raise ValueError('Steady-state solver did not converge.')

    # get distribution, policy function, wage at optimal 
    _, D, c, a_ind, sav_wt, w, a = r_iteration_helper(r)

    # Assign output
    ss = {}
    ss['c']      = c      # steady-state consumption
    ss['gm']     = D      # steady-state distribution
    ss['a_ind']  = a_ind  # index of asset gridpoint agent moves to
    ss['sav_wt'] = sav_wt # weight on that gridpoint vs. the next one
    ss['w']      = w      # steady-state wage
    ss['r']      = r      # steady-state interest rate
    ss['true_a'] = a      # steady-state choice of assets

    return ss


def steady_state_HJB_HANK(p, n, HANK = True):
    """Solve for the steady state for the HANK model using the implicit method.
    For each guess of r, it re-initializes the value function.

    Args:
        p: Dict giving the parameters of the model.
        n: Dict giving the numerical parameters (eg. grid size, etc.)
        HANK: True for HANK, False for HA
    Returns:
        ss: A dictionary of the steady state values.
    """

    # bisection bounds
    rmin = n['rmin']
    rmax = n['rmax']

    # normalization of right-hand-side of KFE for inversion
    gRHS            = np.zeros(n['Ntot'])
    gRHS[n['ifix']] = 1
    gRow            = np.zeros(n['Ntot'])
    gRow[n['ifix']] = 1

    # initialization
    r = (rmin + rmax) / 2 # interest rate

    # In HANK, wage is a function of steady-state objects so can be defined here
    if HANK:
        w = (1 - p['T_share']) * p['Yss']

    # loop over r
    for ir in range(n['Ir']):
        # Express wage as a function of current interest rate guess
        if not HANK:
            alpha = p['alpha']
            w = alpha**(alpha / (1 - alpha)) * (1 - alpha) * p['AgZ'] ** (1 / (1 - alpha)) * (r + p['d'])**(-alpha / (1 - alpha)) * p['N']
        Ra = r * n['aa']

        if w * p['z'][0] + r * p['amin'] < 0:
            print('CAREFUL: borrowing constraint too loose')

        # Initializations
        dVf = np.zeros((n['Nz'], n['Na']))
        dVb = np.zeros((n['Nz'], n['Na']))
        c   = np.zeros((n['Nz'], n['Na']))

        # Initial guess
        if p['gamma'] != 1:
            v = ((w * n['zz'] + Ra) ** (1 - p['gamma'])) / (1 - p['gamma']) / p['rho']
        else:
            v = np.log(w * n['zz'] + Ra) / p['rho']

        # converge value function
        for j in range(n['back_maxit']):
            V = v.copy()

            # forward difference
            dVf[:, :-1] = (V[:, 1:] - V[:, :-1]) / n['da'][np.newaxis, :-1]
            dVf[:, -1]  = (w * p['z'] + Ra[:, -1]) ** (-p['gamma'])

            # backward difference
            dVb[:, 1:] = (V[:, 1:] - V[:, :-1]) / n['da'][np.newaxis, :-1]
            dVb[:, 0]  = (w * p['z'] + Ra[:, 0]) ** (-p['gamma'])

            # consumption and savings with forward difference
            cf = dVf ** (-1 / p['gamma'])
            sf = w * n['zz'] + Ra - cf

            # consumption and savings with backward difference
            cb = dVb ** (-1 / p['gamma'])
            sb = w * n['zz'] + Ra - cb

            # consumption and derivative of value function at steady state
            c0  = w * n['zz'] + Ra
            dV0 = c0 ** (-p['gamma'])

            # indicators for upwind savings rate
            If = sf > 0
            Ib = sb < 0
            I0 = 1 - If - Ib

            # consumption
            dV_Upwind = dVf * If + dVb * Ib + dV0 * I0
            c         = dV_Upwind ** (-1 / p['gamma'])

            # utility
            if p['gamma'] != 1:
                u = (c ** (1 - p['gamma'])) / (1 - p['gamma'])
            else:
                u = np.log(c)

            # savings finite difference matrix
            Sb = -np.minimum(sb, 0) / n['daa']
            Sm = -np.maximum(sf, 0) / n['daa'] + np.minimum(sb, 0) / n['daa']
            Sf = np.maximum(sf, 0) / n['daa']

            S = sp.diags(Sm.ravel(), 0, shape=(n['Ntot'], n['Ntot'])) + \
                sp.diags(Sb.ravel()[1:], -1, shape=(n['Ntot'], n['Ntot'])) + \
                sp.diags(Sf.ravel()[:-1], 1, shape=(n['Ntot'], n['Ntot']))

            # tests
            if np.max(np.abs(S.sum(axis=1))) > 1e-5:
                print('Improper savings transition matrix')

            M = n['M1'] - S

            # invert linear system
            V = spla.spsolve(M, u.ravel() + V.ravel() / n['Delta'])
            V = V.reshape((n['Nz'], n['Na']))

            # update
            Vchange = V - v
            v = V

            dist = np.max(np.abs(Vchange))
            if dist < n['back_tol']:
                break

        # distribution
        LT = S.transpose() + n['Ly_T']

        # normalize transition matrix row
        data    = LT.data
        offsets = LT.offsets
        for i, offset in enumerate(offsets):
            if offset == 0:
                data[i, n['ifix']] = 1
            elif n['ifix'] + offset >= 0:
                data[i, n['ifix']] = 0

        LT = LT.tocsr() # make sparse for fast inversion
        # solve linear system
        g = np.maximum(np.real(spla.spsolve(LT, gRHS)), 0)
        g = g / np.sum(g * n['daa'].flatten())

        # capital supply
        KS = np.sum(g * n['aa'].flatten() * n['daa'].flatten())

        # capital demand
        if HANK:
            KD = (p['T_share'] - p['G_share']) * p['Yss'] / r
        else:
            KD = (alpha * p['AgZ'] / (r + p['d']))**(1 / (1 - alpha)) * p['AgZ'] * p['N']

        # net savings
        Sav = KS - KD

        # update interest rate according to bisection
        if Sav > n['crit_S']:
            rmax = r
            r = 0.5 * (r + rmin)
        elif Sav < -n['crit_S']:
            rmin = r
            r = 0.5 * (r + rmax)
        elif abs(Sav) < n['crit_S']:
            break
            
        if ir == n['Ir'] - 1:
            print("\nCould not find steady-state, r =", r)
            raise ValueError("Could not find steady-state")

    # recompute prices for consistency
    if HANK:
        r = (p['T_share'] - p['G_share']) * p['Yss'] / np.sum(g * (n['aa'] * n['daa']).flatten())
    else:
        alpha = p['alpha'] # capital share
        r = -p['d'] + alpha * p['AgZ'] * max(np.dot(n['aa'].flatten(), g * n['daa'].flatten()), 1e-5)**(alpha - 1) * p['AgZ']**(1 - alpha)
        w = (1 - p['alpha']) * p['AgZ'] * (p['alpha'] * p['AgZ'] / (r + p['d']))**(p['alpha'] / (1 - p['alpha'])) * p['N']

    # Assign output
    ss       = {}
    ss['V']  = V  # steady-state value function
    ss['c']  = c  # steady-state consumption
    ss['sf'] = sf # steady-state savings if positive
    ss['sb'] = sb # steady-state savings if negative
    ss['gm'] = g.reshape((n['Nz'], n['Na'])) * n['daa'] # steady-state distribution mass
    ss['L']  = LT.transpose() # transition for value 
    ss['LT'] = LT # transpose of transition for distribution iteration
    ss['S']  = S  # savings transition matrix
    ss['w']  = w  # steady-state wage
    ss['r']  = r  # steady-state interest rate

    return ss

def steady_state_HJB_HA(p, n, HANK = False):
    """Solve for the steady state for the HA model using the implicit method.
    For each guess of r, it uses the value function from the last guess as the initialization.

    Args:
        p: Dict giving the parameters of the model.
        n: Dict giving the numerical parameters (eg. grid size, etc.)
        HANK: True for HANK, False for HA
    Returns:
        ss: A dictionary of the steady state values.
    """

    # bisection bounds
    rmin = n['rmin']
    rmax = n['rmax']

    # normalization of right-hand-side of KFE for inversion
    gRHS            = np.zeros(n['Ntot'])
    gRHS[n['ifix']] = 1
    gRow            = np.zeros(n['Ntot'])
    gRow[n['ifix']] = 1

    # Initializations
    dVf = np.zeros((n['Nz'], n['Na'])) # forward derivative
    dVb = np.zeros((n['Nz'], n['Na'])) # backward derivative
    c   = np.zeros((n['Nz'], n['Na'])) # consumption
    r   = (rmin + rmax) / 2            # interest rate

    # Express wage as a function of current interest rate guess
    if HANK:
        w = (1 - p['T_share']) * p['Yss'] # post-tax income
    else:
        alpha = p['alpha'] # capital share
        w = alpha ** (alpha / (1 - alpha)) * (1 - alpha) * p['AgZ']**(1 / (1 - alpha)) * (r + p['d'])**(-alpha / (1 - alpha)) * p['N']

    Ra = r * n['aa'] # interest income

    # Initial guess for value assuming agents consume income and don't transition
    if p['gamma'] != 1:
        v = ((w * n['zz'] + Ra)**(1 - p['gamma'])) / (1 - p['gamma']) / p['rho']
    else:
        v = np.log(w * n['zz'] + Ra) / p['rho']

    # loop over r
    for ir in range(n['Ir']):
        # Express wage as a function of current interest rate guess
        if not HANK:
            alpha = p['alpha']
            w = alpha**(alpha / (1 - alpha)) * (1 - alpha) * p['AgZ']**(1 / (1 - alpha)) * (r + p['d'])**(-alpha / (1 - alpha)) * p['N']
        Ra = r * n['aa']

        if w * p['z'][0] + r * p['amin'] < 0:
            print('CAREFUL: borrowing constraint too loose')

        # converge value function
        for j in range(n['back_maxit']):
            V = v.copy()

            # forward difference
            dVf[:, :-1] = (V[:, 1:] - V[:, :-1]) / n['da'][np.newaxis, :-1]
            dVf[:, -1]  = (w * p['z'] + Ra[:, -1]) ** (-p['gamma'])

            # backward difference
            dVb[:, 1:] = (V[:, 1:] - V[:, :-1]) / n['da'][np.newaxis, :-1]
            dVb[:, 0]  = (w * p['z'] + Ra[:, 0]) ** (-p['gamma'])

            # consumption and savings with forward difference
            cf = dVf ** (-1 / p['gamma'])
            sf = w * n['zz'] + Ra - cf

            # consumption and savings with backward difference
            cb = dVb ** (-1 / p['gamma'])
            sb = w * n['zz'] + Ra - cb

            # consumption and derivative of value function at steady state
            c0  = w * n['zz'] + Ra
            dV0 = c0 ** (-p['gamma'])

            # indicators for upwind savings rate
            If = sf > 0
            Ib = sb < 0
            I0 = 1 - If - Ib

            # consumption
            dV_Upwind = dVf * If + dVb * Ib + dV0 * I0
            c         = dV_Upwind ** (-1 / p['gamma'])

            # utility
            if p['gamma'] != 1:
                u = (c ** (1 - p['gamma'])) / (1 - p['gamma'])
            else:
                u = np.log(c)

            # savings finite difference matrix
            Sb = -np.minimum(sb, 0) / n['daa']
            Sm = -np.maximum(sf, 0) / n['daa'] + np.minimum(sb, 0) / n['daa']
            Sf = np.maximum(sf, 0) / n['daa']

            S = sp.diags(Sm.ravel(), 0, shape=(n['Ntot'], n['Ntot'])) + \
                sp.diags(Sb.ravel()[1:], -1, shape=(n['Ntot'], n['Ntot'])) + \
                sp.diags(Sf.ravel()[:-1], 1, shape=(n['Ntot'], n['Ntot']))

            # tests
            if np.max(np.abs(S.sum(axis=1))) > 1e-5:
                print('Improper savings transition matrix')

            M = n['M1'] - S

            # invert linear system
            V = spla.spsolve(M, u.ravel() + V.ravel() / n['Delta'])
            V = V.reshape((n['Nz'], n['Na']))

            # update
            Vchange = V - v
            v = V

            dist = np.max(np.abs(Vchange))
            if dist < n['back_tol']:
                break

        # distribution
        LT = S.transpose() + n['Ly_T']

        # normalize transition matrix row
        data    = LT.data
        offsets = LT.offsets
        for i, offset in enumerate(offsets):
            if offset == 0:
                data[i, n['ifix']] = 1
            elif n['ifix'] + offset >= 0:
                data[i, n['ifix']] = 0

        LT = LT.tocsr() # make sparse for fast inversion
        # solve linear system
        g = np.maximum(np.real(spla.spsolve(LT, gRHS)), 0)
        g = g / np.sum(g * n['daa'].flatten())

        # capital supply
        KS = np.sum(g * n['aa'].flatten() * n['daa'].flatten())

        # capital demand
        if HANK:
            KD = (p['T_share'] - p['G_share']) * p['Yss'] / r
        else:
            KD = (alpha * p['AgZ'] / (r + p['d']))**(1 / (1 - alpha)) * p['AgZ'] * p['N']

        # net savings
        Sav = KS - KD

        # update interest rate according to bisection
        if Sav > n['crit_S']:
            rmax = r
            r = 0.5 * (r + rmin)
        elif Sav < -n['crit_S']:
            rmin = r
            r = 0.5 * (r + rmax)
        elif abs(Sav) < n['crit_S']:
            break
            
        if ir == n['Ir'] - 1:
            print("\nCould not find steady-state, r =", r)
            raise ValueError("Could not find steady-state")

    # recompute prices for consistency
    if HANK:
        r = (p['T_share'] - p['G_share']) * p['Yss'] / np.sum(g * (n['aa'] * n['daa']).flatten())
    else:
        alpha = p['alpha'] # capital share
        r = -p['d'] + alpha * p['AgZ'] * max(np.dot(n['aa'].flatten(), g * n['daa'].flatten()), 1e-5)**(alpha - 1) * p['AgZ']**(1 - alpha)
        w = (1 - p['alpha']) * p['AgZ'] * (p['alpha'] * p['AgZ'] / (r + p['d']))**(p['alpha'] / (1 - p['alpha'])) * p['N']

    # Assign output
    ss       = {}
    ss['V']  = V  # steady-state value function
    ss['c']  = c  # steady-state consumption
    ss['sf'] = sf # steady-state savings if positive
    ss['sb'] = sb # steady-state savings if negative
    ss['gm'] = g.reshape((n['Nz'], n['Na'])) * n['daa'] # steady-state distribution mass
    ss['L']  = LT.transpose() # transition for value 
    ss['LT'] = LT # transpose of transition for distribution iteration
    ss['S']  = S  # savings transition matrix
    ss['w']  = w  # steady-state wage
    ss['r']  = r  # steady-state interest rate

    return ss

def backward_init(p, n, income):
    # Initialize V, dV
    dVf = np.zeros((n['Nz'], n['Na']))
    dVb = np.zeros((n['Nz'], n['Na']))

    # Initial guess
    if p['gamma'] != 1:
        v = (income**(1 - p['gamma'])) / (1 - p['gamma']) / p['rho']
    else:
        v = np.log(income) / p['rho']

    # To calculate the steady state, discrete time asks for the backward iteration function.
    ## We do the same but first, need to calculate dV_{t+1}/da. Discrete time applies
    ## forward differencing but in continuous time, it is more stable to use upwinding.

    # Converge value function
    # Forward difference
    dVf[:, :-1] = (v[:, 1:] - v[:, :-1]) / n['da'][np.newaxis, :-1]
    dVf[:, -1]  = (income[:, -1])**(-p['gamma'])

    # Backward difference
    dVb[:, 1:] = (v[:, 1:] - v[:, :-1]) / n['da'][np.newaxis, :-1]
    dVb[:, 0]  = (income[:, 0])**(-p['gamma'])

    # Consumption and savings with forward difference
    cf = dVf ** (-1 / p['gamma'])
    sf = income - cf

    # Consumption and savings with backward difference
    cb = dVb ** (-1 / p['gamma'])
    sb = income - cb

    # Consumption and derivative of value function at steady state
    c0  = income
    dV0 = c0 ** (-p['gamma'])

    # Indicators for upwind savings rate
    If = sf > 0
    Ib = sb < 0
    I0 = 1 - If - Ib

    # Consumption
    dV_Upwind = dVf * If + dVb * Ib + dV0 * I0

    return dV_Upwind

def backward_steady_state(dV_Upwind, p, n, w, r, model_backward):
    # Iterate backward until convergence to choice of policy
    for j in range(n['back_maxit']):
        exp_Va          = n['Pi'] @ dV_Upwind
        dV_Upwind, a, c = model_backward(exp_Va, n['a'], w * p['z'], r, np.exp(-p['rho']), p['gamma'])

        if j % 10 == 1 and utils.within_tolerance(old_a, a, n['back_tol']):
            break
    
        old_a = a.copy()

    # Interpolate policy rule for assets
    a_ind, sav_wt = utils.interpolate_coord_robust(n['a'], a)

    return a_ind, sav_wt, c, a

def forward_steady_state(D, n, Pi_T, a_ind, sav_wt):
    """Iterate distribution forward until convergence.

    Args:
        D: Initial distribution
        n: Dict giving the numerical parameters (eg. grid size, etc.)
        Pi_T: Transpose of the exogenous state transition matrix
        a_ind: Index of endogenous state gridpoint agent moves to
        sav_wt: Weight on that gridpoint. Weight on next gridpoint is 1 - sav_wt
    Returns:
        D: Distribution after convergence
    """
    for it in range(n['fwd_maxit']):
        Dnew = utils.forward_step_1d(D, Pi_T, a_ind, sav_wt)
        if it % 10 == 0 and utils.within_tolerance(D, Dnew, n['fwd_tol']):
            break
        D = Dnew
    else:
        print("Max iterations reached")

    return D

# Check only endogenous part for convergence to align with
## discrete-time SSJ code for HANK
def forward_steady_state_endog(D_endog, n, Pi_T, a_ind, sav_wt):
    """Iterate distribution forward until convergence.
        Check for convergence depends on endogenous part.

    Args:
        D: Initial distribution
        n: Dict giving the numerical parameters (eg. grid size, etc.)
        Pi_T: Transpose of the exogenous state transition matrix
        a_ind: Index of endogenous state gridpoint agent moves to
        sav_wt: Weight on that gridpoint. Weight on next gridpoint is 1 - sav_wt
    Returns:
        D: Distribution after convergence
    """
    D = Pi_T @ D_endog
    for it in range(n['fwd_maxit']):
        D_endog_new = utils.forward_step_1d_endog(D, a_ind, sav_wt)
        D = Pi_T @ D_endog_new
        if it % 10 == 0 and utils.within_tolerance(D_endog_new, D_endog, n['fwd_tol']):
            break
        D_endog = D_endog_new
    else:
        print("Max iterations reached")

    return D

def prelim_step(ss, n, EGM = True, dt = 1.0):
    """Calculate necessary values for Jacobian calculations after getting the steady state.
    
    Args:
        ss: Dict giving the steady state values.
        n: Dict giving the numerical parameters (eg. grid size, etc.)
        EGM: False if implicit method used to get SS. True for EGM
        dt: Time step
    """

    # get which constrained
    ss['consted']     = (n['aa'] == n['a'][0]) * ((ss['r'] * n['aa'] + n['zz'] * ss['w'] - ss['c']) < 1e-3) # boolean matrix of constrained agents
    ss['consted_ind'] = np.where(ss['consted'].ravel())[0] # indices of constrained agents

    ss['Pi'] = n['Pi'] # used for distribution iteration if iter_style is DT_loop
    ss['ly'] = n['ly'] # used for distribution iteration if iter_style is not DT_loop

    # get asset derivative matrix with upwinding scheme

    # get matrix transposed w/ s treated as 1 to get upwinding Da
    if EGM:
        sav = ss['r'] * n['aa'] + ss['w'] * n['zz'] - ss['c']
        Sf1 = (np.minimum(sav, 0) < 0) / n['daa']
        Sm1 = ((np.maximum(sav, 0) > 0).astype(int) - (np.minimum(sav, 0) < 0).astype(int)) / n['daa']
        Sb1 = -((np.maximum(sav, 0) > 0) / n['daa'])
    else:
        Sf1 = (np.minimum(ss['sb'], 0) < 0) / n['daa']
        Sm1 = ((np.maximum(ss['sf'], 0) > 0).astype(int) - (np.minimum(ss['sb'], 0) < 0).astype(int)) / n['daa']
        Sb1 = -((np.maximum(ss['sf'], 0) > 0) / n['daa'])

    # construct d/da matrix for distribution
    ss['DA_T'] = sp.diags(Sm1.flatten(), 0, shape = (n['Ntot'], n['Ntot'])) + \
                        sp.diags(Sb1.flatten()[:-1], -1, shape = (n['Ntot'], n['Ntot'])) + \
                        sp.diags(Sf1.flatten()[1:], 1, shape =(n['Ntot'], n['Ntot']))

    ss['DA'] = -ss['DA_T'].transpose() # construct d/da matrix for values

    # get asset index agent moving to via saving for later use

    # Need to get a_ind and sav_wt if not EGM for iterating distribution forward
    ## in phi and cE calculations
    if not EGM:
        # Get lhs_inv for phi calculation if using calc_phi_HJB.
        lhs_inv = sp.eye(n['Ntot'], format = "csc")
        lhs_inv[ss['consted_ind'], ss['consted_ind'] + 1] = 1
        M_L = sp.eye(n['Ntot'], format = "csc") + ss['L']
        M_L[ss['consted_ind'], :] = 0
        ss['lhs_mat'] = lhs_inv @ M_L

        # get asset index agent moving to via saving for later use
        sav_sign = np.sign(Sm1)                              # get whether saving/dissaving
        S_da     = -ss['S'].diagonal() / n['mu'] * dt        # amount saved at each index
        S_pts    = S_da.reshape(n['Nz'], n['Na']) * sav_sign # multiply change in gridpts by savings sign
        a_ind    = np.floor(S_pts).astype(int)               # get integer number of gridpts moved 
        sav_wt   = 1 - np.mod(S_pts, 1)                      # interpolation weight on left gridpt (lower weight if far from floor)

        ss['a_ind']  = a_ind + np.arange(n['Na']) # get index of asset gridpoint moving to, not change in index
        ss['sav_wt'] = sav_wt
    else:
        S_pts = (ss['true_a'] - n['aa']) / n['daa'] / n['mu'] * dt # asset gridpts moved at each index    
        ss['a_ind'] = np.floor(S_pts).astype(int) + np.arange(n['Na'])
        ss['sav_wt'] = 1 - np.mod(S_pts, 1)

    # adjust a_ind and sav_wt so don't go out of grid
    out_of_bounds_high               = ss['a_ind'] >= n['Na'] - 1      # indices saving outside of grid
    ss['sav_wt'][out_of_bounds_high] = 0                               # move agents to last gridpt if out of bounds
    ss['a_ind'][out_of_bounds_high]  = n['Na'] - 2     

    out_of_bounds_low               = ss['a_ind'] < 0                  # indices saving outside of grid
    ss['sav_wt'][out_of_bounds_low] = 1                                # move agents to last gridpt if out of bounds
    ss['a_ind'][out_of_bounds_low]  = 0
    ss['a_ind']                     = ss['a_ind'] - np.arange(n['Na']) # convert back to change in gridpts

    # adjust mu at gridpts where savings would push agents out of
    # bounds to ensure correct mass moving to endpoint
    mu_mat = np.ones((n['Nz'], n['Na'])) * n['mu']         # define a separate mass leaving for each gridpt

    # saving too much
    pts_moved                  = (n['amax'] - n['aa'][out_of_bounds_high])                # gridpts actually moved
    pts_supposed               = S_pts[out_of_bounds_high] * n['daa'][out_of_bounds_high] # gridpts supposed to move
    pts_moved                  = np.maximum(pts_moved, 1e-12)                             # pts_moved can be 0 if starting at max gridpt; then, set mass to 1
    mu_mat[out_of_bounds_high] = np.minimum(pts_supposed / pts_moved * n['mu'], 1)        # increase mass leaving so correct transition maintained

    # repeat for case when dissaving too much
    pts_moved                 = n['aa'][out_of_bounds_low] - n['a'][0]                  # gridpts actually moved
    pts_supposed              = -S_pts[out_of_bounds_low] * n['daa'][out_of_bounds_low] # gridpts supposed to move
    mu_mat[out_of_bounds_low] = np.minimum(pts_supposed / pts_moved * n['mu'], 1)       # increase mass leaving so correct transition maintained

    # convert a_ind and sav_wt into a transition matrix
    # akin to LT but can move multiple asset gridpts for stability and rows sum to 1, not 0
    sav_data = np.concatenate([(ss['sav_wt'] * mu_mat).ravel(), ((1 - ss['sav_wt']) * mu_mat).ravel()]) # multiply by mu so only fraction moves
    sav_rows = np.concatenate([np.arange(n['Ntot']), np.arange(n['Ntot'])])
    cols     = range(n['Ntot']) + ss['a_ind'].ravel()
    sav_cols = np.concatenate([cols, cols + 1])
    S_stable = sp.csr_matrix((sav_data, (sav_rows, sav_cols)), shape = (n['Ntot'], n['Ntot']))

    ss['S_stable'] = (S_stable - sp.diags(mu_mat.ravel())) / dt # subtract mass leaving
    ss['a_ind']    = ss['a_ind'] + np.arange(n['Na'])           # get index of asset gridpoint moving to, not change in index

    # write S_stable as numpy array for compatibility with numba; needed if iter_style is CT_loop
    ss['S_npy'] = np.vstack((sav_data[:n['Ntot']], sav_data[n['Ntot']:], -mu_mat.ravel())).T / dt

    return ss
