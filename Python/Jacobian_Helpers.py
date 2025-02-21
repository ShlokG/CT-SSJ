# This file includes key functions for continuous-time SSJ
## that the user can call
import numpy as np
from scipy.linalg import solve
from numba import njit
import numexpr as ne

@njit
def forward_step_transpose_1d(D, ly, x_i, S_npy, dt = 1.0):
    """Transpose of distribution iteration with ly = Pi - Id

    Args:
        D: Array to be iterated on (Nz x Na array)
        ly: Change in exogenous state transition matrix (Nz x Nz matrix). Rows sum to 0
        x_i: Index in endogenous state space of where agents are moving to at each gridpoint (Nz x Na array)
        S_npy: Nz*Na x 3 array of the 3 probabilities of moving to x_i, x_i + 1, and i where i is current index at each gridpoint
            Rows sum to 0 since it accounts for moving away from i
        dt: Time step
        
    Returns:
        Dnew: Array after iterating once as in continuous time (Ne x Na array)
    """
    # add value from where agents are moving to via endogenous policy, adjusted for dt
    Nz, Na = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(Nz):
        for ix in range(Na):
            i = iz * Na + ix
            a_left = x_i[iz, ix]
            Dnew[iz, ix] = S_npy[i, 0] * D[iz, a_left] + S_npy[i, 1] * D[iz, a_left + 1] + S_npy[i, 2] * D[iz, ix]

    return D + dt * (ly @ D + Dnew) # update using exogenous and endogenous transition

def forward_step_matrix(D, ly, S_stable, dt = 1.0):
    """Transpose of distribution iteration with sparse matrix multiplication

    Args:
        D: Array to be iterated on (Nz x Na array)
        ly: Change in exogenous state transition matrix (Nz x Nz matrix). Rows sum to 0
        S_stable: Change in endogenous state transition matrix (Nz*Na x Nz*Na matrix). Rows sum to 0
        dt: Time step
        
    Returns:
        Dnew: Array after iterating once as in continuous time (Nz x Na array)
    """
    Nz, Na = D.shape
    return D + dt * (ly @ D + (S_stable @ D.ravel()).reshape(Nz, Na))

@njit
def forward_step_transpose_1d_dt(D, Pi, x_i, x_pi):
    """Transpose of distribution iteration with Pi instead of ly = Pi - Id

    Args:
        D: Array to be iterated on (Nz x Na array)
        Pi: Exogenous state transition matrix (Nz x Nz matrix). Rows sum to 1 with non-negative elements
        x_i: Index in endogenous state space of where agents are moving to at each gridpoint (Nz x Na array)
        x_pi: Probability of moving to x_i at each gridpoint (for interpolation) at each gridpoint (Nz x Na array)
            Implies 1-x_pi is probability of moving to x_i + 1
        
    Returns:
        Dnew: Array after iterating once (Nz x Na array)
    """
    # first update using exogenous transition matrix
    D = Pi @ D

    # then update using (transpose) endogenous policy
    Nz, nX = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(Nz):
        for ix in range(nX):
            i = x_i[iz, ix]
            pi = x_pi[iz, ix]
            # add value from where agents are moving to, adjusted for dt
            Dnew[iz, ix] = pi * D[iz, i] + (1 - pi) * D[iz, i + 1]
    return Dnew

def check_if_zero(C_V):
    """Checks if given input is zero and can be applied both to scalar or array"""
    if np.isscalar(C_V):
        # C_V is a number
        return C_V == 0
    else:
        # C_V is a matrix
        return np.all(C_V == 0)

def phi_constraints(phi_t, consted, daa, C_p = 0, C_V = 0, C_dV = 0):
    """Apply the constraints in calculating phi_t

    Args:
        phi_t: Change in value for agents at x at time t from price/aggregate shock (Nz*Na x T matrix)
        consted: Boolean array indicating indices of agents that are constrained
        C_p: Derivative of constraint w.r.t. price (either float or Nz*Na x 1 vector for constrained indices only)
        C_V: Derivative of constraint w.r.t. value (either float or Nz*Na x 1 vector for constrained indices only)
        C_dV: Derivative of constraint w.r.t. change in value (either float or Nz*Na x 1 vector for constrained indices only)

    Returns:
        phi_t (after adjusting the constrained indices)
    """

    if check_if_zero(C_V) and not check_if_zero(C_dV):
        phi_t[consted] = phi_t[:, 1:][consted[:, :-1]] + daa[consted] * C_p / C_dV
    elif check_if_zero(C_dV) and not check_if_zero(C_V):
        phi_t[consted] = -C_p / C_V
    elif not (check_if_zero(C_V) or check_if_zero(C_dV)):
        phi_t[consted] = (phi_t[:, 1:][consted[:, :-1]] - daa[consted] * C_p / C_dV) / (1 + daa[consted] * C_V / C_dV) 

    return phi_t

def calc_dphi_dx(phi_t, DA_mat, endog_states):
    """Calculates dphi/dx for each x in endog_states

    Args:
        phi_t: Change in value for agents at x at time t from price/aggregate shock (Nz*Na x T matrix)
        DA_mat: Dictionary of derivative matrices w.r.t. x for numerical differentiation
        endog_states: List of endogenous states

    Returns:
        dphi_da: Derivative of phi_t w.r.t. x where given derivative matrix determines x (Nz*Na x T matrix)
    """

    dphi_da = {}
    for state in endog_states:
        dphi_da[state] = DA_mat[state] @ phi_t
    
    return dphi_da

def calc_phi(ss, n, T, dt = 1.0, L_p = None, Vss = None, phi0 = None, u_p = 0, C_p = 0, C_V = 0, C_dV = 0, price = True, iter_style = 'DT_loop'):
    """Calculates varphi for price or phi for aggregate shock, effect of shock on value

    Args:
        ss: Steady state dictionary
        n: Numerical parameters dictionary
        T: Number of time periods
        dt: Time step
        L_p: Steady state expectation matrix (mathcal{L} in text) gradient w.r.t. price (Nz*Na x Nz*Na matrix)
        Vss: Steady state value function (Nz x Na matrix or Nz*Na x 1 vector)
        phi0: Initial change in value from shock (Nz*Na x 1 vector)
            Note that if phi0 is None, L_p @ Vss is used
        u_p: Derivative of utility w.r.t. price or aggregate shock (in that case, this is u_z) (either number or Nz*Na x 1 vector)
        C_p: Derivative of constraint w.r.t. price (either float or Nz*Na x 1 vector for constrained indices only)
        C_V: Derivative of constraint w.r.t. value (either float or Nz*Na x 1 vector for constrained indices only)
        C_dV: Derivative of constraint w.r.t. change in value (either float or Nz*Na x 1 vector for constrained indices only)
        price: Whether shock is to price or aggregate shock
        iter_style: How to iterate forward in phi and E calculations
            3 options: 'CT_matrix', 'CT_loop', 'DT_loop' where 'CT' uses ly and DT uses Pi and 'matrix' does a sparse matrix multiplication while 'loop' uses a for loop with numba
        
    Returns:
        phi_t: Change in value for agents at x at time t from price/aggregate shock (Nz*Na x T matrix)
    """

    # Initialize phi
    if price and phi0 is None:
        phi0 = L_p @ Vss.flatten()
    elif phi0 is None:
        phi0 = np.zeros_like(Vss)

    phi0 += u_p # Add u_p or u_z to phi0

    # Adjust for constraints
    phi0 = phi_constraints(phi0, ss['consted'], n['daa'], C_p, C_V, C_dV)

    # Solve for phi_t by iterating forward
    phi_t          = np.zeros((n['Nz'], n['Na'], T))
    phi_t[:, :, 0] = phi0

    for t in range(T - 1):
        if iter_style == 'DT_loop':
            phi_t[:, :, t + 1] = forward_step_transpose_1d_dt(np.ascontiguousarray(phi_t[:, :, t]), n['Pi'], ss['a_ind'], ss['sav_wt'])
        elif iter_style == 'CT_loop':
            phi_t[:, :, t + 1] = forward_step_transpose_1d(np.ascontiguousarray(phi_t[:, :, t]), n['ly'], ss['a_ind'], ss['S_npy'], dt = dt)
        else:
            phi_t[:, :, t + 1] = forward_step_matrix(np.ascontiguousarray(phi_t[:, :, t]), n['ly'], ss['S_stable'], dt = dt)
        
        phi_t[:, :, t + 1] = phi_constraints(phi_t[:, :, t + 1], ss['consted'], n['daa'], 0, C_V, C_dV) # adjust constrained agents' value

    # Reshape into vector for later use
    phi_t = phi_t.reshape((n['Ntot'], T))

    return phi_t

def calc_phi_HJB(lhs_mat, n, T, L_p = None, Vss = None, phi0 = None, u_p = 0, price = True):
    """Calculates varphi for price or phi for aggregate shock, effect of shock on value,
        using a matrix for transitioning forward instead of the function.
    Note: Not used in the scripts.

    Args:
        lhs_mat: Matrix to iterate phi_t forward (faster if sparse)
        n: Numerical parameters dictionary
        T: Number of time periods
        L_p: Steady state expectation matrix (mathcal{L} in text) gradient w.r.t. price (Nz*Na x Nz*Na matrix)
        Vss: Steady state value function (Nz x Na matrix or Nz*Na x 1 vector)
        phi0: Initial change in value from shock (Nz*Na x 1 vector)
            Note that if phi0 is None, L_p @ Vss is used
        u_p: Derivative of utility w.r.t. price or aggregate shock (in that case, this is u_z) (either number or Nz*Na x 1 vector)
        price: Whether shock is to price or aggregate shock

    Returns:
        phi_t: Change in value for agents at x at time t from price/aggregate shock (Nz*Na x T matrix)
    """
    # Initialize phi
    if price and phi0 is None:
        phi0 = L_p @ Vss.flatten()
    elif phi0 is None:
        phi0 = np.zeros_like(Vss)

    phi0 += u_p # Add u_p or u_z to phi0

    # Solve for phi_t by iterating forward
    phi_t = np.zeros((n['Ntot'], T))
    phi_t[:,0] = phi0

    # Iterate forward
    for t in np.arange(1,T):
        phi_t[:,t] = lhs_mat @ phi_t[:,t-1]

    return phi_t

def calc_policy_fn(U, T, rho, gm, dt = 1.0, u_cp = None, u_cz = None, dphi_da = None, L_c = None, phi_t = None):
    """Calculates change in consumption at time 0 for each agent from anticipated price/aggregate shock at each time s

    Args:
        U: u_{cc} + L_{cc}[V^{ss}] (either Nz x Na matrix or Nz*Na x 1 vector)
        T: Number of time periods
        rho: Discount rate
        gm: Steady state distribution (either Nz x Na matrix or Nz*Na x 1 vector, same dimensions as U)
        dt: Time step
        u_cp: Derivative of utility w.r.t. consumption and price (either Nz x Na matrix or Nz*Nz x 1 vector)
            Assumed to be 0 if None or unspecified
        u_cz: Derivative of utility w.r.t. consumption and aggregate shock (either Nz x Na matrix or Nz x 1 vector)
            Assumed to be 0 if None or unspecified
        dphi_da: Sum over endogenous states x of Derivative of phi_t w.r.t. x (Nz*Na x T matrix)
            If unspecified or None, use L_c
        L_c: Steady state expectation matrix (mathcal{L} in text) gradient w.r.t. consumption (Nz*Na x Nz*Na matrix)
            If unspecified or None, use dphi_da (so assumed to be -sum_i d/da in that case)
        phi_t: Change in value from shock (varphi for price and phi for aggregate shock) (Nz*Na x T matrix)

    Returns:
        c_t: Change in consumption at time 0 from anticipated price/aggregate shock at each time t (T x 1 vector)
    """

    rho_T  = np.expand_dims(np.exp(-rho * np.arange(T) * dt), axis=0)
    gU_inv = (gm / U).ravel()[:, np.newaxis] # U^{-1} g(x)

    # if dphi_da is available, assume L_c(x)[\psi_t] = \int_s e^{-\rho s} dphi/da_s(x) p-hat_s ds
    if dphi_da is None and L_c is not None and phi_t is not None:
        dphi_da = -L_c @ phi_t
    elif dphi_da is not None:
        RuntimeError("Need to provide either dphi_da or (L_c and phi_t)")

    # calculate c_t(x) = e^{-\rho t} dphi/da_t(x) p-hat_t
    c_t = ne.evaluate('dphi_da * gU_inv * rho_T')

    if u_cp is not None:
        c_t[:, 0] -= u_cp.flatten() * gU_inv    
    if u_cz is not None:
        c_t[:, 0] -= u_cz.flatten() * gU_inv

    return c_t

def calc_policy_fn_sum(U, T, rho, gm, dt = 1.0, u_cp = None, u_cz = None, dphi_da = None, L_c = None, phi_t = None):
    """Calculates change in aggregate consumption at time 0 from anticipated price/aggregate shock at each time s

    Args:
        U: u_{cc} + L_{cc}[V^{ss}] (either Nz x Na matrix or Nz*Nz x 1 vector)
        T: Number of time periods
        rho: Discount rate
        gm: Steady state distribution (either Nz x Na matrix or Nz*Nz x 1 vector)
        dt: Time step
        u_cp: Derivative of utility w.r.t. consumption and price (either Nz x Na matrix or Nz*Nz x 1 vector)
            Assumed to be 0 if None or unspecified
        u_cz: Derivative of utility w.r.t. consumption and aggregate shock (either Nz x Na matrix or Nz x 1 vector)
            Assumed to be 0 if None or unspecified
        dphi_da: Sum over endogenous states x of Derivative of phi_t w.r.t. x (Nz*Na x T matrix)
            If unspecified or None, use L_c
        L_c: Steady state expectation matrix (mathcal{L} in text) gradient w.r.t. consumption (Nz*Na x Nz*Na matrix)
            If unspecified or None, use dphi_da (so assumed to be -sum_i d/da in that case)
        phi_t: Change in value from shock (varphi for price and phi for aggregate shock) (Nz*Na x T matrix)

    Returns:
        c_t: Change in consumption at time 0 from anticipated price/aggregate shock at each time t (T x 1 vector)
    """

    U_inv  = 1 / U.flatten() # U^{-1}
    rho_T  = np.exp(-rho * np.arange(T) * dt) # discounting future
    gm     = gm.flatten()    # g(x)

    # if dphi_da is available, assume L_c(x)[\psi_t] = \int_s e^{-\rho s} dphi/da_s(x) p-hat_s ds
    if dphi_da is not None:
        c_t = dphi_da.T @ (U_inv * gm) * rho_T
    elif L_c is not None and phi_t is not None:
        c_t = ((L_c @ phi_t).T @ (-U_inv * gm)) * rho_T
    else:
        RuntimeError("Need to provide either dphi_da or (L_c and phi_t)")

    if u_cp is not None:
        c_t[0] -= np.sum(u_cp.flatten() * gm * U_inv)
    
    if u_cz is not None:
        c_t[0] -= np.sum(u_cz.flatten() * gm * U_inv)

    return c_t


def calc_U(u_cc, L_cc = None, Vss = None):
    """Calculates change in aggregate consumption at time 0 from anticipated price/aggregate shock at each time s

    Args:
        u_cc: Second derivative of utility with respect to consumption
        L_cc: Second derivative of expectation operator with respect to consumption (assumed to be 0 if None or unspecified)
        Vss: Steady state value function (Nz x Na matrix or Nz*Na x 1 vector). Must be specified if L_cc is not None

    Returns:
        U: u_{cc} + L_{cc}[V^{ss}] (either Nz x Na matrix or Nz*Nz x 1 vector)
    """
    if L_cc is None:
        return u_cc
    
    return u_cc + L_cc @ Vss

# Calculates D, change in distribution
def calc_D(gm, rho, LT_c, U, phi, LT_p, T, dt = 1.0, L_c = None, dphi_da = None, u_cp = None, u_cz = None, c_t = None, price = True):
    """Calculates D, change in distribution from either price or aggregate shock

    Args:
        gm: Steady state distribution (either Nz x Na matrix or Nz*Nz x 1 vector)
        rho: Discount rate
        LT_c: Function for transition matrix gradient w.r.t. consumption (Nz*Na x Nz*Na matrix)
        U: u_{cc} + L_{cc}[V^{ss}] (either Nz x Na matrix or Nz*Nz x 1 vector)
        phi: Change in value from shock (varphi for price and phi for aggregate shock) (Nz*Na x T matrix)
        LT_p: Function for steady state transition matrix (mathcal{L}* in text) gradient w.r.t. price (Nz*Na x Nz*Na matrix)
        T: Number of time periods
        dt: Time step
        L_c: Steady state expectation matrix (mathcal{L} in text) gradient w.r.t. consumption (Nz*Na x Nz*Na matrix)
            If unspecified or None, use dphi_da
        dphi_da: Sum over x of Derivative of phi_t w.r.t. x (Nz*Na x T matrix)
        u_cp: Derivative of utility w.r.t. consumption and price (either Nz x Na matrix or Nz*Nz x 1 vector)
        u_cz: Derivative of utility w.r.t. consumption and aggregate shock (either Nz x Na matrix or Nz*Nz x 1 vector)
        c_t: Change in policy at time 0 from anticipated price/aggregate shock at each time s (T x 1 vector)
            If not None, use to calculate D directly from c_t
        price: True if price shock, False if aggregate shock

    Returns:
        D: Change in distribution (Nz*Na x T matrix)
        P: Change in distribution from immediate income shock (Nz*Na x 1 vector)
    """

    U_inv = 1 / U.flatten()
    gm = gm.flatten()

    if price: # Distribution change from change in income due to price shock
        P = LT_p(gm)
        if u_cp is not None:
            P -= LT_c(U_inv * u_cp.flatten())
    else: # Distribution change from change in income due to aggregate shock
        P = np.zeros_like(gm)
        if u_cz is not None:
            P = -LT_c(U_inv * u_cz.flatten())

    # If change in policy available, calculate D directly
    if c_t is not None:
        D = LT_c(c_t)
        D[:, 0] += P # Add change in distribution directly from changed incomes
        return D, P

    # Solve for \cM
    if L_c is not None:
        cM = -LT_c(U_inv * gm * (L_c @ phi))
    else:
        cM = LT_c((U_inv * gm)[:,np.newaxis] * dphi_da)

    # Solve for D
    D = np.exp(-rho * dt * np.arange(T))[np.newaxis,:] * cM
    D[:, 0] += P # Add change in distribution directly from changed incomes

    return D, P

def calc_E(E, T, ss, dt = 1.0, outputs = None, iter_style = 'DT_loop'):
    """Calculates E_t = T_t E = (I + dt L)^t E, the effect on output y at time t
    from a change in the mass at gridpoint x at time 0 under steady-state policy rules

    Args:
        E: Dictionary of E_0 matrices by output giving dy/dg(x) (the effect of output y from a change in the mass at x)
        T: Number of time periods
        ss: Steady-state dictionary
        dt: Time step
        outputs: List of output variables
        iter_style: How to iterate forward in phi and E calculations
            3 options: 'CT_matrix', 'CT_loop', 'DT_loop' where 'CT' uses ly and DT uses Pi and 'matrix' does a sparse matrix multiplication while 'loop' uses a for loop with numba

    Returns:
        E_t_struct: Dictionary of E_t matrices by output
    """

    if outputs is None:
        outputs = E.keys()
    E_t_struct = {output: np.zeros((T, E[output].shape[0], E[output].shape[1])) for output in outputs}

    for output in outputs:
        E_t_struct[output][0, ...] = E[output]

    # Iterate forward to get E_t
    for output in outputs:
        for t in range(T - 1):
            if iter_style == 'DT_loop':
                E_t_struct[output][t + 1, ...] = forward_step_transpose_1d_dt(E_t_struct[output][t, ...], ss['Pi'], ss['a_ind'], ss['sav_wt'])
            elif iter_style == 'CT_loop':
                E_t_struct[output][t + 1, ...] = forward_step_transpose_1d(E_t_struct[output][t, ...], ss['ly'], ss['a_ind'], ss['S_npy'], dt = dt)
            else:
                E_t_struct[output][t + 1, ...] = forward_step_matrix(E_t_struct[output][t, ...], ss['ly'], ss['S_stable'], dt = dt)

        E_t_struct[output] = E_t_struct[output]

    return E_t_struct


def calc_F(E, D, outputs, T, c_t = {}):
    """Calculates the fake news matrix for each output

    Args:
        E: Dictionary of E matrices (response of output to distribution change) by output where E_t = T_t E = (I + dt L)^t E
        D: Array giving the change in the distribution from shock
        outputs: List of output variables
        T: Number of time periods
        c_t: Dictionary for effect at time 0 for control variables

    Returns:
        F: Dictionary of fake news matrices by output
    """

    controls = c_t.keys()
    F = {o: {} for o in outputs}
    for output in outputs:
        F[output] = np.zeros((T, T))
        if output in controls:
            F[output][0, :] = c_t[output]
        F[output][1:, :] = (E[output][:T - 1, ...].reshape(T - 1, -1) @ D)

    return F

@njit
def J_from_F(F, dt = 1.0):
    """Helper function to calculate the Jacobian by adding up diagonals of the fake news matrix

    Args:
        F: Fake news matrix

    Returns:
        J: Jacobian matrix
    """
    J = F.copy()
    for t in range(1, J.shape[0]):
        J[1:, t] = J[:-1, t - 1] + dt * F[1:, t]
    return J

def calc_J(F, outputs, dt = 1.0):
    """Calculates the Jacobian for each output by adding up diagonals of the fake news matrix

    Args:
        F: A dictionary of fake news matrices.
        outputs: A list of output variables.

    Returns:
        J: A dictionary of Jacobian matrices.
    """
    J = {}
    for output in outputs:
        J[output] = J_from_F(F[output], dt = dt) * dt

    return J

def calc_irfs(J, shock, zeta, prices):
    """Calculates the impulse responses using partial equilibrium price Jacobians.
    Need Jacobian of each price w.r.t. all prices (including itself)

    Args:
        J: Dictionary of Jacobian matrices. Must include Jacobians of each price to each price.
        shock: Vector of shock (of length T)
        zeta: Dictionary where each price gives a number for how the shock affects each price (dp/dz(0,g^{ss})). 0 if t not equal to s
        prices: List of prices
    Returns:
        irfs: A dictionary of IRFs of each price (which can then be used to get IRFs of other variables)
    """
    dp_vec = np.zeros(len(prices) * len(shock))
    for price in prices:
        dp_vec = np.concatenate((dp_vec, zeta[price] * shock))

    # Create block matrix that will be inverted
    full_jac = []

    # Loop through each row
    for j in range(len(prices)):
        # List to hold the blocks for the current row
        row_blocks = []
        # Loop through each column
        for i in range(len(prices)):
            # Append the current block to the row
            row_blocks.append(J[prices[i]][prices[j]])
        # Append the current row to the big matrix
        full_jac.append(row_blocks)

    # Combine list into a matrix
    full_jac = np.block(full_jac)

    # Solve given system of equations
    solved_vec = solve(np.eye(len(full_jac)) - full_jac, dp_vec)

    # Convert into dictionary
    irfs = {}
    for i, price in enumerate(prices):
        irfs[price] = solved_vec[i * len(shock):(i + 1) * len(shock)]

    return irfs

