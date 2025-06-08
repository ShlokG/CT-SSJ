# Everything for the HA Model in 1 file

import numpy as np
import matplotlib.pyplot as plt
from toolkit.utils import markov_rouwenhorst
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve
import time

# start timer
start_time = time.time()

# Baseline numerical parameters
dt   = 1.0           # length of time period
T    = int(300 / dt) # number of time periods
Nz   = 25            # number of productivity gridpoints
Na   = 2500          # number of asset gridpoints
Ntot = Na * Nz       # number of total gridpoints

anticipate = False # whether to anticipate the time-0 r shock
mu         = 0.5   # mass of agents moving out of current gridpt in transition
plot_T     = 30    # number of periods to plot IRFs for

# directories
fig_dir     = 'Figures/HA_'                       # directory to save figures + beginning of file name
str_append  = f'_Nz{Nz}_Na{Na}'                   # string to append to file name
str_append += f'_dt{dt:.1f}' if dt != 1.0 else '' # add dt to file name if not 1

# define economic and numerical parameters

# preferences
gamma = 2    # risk aversion
rho   = 0.05 # annualized discount rate

# individual productivity process
rho_e   = 0.91 # rho for AR(1) of log(e)
sigma_e = 0.5  # stdev of cross-section of idiosyncratic productivity

# create idio. prod. grid & transition matrix via Rouwenhorst
z, _, Pi = markov_rouwenhorst(rho_e, sigma_e, Nz)
ly       = Pi - np.eye(Nz) # subtract where agent comes from in transition matrix
N        = 1               # mean idiosyncratic productivity in economy

# production
d     = 0.1          # annualized capital depreciation rate
alpha = 1/3          # capital share in production
AgZ   = 1            # mean aggregate productivity
rho_Z = -np.log(0.7) # persistence of aggregate productivity

# borrowing constraint
amin = 0

# grids
amax = 200                             # maximum assets
a    = np.linspace(amin, amax, Na)     # asset grid
da0  = a[1:] - a[:-1]
da   = np.concatenate(([da0[0]], da0)) # grid of asset steps da

# grids in asset x income space
aa  = np.tile(a, (Nz, 1))   # assets
daa = np.tile(da, (Nz, 1))  # asset steps
zz  = np.tile(z, (Na, 1)).T # productivity

# convergence and smoothing criteria
Delta      = 10000         # time step smoothing for HJB
crit_S     = 1e-6          # convergence criterion
rmin       = 1e-5          # lower bound on possible interest rate
rmax       = rho * 0.9999  # upper bound on possible interest rate
Ir         = 300           # maximum number of interest rate iterations
ifix       = 0             # index where to normalize the distribution inversion
back_tol   = 1e-8          # backward iteration convergence tolerance
back_maxit = 5000          # backward iteration maximum iterations

# transition matrices
Ly   = sp.dia_matrix(sp.kron(ly, sp.eye(Na)))
Ly_T = Ly.T

time2 = time.time()
print(f'defining parameters: {time2 - start_time:.3f} s.')

# solve for steady-state

# normalization of right-hand-side of KFE for inversion
gRHS       = np.zeros(Ntot)
gRHS[ifix] = 1
gRow       = np.zeros(Ntot)
gRow[ifix] = 1

# initialization
dVf = np.zeros((Nz, Na))  # forward finite difference of value
dVb = np.zeros((Nz, Na))  # backward finite difference of value
c   = np.zeros((Nz, Na))  # consumption
r   = (rmin + rmax) / 2  # interest rate

# express wage as a function of current interest rate
w  = alpha ** (alpha / (1 - alpha)) * (1 - alpha) * AgZ ** (1 / (1 - alpha)) * (r + d)**(-alpha / (1 - alpha))
ra = r * aa  # interest income
wz = w * zz  # labor income

# initial guess of value function
if gamma != 1:
    v = ((wz + ra) ** (1 - gamma)) / (1 - gamma) / rho
else:
    v = np.log(wz + ra) / rho

# loop over r
for ir in range(Ir):
    # express wage as a function of current interest rate guess
    w  = alpha ** (alpha / (1 - alpha)) * (1 - alpha) * AgZ ** (1 / (1 - alpha)) * (r + d)**(-alpha / (1 - alpha))
    ra = r * aa  # interest income
    wz = w * zz  # labor income

    if wz[0, 0] + r * amin < 0:
        print('CAREFUL: borrowing constraint too loose')

    # converge value function
    for j in range(back_maxit):
        V = v

        # forward difference
        dVf[:, :-1] = (V[:, 1:] - V[:, :-1]) / daa[:, :-1]
        dVf[:, -1]  = (wz[:, -1] + ra[:, -1]) ** (-gamma)

        # backward difference
        dVb[:, 1:] = (V[:, 1:] - V[:, :-1]) / daa[:, :-1]
        dVb[:, 0]  = (wz[:, 0] + ra[:, 0]) ** (-gamma)

        # consumption and savings with forward difference
        cf = dVf ** (-1 / gamma)
        sf = wz + ra - cf

        # consumption and savings with backward difference
        cb = dVb ** (-1 / gamma)
        sb = wz + ra - cb

        # consumption and derivative of value function at steady state
        c0  = wz + ra
        dV0 = c0 ** (-gamma)

        # indicators for upwind savings rate
        If = sf > 0
        Ib = sb < 0
        I0 = 1 - If - Ib

        # consumption
        dV_Upwind = dVf * If + dVb * Ib + dV0 * I0
        c         = dV_Upwind ** (-1 / gamma)

        # utility
        if gamma != 1:
            u = (c ** (1 - gamma)) / (1 - gamma)
        else:
            u = np.log(c)

        # savings finite difference matrix
        Sb = -np.minimum(sb, 0) / daa
        Sm = -np.maximum(sf, 0) / daa + np.minimum(sb, 0) / daa
        Sf = np.maximum(sf, 0) / daa
        S  = sp.diags(Sm.flatten(), 0, shape = (Ntot, Ntot)) + \
             sp.diags(Sb.flatten()[1:], -1, shape = (Ntot, Ntot)) + \
             sp.diags(Sf.flatten()[:-1], 1, shape = (Ntot, Ntot))

        # test rows sum to 0
        if np.max(np.abs(S.sum(axis=1))) > 1e-5:
            print('Improper savings transition matrix')

        # matrix to invert in finite difference scheme
        M = (1 / Delta + rho) * sp.eye(Ntot) - Ly - S

        # invert linear system
        V = sp.linalg.spsolve(M, u.ravel() + V.ravel() / Delta)
        V = V.reshape(Nz, Na)

        # update
        Vchange = V - v
        v       = V
        dist    = np.max(np.abs(Vchange))
        if dist < back_tol:
            break

    # update aggregates
    LT = S.transpose() + Ly_T  # transpose of transition matrix

    # normalize transition matrix row to make system invertible
    # directly adjusting data in sparse matrix for speed
    data    = LT.data
    offsets = LT.offsets
    for i, offset in enumerate(offsets):
        if offset == 0:
            data[i, ifix] = 1
        elif ifix + offset >= 0:
            data[i, ifix] = 0

    LT = LT.tocsr()                                     # convert to csr for faster row access
    g  = np.maximum(np.real(spla.spsolve(LT, gRHS)), 0) # solve for distribution by inverting linear system
    g  = g / np.sum(g * daa.ravel())                    # normalize back to unit mass

    KS  = np.sum(g * aa.ravel() * daa.ravel())             # capital supply
    KD  = (alpha * AgZ / (r + d)) ** (1 / (1 - alpha)) * N # capital demand
    Sav = KS - KD                                          # net savings

    # update interest rate according to bisection
    if Sav > crit_S:
        rmax = r
        r    = 0.5 * (r + rmin)
    elif Sav < -crit_S:
        rmin = r
        r    = 0.5 * (r + rmax)
    elif np.abs(Sav) < crit_S:
        break

    if ir == Ir - 1:
        print(f'\nCould not find steady-state, r = {r:.4f}')
        raise ValueError('Could not find steady-state')

# recompute interest rate and wage for consistency
r = -d + alpha * AgZ * np.maximum(np.dot(aa.ravel(), g * daa.ravel()), 1e-5) ** (alpha - 1) * N ** (1 - alpha)
w = alpha ** (alpha / (1 - alpha)) * (1 - alpha) * AgZ ** (1 / (1 - alpha)) * (r + d) ** (-alpha / (1 - alpha))

# distribution probability mass function and generator
gm = g.reshape(Nz, Na) * daa
L  = LT.T

# get which agents are constrained
consted     = (aa == amin) * (I0 == 1)     # boolean matrix of constrained agents
consted_ind = np.where(consted.ravel())[0] # indices of constrained agents

# first and second derivatives of utility
up    = c ** (-gamma)               # u'(c)
upp   = -gamma * c ** (-gamma - 1)  # u"(c)
U_inv = 1 / upp

time3 = time.time()
print(f'solving steady-state: {time3 - time2:.3f} s.')

# construct sequence-space Jacobians

# construct d/da matrix for distribution using upwinding

# use savings matrix for upwinding
Sf1  = (np.minimum(sb, 0) < 0).astype(int) / daa
Sm1  = ((np.maximum(sf, 0) > 0).astype(int) - (np.minimum(sb, 0) < 0).astype(int)) / daa
Sb1  = -(np.maximum(sf, 0) > 0).astype(int) / daa

# construct d/da matrix for distribution
DA_T = sp.diags(Sm1.flatten(), 0, shape = (Ntot, Ntot)) + \
                    sp.diags(Sb1.flatten()[:-1], -1, shape = (Ntot, Ntot)) + \
                    sp.diags(Sf1.flatten()[1:], 1, shape = (Ntot, Ntot))

# construct d/da matrix for values
Sm1[:,0] = 1/da[0]
Sb1[:,0] = -1/da[0]
Sf1[:,0] = 0

DA = -sp.diags(Sm1.flatten(), 0, shape = (Ntot, Ntot)) - \
                    sp.diags(Sf1.flatten()[1:], -1, shape = (Ntot, Ntot)) - \
                    sp.diags(Sb1.flatten()[:-1], 1, shape = (Ntot, Ntot))

# calculate other necessary values

# get asset index agent moving to via saving
sav_sign = np.sign(Sm1)                                # get whether saving/dissaving
S_da     = -S.diagonal() / mu * dt                     # amount saved at each index
S_pts    = S_da.reshape(Nz, Na) * sav_sign             # multiply change in gridpts by savings sign
a_ind    = np.floor(S_pts).astype(int) + np.arange(Na) # get gridpt moved to 
sav_wt   = 1 - np.mod(S_pts, 1)                        # interpolation weight on left gridpt (lower weight if far from floor)

# adjust a_ind and sav_wt so don't go out of grid
out_of_bounds_high         = a_ind >= Na - 1  # indices saving outside of grid
sav_wt[out_of_bounds_high] = 0                # move agents to last gridpt if out of bounds
a_ind[out_of_bounds_high]  = Na - 2     

out_of_bounds_low          = a_ind < 0        # indices saving outside of grid
sav_wt[out_of_bounds_low]  = 1                # move agents to last gridpt if out of bounds
a_ind[out_of_bounds_low]   = 0
a_ind                     -= np.arange(Na)    # convert back to change in gridpts

# adjust mu at gridpts where savings would push agents out of
# bounds to ensure correct mass moving to endpoint
mu_mat = np.ones((Nz, Na)) * mu # define a separate mass leaving for each gridpt

# saving too much
pts_moved                  = (amax - aa[out_of_bounds_high])                     # gridpts actually moved
pts_supposed               = S_pts[out_of_bounds_high] * daa[out_of_bounds_high] # gridpts supposed to move
pts_moved                  = np.maximum(pts_moved, 1e-12)                        # pts_moved can be 0 if starting at max gridpt; then, set mass to 1
mu_mat[out_of_bounds_high] = np.minimum(pts_supposed / pts_moved * mu, 1)        # increase mass leaving so correct transition maintained

# repeat for case when dissaving too much
pts_moved                 = aa[out_of_bounds_low] - a[0]                       # gridpts actually moved
pts_supposed              = -S_pts[out_of_bounds_low] * daa[out_of_bounds_low] # gridpts supposed to move
mu_mat[out_of_bounds_low] = np.minimum(pts_supposed / pts_moved * mu, 1)       # increase mass leaving so correct transition maintained

# convert a_ind and sav_wt into a transition matrix
# akin to LT but can move multiple asset gridpts for stability and rows sum to 1, not 0
sav_data = np.concatenate([(sav_wt * mu_mat).ravel(), ((1 - sav_wt) * mu_mat).ravel()]) # multiply by mu so only fraction moves
sav_rows = np.concatenate([np.arange(Ntot), np.arange(Ntot)])
cols     = range(Ntot) + a_ind.ravel()
sav_cols = np.concatenate([cols, cols + 1])
S_stable = sp.csr_matrix((sav_data, (sav_rows, sav_cols)), shape = (Ntot, Ntot))

# subtract mass leaving
S_stable = (S_stable - sp.diags(mu_mat.ravel())) / dt

# get values useful for solution
K = np.sum(gm * aa)  # aggregate capital
C = np.sum(gm * c)   # aggregate consumption

# aggregate shock
zeta_r    = alpha * (N / K) ** (1 - alpha)    # zeta_r = dr/dz(0, g^{SS}). 0 if t \neq s
zeta_w    = (1 - alpha) * (N / K) ** (-alpha) # zeta_w = dw/dz(0, g^{SS})
omega_bar = -K / N                            # dw/dr

# check that income equals output
assert np.abs(w * N + (r + d) * K - AgZ * N ** (1 - alpha) * K ** alpha) < 1e-3

time4 = time.time()
print(f'prepping for Jacobians: {time4 - time3:.3f} s.')

# step 1: calculate change in value from future shock and consumption at time 0
# solve for phi_t by iterating forward
phi_r          = np.zeros((Nz, Na, T))  # initializations
phi_r[:, :, 0] = aa * up                # get \vp_0: income change from r shock * u'(c)
phi_w          = np.zeros((Nz, Na, T))
phi_w[:, :, 0] = zz * up                # income change from wage shock * u'(c)

# apply constraints
phi_r[consted, 0] = phi_r[:, 1:, 0][consted[:, :-1]] - da[0] * (upp * aa)[consted]
phi_w[consted, 0] = phi_w[:, 1:, 0][consted[:, :-1]] - da[0] * (upp * zz)[consted]

# iteration
for t in range(T - 1):
    # save last period's phi
    phi_rt = phi_r[:, :, t]
    phi_wt = phi_w[:, :, t]

    # transition to next period
    phi_r[:, :, t + 1] = phi_rt + dt * (ly @ phi_rt + (S_stable @ phi_rt.ravel()).reshape(Nz, Na))
    phi_w[:, :, t + 1] = phi_wt + dt * (ly @ phi_wt + (S_stable @ phi_wt.ravel()).reshape(Nz, Na))

    # adjust constrained agents' value
    phi_r[:, :, t + 1][consted] = phi_r[:, 1:, t + 1][consted[:,:-1]]
    phi_w[:, :, t + 1][consted] = phi_w[:, 1:, t + 1][consted[:,:-1]]

# reshape into vector at each time for later matrix multiplication
phi_r = phi_r.reshape(Ntot, T)
phi_w = phi_w.reshape(Ntot, T)

# derivative with respect to assets
dphi_da_r = DA @ phi_r
dphi_da_w = DA @ phi_w

# adjust time-0 for ex-post shock
if not anticipate:
    dphi_da_r[:, 0] = (DA @ up.ravel()) * aa.ravel()
    dphi_da_w[:, 0] = (DA @ up.ravel()) * zz.ravel()

# calculate change in consumption at time 0
rho_T  = np.expand_dims(np.exp(-rho * np.arange(T) * dt), axis=0) # discounting of future
Ug_rho = (U_inv * gm).ravel()[:, np.newaxis] * rho_T              # U^{-1} g(x) \rho_t
c_r   = dphi_da_r * Ug_rho  # consumption response to a future r shock
c_w   = dphi_da_w * Ug_rho  # consumption response to a future w shock

# adjust distribution for consumption adjustment at time 0
if not anticipate:
    c_prime   = DA @ c.ravel()          # c'(a)
    c_r[:, 0] = (gm * aa).ravel() * c_prime # adjusted time-0 consumption

# total change in distribution at each time
D_r        = DA_T @ c_r               # change in distribution from consumption change
D_r[:, 0] -= DA_T @ (gm * aa).ravel() # adjust for change in distribution at time 0 directly from income change
D_w        = DA_T @ c_w               # change in distribution from consumption change
D_w[:, 0] -= DA_T @ (gm * zz).ravel() # adjust for change in distribution at time 0 directly from income change

# aggregate change in consumption
C_r = np.sum(c_r, axis = 0)
C_w = np.sum(c_w, axis = 0)

# start timer
time5 = time.time()
print(f'Step 1 in Jacobians: policy functions: {time5 - time4:.3f} s.')

# calculate expectation vectors E (dy_t/dg(x)_0)
# change in output from change in mass at x at time 0
# under steady state policy function

# initialize
E_r = np.zeros((T, Nz, Na))
E_w = np.zeros((T, Nz, Na))
E_K = np.zeros((T, Nz, Na))
E_C = np.zeros((T, Nz, Na))

# effect at time 0
E_r[0, :, :] = AgZ * alpha * (1 - alpha) * (N / K) ** (1 - alpha) * (zz / N - aa / K)
E_w[0, :, :] = E_r[0, :, :] * omega_bar
E_K[0, :, :] = aa
E_C[0, :, :] = c

# iteration
# equivalent to E_t = T_t E_0 in the paper
for t in range(T - 1):
    # save last period's E
    E_rt = E_r[t, :, :]
    E_wt = E_w[t, :, :]
    E_Kt = E_K[t, :, :]
    E_Ct = E_C[t, :, :]

    # exogenous idiosyncratic productivity transition
    E_r[t + 1, :, :] = E_rt + dt * (ly @ E_rt + (S_stable @ E_rt.ravel()).reshape(Nz, Na))
    E_w[t + 1, :, :] = E_wt + dt * (ly @ E_wt + (S_stable @ E_wt.ravel()).reshape(Nz, Na))
    E_K[t + 1, :, :] = E_Kt + dt * (ly @ E_Kt + (S_stable @ E_Kt.ravel()).reshape(Nz, Na))
    E_C[t + 1, :, :] = E_Ct + dt * (ly @ E_Ct + (S_stable @ E_Ct.ravel()).reshape(Nz, Na))

E_r = E_r.reshape(T, Ntot)
E_w = E_w.reshape(T, Ntot)
E_K = E_K.reshape(T, Ntot)
E_C = E_C.reshape(T, Ntot)

time6 = time.time()
print(f'Step 2 in Jacobians: expectation functions: {time6 - time5:.3f} s.')

# jacobian matrices
# compute fake news kernel, F_{t,s} = E_t^* D_s
F_rr = E_r @ D_r  # r shock on r
F_rK = E_K @ D_r  # r shock on K
F_rC = E_C @ D_r  # r shock on C

F_ww = E_w @ D_w  # w shock on w
F_wK = E_K @ D_w  # w shock on K
F_wC = E_C @ D_w  # w shock on C

# compute the Jacobian
# initialization
J_rr = np.zeros((T, T))  # jacobian of r to a r shock
J_rK = np.zeros((T, T))  # jacobian of K to a r shock
J_rC = np.zeros((T, T))  # jacobian of C to a r shock

J_ww = np.zeros((T, T))  # jacobian of w to a w shock
J_wK = np.zeros((T, T))  # jacobian of K to a w shock
J_wC = np.zeros((T, T))  # jacobian of C to a w shock

# get the first column from change in dist.
J_rr[1:, 0] = F_rr[:-1, 0]
J_rK[1:, 0] = F_rK[:-1, 0]
J_rC[1:, 0] = F_rC[:-1, 0]

J_ww[1:, 0] = F_ww[:-1, 0]
J_wK[1:, 0] = F_wK[:-1, 0]
J_wC[1:, 0] = F_wC[:-1, 0]

# consumption is a control so need to account for change in behavior
# get change in policy function consumption
J_rC[0, :] = C_r
J_wC[0, :] = C_w

# elements of Jacobian from summing across diagonals
for t in range(1, T):
    J_rr[1:, t] = J_rr[:-1, t - 1] + dt * F_rr[1:, t]
    J_rK[1:, t] = J_rK[:-1, t - 1] + dt * F_rK[1:, t]
    J_rC[1:, t] = J_rC[:-1, t - 1] + dt * F_rC[1:, t]

    J_ww[1:, t] = J_ww[:-1, t - 1] + dt * F_ww[1:, t]
    J_wK[1:, t] = J_wK[:-1, t - 1] + dt * F_wK[1:, t]
    J_wC[1:, t] = J_wC[:-1, t - 1] + dt * F_wC[1:, t]

# adjust by dt to convert into operator
J_rr *= dt
J_rK *= dt
J_rC *= dt

J_ww *= dt
J_wK *= dt
J_wC *= dt

time7 = time.time()
print(f'Step 3 & 4 in Jacobians: fake news and Jacobians: {time7 - time6:.3f} s.')

# impulse responses to aggregate shocks

# 1% TFP shock
timeZ     = np.linspace(0, (T - 1) * dt, T)
Z_impulse = 0.01 * np.exp(-rho_Z * timeZ)

# write system of eqs in matrix form, w/ r and w as 1 vector:
# p-hat = [r-hat;w-hat] = [J_rr, J_rw; J_wr, J_ww] *
# p-hat + [zeta_r I, 0; 0, zeta_w I]z-hat
full_jac = np.block([[J_rr, J_ww / omega_bar], [omega_bar * J_rr, J_ww]])

# repeat z_hat for each price and multiply by dp/dz
z_hat      = np.tile(Z_impulse, 2)
z_hat[:T] *= zeta_r  # TFP shock multiplied by direct effect on r
z_hat[T:] *= zeta_w  # TFP shock multiplied by direct effect on w

# solve system of equations to get r and w
p_hat = solve(np.eye(2 * T) - full_jac, z_hat)
dr    = p_hat[:T]
dw    = p_hat[T:]

# get implied capital and consumption from Jacobians
dK = J_rK @ dr + J_wK @ dw
dC = J_rC @ dr + J_wC @ dw

print(f'Step 5 in Jacobians: impulse responses: {time.time() - time7:.3f} s.')

# plot Jacobians for Comparison
columns_to_plot = [0, int(100 / dt), int(200 / dt)]
linestyles      = ['-', '--', '-.', ':', (0, (1, 10))]
default_colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

# function to plot Jacobian columns
def plot_Jac(jac, shock_name = 'K', price = 'r', fig_dir = 'Figures/', str_append = ''):
    CT_Val = 1
    if shock_name == 'K':
        CT_Val = K
    elif shock_name == 'C':
        CT_Val = C

    for i, col in enumerate(columns_to_plot):
        plt.plot(np.arange(0, T * dt, dt), jac[:, col] / CT_Val, label = f"s = {col * dt:.0f}",
            linestyle = linestyles[0], linewidth = 3, color = default_colors[i])
        plt.legend(framealpha=0)

    plt.xlabel("Year", fontsize = 14)
    plt.ylabel("p.p. deviation from SS", fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.tight_layout()

    plt.savefig(fig_dir + "Jac_" + shock_name + price + str_append + ".pdf")
    plt.show()

plot_Jac(J_rr, shock_name = 'r', price = 'r', fig_dir = fig_dir, str_append = str_append)
plot_Jac(J_rK, shock_name = 'K', price = 'r', fig_dir = fig_dir, str_append = str_append)
plot_Jac(J_rC, shock_name = 'C', price = 'r', fig_dir = fig_dir, str_append = str_append)

plot_Jac(J_ww, shock_name = 'w', price = 'w', fig_dir = fig_dir, str_append = str_append)
plot_Jac(J_wK, shock_name = 'K', price = 'w', fig_dir = fig_dir, str_append = str_append)
plot_Jac(J_wC, shock_name = 'C', price = 'w', fig_dir = fig_dir, str_append = str_append)

# impulse responses of r, w, K, C to a Z (productivity) shock
plt.rc('legend', fontsize = 16)
plot_t = np.minimum(plot_T, T * dt)
def plot_IRF(CT_irf, shock_name = "r", fig_dir = 'Figures/', str_append = ''):
    if shock_name == "K":
        CT_val = K
    elif shock_name == "C":
        CT_val = C
    else:
        CT_val = 1

    plt.plot(np.arange(0, plot_t, dt), CT_irf[:int(plot_T / dt)] * 100 / CT_val,
        linestyle = linestyles[0], color = default_colors[0],
        linewidth = 3)

    # increase tick label sizes
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel("Year", fontsize = 14)
    plt.ylabel("p.p. deviation from SS", fontsize = 14)

    plt.tight_layout()
    plt.savefig(fig_dir + "IRF_" + shock_name + str_append + ".pdf")
    plt.show()

plot_IRF(dr, shock_name = "r", fig_dir = fig_dir, str_append = str_append)
plot_IRF(dw, shock_name = "w", fig_dir = fig_dir, str_append = str_append)

CT_K = J_rK @ dr + J_wK @ dw
CT_C = J_rC @ dr + J_wC @ dw

plot_IRF(CT_K, shock_name = "K", fig_dir = fig_dir, str_append = str_append)
plot_IRF(CT_C, shock_name = "C", fig_dir = fig_dir, str_append = str_append)

