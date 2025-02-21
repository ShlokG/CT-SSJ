import numpy as np
from toolkit import utils
import scipy.sparse as sp

def param_econ():
    p = {}

    ## preferences
    p['gamma'] = 2    # risk aversion          
    p['rho']   = 0.05 # annualized discount rate

    ## production
    p['d']     = 0.1          # annualized capital depreciation rate
    p['alpha'] = 1/3          # capital share in production
    p['AgZ']   = 1            # mean aggregate productivity
    p['rho_Z'] = -np.log(0.7) # persistence of aggregate productivity
    p['N']     = 1            # mean idiosyncratic productivity in economy

    ## individual productivity process
    p['rho_e']   = 0.91 # rho for AR(1) of log(e)
    p['sigma_e'] = 0.5  # stdev of cross-section of idiocyncratic productivity 

    # borrowing constraint
    p['amin'] = 0

    return p

# Numerical Parameters
def param_num(p, Nz, Na = 500, mu = 1.0):
    ## Idiosyncratic states
    n = {}
    n['Nz'] = Nz
    n['Na'] = Na
    n['Ntot'] = n['Na'] * n['Nz']

    # create idio. prod. grid & transition matrix via Rouwenhorst
    p['z'], _, n['Pi'] = utils.markov_rouwenhorst(rho = p['rho_e'], sigma = p['sigma_e'], N = Nz)
    n['ly']            = n['Pi'] - np.eye(Nz) # subtract where agent comes from in transition matrix
    n['Pi_T']          = n['Pi'].T            # transpose of Pi used in forward iteration of EGM calculation

    n['amax'] = 200                                        # maximum assets
    n['a']    = np.linspace(p['amin'], n['amax'], n['Na']) # asset grid
    da0       = n['a'][1:] - n['a'][:-1]
    n['da']   = np.concatenate(([da0[0]], da0))            # grid of asset steps da

    # grids in asset x income space
    n['aa']  = np.tile(n['a'], (Nz, 1))   # assets
    n['daa'] = np.tile(n['da'], (Nz, 1))  # asset steps
    n['zz']  = np.tile(p['z'], (Na, 1)).T # productivity

    # convergence and smoothing criteria
    n['Delta']      = 10000              # time step smoothing for HJB
    n['crit_S']     = 1e-6               # convergence criterion
    n['rmin']       = 1e-5               # lower bound on possible interest rate
    n['rmax']       = p['rho'] * 0.9999  # upper bound on possible interest rate
    n['Ir']         = 300                # maximum number of interest rate iterations
    n['ifix']       = 0                  # index where to normalize the distribution inversion
    n['back_tol']   = 1e-8               # backward iteration convergence tolerance
    n['back_maxit'] = 5000               # backward iteration maximum iterations
    n['fwd_tol']    = 1e-10              # forward iteration convergence tolerance
    n['fwd_maxit']  = 100_000            # forward iteration maximum iterations

    # transition matrices
    n['Ly']   = sp.dia_matrix(sp.kron(n['ly'], sp.eye(n['Na']))) # productivity transition extended to asset space
    n['Ly_T'] = n['Ly'].transpose()
    n['M1']   = (1 / n['Delta'] + p['rho']) * sp.eye(n['Ntot']) - n['Ly']

    n['mu']   = mu   # mass of agents moving out of current gridpt in transition

    return n

# Economic Parameters for HANK
def param_econ_hank(calib):
    p = {}

    ## preferences
    p['gamma'] = 1 / calib['eis']   # elasticity of substitution
    p['beta']  = calib['beta']      # discount factor in discrete time
    p['rho']   = -np.log(p['beta']) # annualized discount rate

    # income
    p['theta'] = 0.181              # progressivity of HSV
    p['Z']     = 0.4709992940007061 # Y - T

    # individual productivity process
    p['rho_e']   = 0.91                     # rho for AR(1) of log(e)
    p['sigma_e'] = (1 - p['theta']) * 0.92  # sigma for AR(1) of log(e), adjusted for taxes

    # output
    p['AgZ'] = 1.0                 # mean aggregate productivity
    p['Nss'] = 1.0                 # steady-state labor
    p['Yss'] = p['AgZ'] * p['Nss'] # steady-state output

    # labor market
    p['mu_w']    = 1.1  # wage markup
    p['kappa_w'] = 0.03 # wage flexibility
    p['xi']      = 1    # labor supply elasticity

    # Taylor Rule coefficient on inflation
    p['phi'] = 1.0

    # steady state calibration
    p['G_share'] = 0.2                            # G/Y
    p['T_share'] = (p['Yss'] - p['Z']) / p['Yss'] # T/Y

    # borrowing constraint
    p['amin'] = 0

    # shock processes
    p['rho_G'] = -np.log(0.8) # persistence of G shock
    p['rho_r'] = 0.8          # persistence of r shock
    p['rho_b'] = -np.log(0.5) # beta in paper (how quickly debt paid off)
    
    return p

def param_num_hank(p, Nz = 7, Na = 500, amax = 200, mu = 1.0):
    ## Idiosyncratic states
    n         = {}
    n['Nz']   = Nz
    n['Na']   = Na
    n['Ntot'] = n['Na'] * n['Nz']
    n['amax'] = amax

    # create idio. prod. grid & transition matrix via Rouwenhorst
    p['z'], _, n['Pi'] = utils.markov_rouwenhorst(p['rho_e'], p['sigma_e'], Nz)
    n['ly']            = n['Pi'] - np.eye(Nz)  # subtract where agent comes from in transition matrix

    n['a']  = np.linspace(p['amin'], amax, Na) # asset grid
    da0     = n['a'][1:] - n['a'][:-1]
    n['da'] = np.concatenate(([da0[0]], da0))  # grid of asset steps da

    # grids in asset x income space
    n['aa']  = np.tile(n['a'], (Nz, 1))   # assets
    n['daa'] = np.tile(n['da'], (Nz, 1))  # asset steps
    n['zz']  = np.tile(p['z'], (Na, 1)).T # productivity

    # convergence and smoothing criteria
    n['Delta']      = 10000             # time step smoothing for HJB
    n['crit_S']     = 1e-6              # convergence criterion
    n['rmin']       = 1e-5              # lower bound on possible interest rate
    n['rmax']       = p['rho'] * 0.9999 # upper bound on possible interest rate
    n['Ir']         = 300               # maximum number of interest rate iterations
    n['ifix']       = 0                 # index where to normalize the distribution inversion
    n['back_tol']   = 1e-8              # backward iteration convergence tolerance
    n['back_maxit'] = 5000              # backward iteration maximum iterations
    n['fwd_tol']    = 1e-10             # backward iteration convergence tolerance
    n['fwd_maxit']  = 100_000           # forward iteration maximum iterations

    # transition matrices
    n['Ly']   = sp.dia_matrix(sp.kron(n['ly'], sp.eye(Na))) # productivity transition extended to asset space
    n['Ly_T'] = n['Ly'].T # transpose of Ly used to get LT
    n['Pi_T'] = n['Pi'].T # transpose used in forward iteration of EGM calculation
    n['M1']   = (1 / n['Delta'] + p['rho']) * sp.eye(n['Ntot']) - n['Ly'] # pre-calculating for implicit method

    n['mu']   = mu # mass of agents moving out of current gridpt in transition

    return n, p












