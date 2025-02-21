"""Calibration parameters for models and IKC / quantitative environment
Dicts should be accessed by end user via functions, which return separate
copies to avoid accidentally modifying a shared dict."""

# core parameters consistent across all models and environments
r = 0.05        # annual real interest rate
eis = 1         # elasticity of intertemporal substitution
theta = 0.181   # progressivity of HSV
core = dict(r = r, eis = eis, theta = theta)

# parameters for fiscal shocks
rhoG = 0.76         # persistence of G shock (always used)
rhoB = 0.93         # persistence of B shock (maximum used)

# shared calibration parameters for HA models
calibration_ha = dict(
    theta = theta,
    rho_e = 0.91,           # persistence of idiosyncratic productivity shock
    sd_e = (1-theta)*0.92,  # stdev of post-tax idiosyncratic productivity
)

# specific parameters for HA-one model
calibration_ha_one = dict(
    min_a = 0.,    # min asset on grid
    max_a = 1000,  # max asset on grid
    n_a = 200,     # number of asset grid points
    n_e = 11,      # number of productivity grid points
)

# specific parameters for HA-two model
gamma = 5 
calibration_ha_two = dict(
    zeta = 0.08/(1+r),  # 8% spread
    min_a = 0.,         # min illiquid asset on grid
    max_a = 10000,      # max illiquid asset on grid
    min_b = 0.,         # min liquid asset on grid
    max_b = 10000,      # max liquid asset on grid
    n_a = 50,           # number of illiquid asset grid points
    n_b = 50,           # number of liquid asset grid points
    n_e = 11,           # number of productivity grid points
)

def get_ha_calibrations():
    """Return copies of calibration info for HA-one and HA-two models"""
    return ({**core, **calibration_ha, **calibration_ha_one},
            {**core, **calibration_ha, **calibration_ha_two})
