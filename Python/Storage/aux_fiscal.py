import numpy as np

# TODO: change R to 1+r in Tplan, write compute_multipliers more generically

def Bplan(dG, rho_B):
    """Determine financing for any desired time path of spending dG_t,
    solving dB_t = rho_B (dB_{t-1} + dG_t)"""
    T = len(dG)
    dB = np.empty(T)
    for t in range(T):
        dB_lag = dB[t-1] if t>0 else 0
        dB[t] = rho_B * (dB_lag + dG[t])
    return dB


def Tplan(dG, dB, R):
    """Get tax plan dT_t coresponding to any path dG_t, dB_t"""
    dB_lag = np.concatenate(([0], dB[:-1]))
    dT = dG + R*dB_lag - dB
    return dT


def compute_multipliers(dY, dG, r):
    """Compute multipliers given impulses dG_t, dY_t"""
    q = (1 + r) ** (-np.arange(len(dY)))

    mult_impact = dY[0] / dG[0]
    mult_5Y     = (q[:5] @ dY[:5]) / (q[:5] @ dG[:5])
    mult_cumul  = (q @ dY) / (q @ dG)

    return mult_impact, mult_5Y, mult_cumul
