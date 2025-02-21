"""Define HA-one model block."""

import sequence_jacobian as sj
from . import calibration

"""HA-one model"""

make_grids = sj.hetblocks.hh_sim.make_grids

def income(Z, e_grid):
    y = Z * e_grid
    return y

ha_one = sj.hetblocks.hh_sim.hh.add_hetinputs([make_grids, income])

"""Convenience routines to calculate HA-hi-liq, HA-one, HA-two given parameters and calibration"""

def get_all(params):
    calib_ha_one, _ = calibration.get_ha_calibrations()

    hh_het, ss_het = {}, {}
    hh_het['HA-hi-liq'], ss_het['HA-hi-liq'] = ha_one, ha_one.steady_state({**calib_ha_one, **params['HA-hi-liq']})
    hh_het['HA-one'], ss_het['HA-one'] = ha_one, ha_one.steady_state({**calib_ha_one, **params['HA-one']})
    return hh_het, ss_het
