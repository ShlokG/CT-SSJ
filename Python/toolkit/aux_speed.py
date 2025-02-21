"""
This file contains functions used to measure computation times.
"""

import re
import inspect
import numpy as np
from numba import njit

from toolkit import utils
from toolkit import jacobian as jac


'''Part 1: Computing times for G'''


def step0_preliminaries(block_list, exogenous, unknowns, ss):
    curlyJs, required = jac.curlyJ_sorted(block_list, unknowns + exogenous, ss)
    return curlyJs, required


def step1_forward_H(curlyJs, exogenous, unknowns, targets, required):
    J_curlyH = jac.forward_accumulate(curlyJs, unknowns + exogenous, targets, required)
    return J_curlyH


def step2_solve_GU(J_curlyH, exogenous, unknowns, targets, T):
    H_U_packed = jac.pack_jacobians(J_curlyH, unknowns, targets, T)
    H_Z_packed = jac.pack_jacobians(J_curlyH, exogenous, targets, T)
    G_U = jac.unpack_jacobians(-np.linalg.solve(H_U_packed, H_Z_packed), exogenous, unknowns, T)
    return G_U


def step3_forward_G(G_U, curlyJs, exogenous, targets, outputs, required, unknowns):
    curlyJs = [G_U] + curlyJs
    if outputs is None:
        outputs = set().union(*(curlyJ.keys() for curlyJ in curlyJs)) - set(targets)
    G = jac.forward_accumulate(curlyJs, exogenous, outputs, required | set(unknowns))
    return G


'''Part 2: Computing times for irf'''


def step0_irf_preliminaries(block_list, dZ, unknowns, ss):
    curlyJs, required = jac.curlyJ_sorted(block_list, unknowns + list(dZ.keys()), ss)
    return curlyJs, required


def step1_irf_forward_HU(curlyJs, unknowns, targets, required):
    H_U_unpacked = jac.forward_accumulate(curlyJs, unknowns, targets, required)
    return H_U_unpacked


def step2_irf_forward_Z(curlyJs, dZ, targets, outputs, required):
    alloutputs = None
    if outputs is not None:
        alloutputs = set(outputs) | set(targets)
    J_curlyZ_dZ = jac.forward_accumulate(curlyJs, dZ, alloutputs, required)
    return J_curlyZ_dZ


def step3_irf_solve_dU(H_U_unpacked, J_curlyZ_dZ, unknowns, targets, T):
    H_U_packed = jac.pack_jacobians(H_U_unpacked, unknowns, targets, T)
    dU_packed = - np.linalg.solve(H_U_packed, jac.pack_vectors(J_curlyZ_dZ, targets, T))
    dU = jac.unpack_vectors(dU_packed, unknowns, T)
    return dU


def step4_irf_forward_U_combine(curlyJs, dU, J_curlyZ_dZ, outputs, required, T):
    J_curlyU_dU = jac.forward_accumulate(curlyJs, dU, outputs, required)
    if outputs is None:
        outputs = J_curlyZ_dZ.keys() | J_curlyU_dU.keys()
    dX = {o: J_curlyZ_dZ.get(o, np.zeros(T)) + J_curlyU_dU.get(o, np.zeros(T))
          for o in outputs}
    return dX


'''Part 3: Fake-news algorithm to obtaining Jacobian of the household block'''


def step_fake_0(back_step_fun, ss):
    # preliminary a: process back_step_funtion
    ssinput_dict, ssy_list, outcome_list, V_name = extract_info(back_step_fun, ss)

    # preliminary b: get sparse representation of asset policy rule, then distance between neighboring policy gridpoints
    a_pol_i, a_pol_pi = utils.interpolate_coord(ss['a_grid'], ss['a'])
    a_space = ss['a_grid'][a_pol_i + 1] - ss['a_grid'][a_pol_i]

    return ssinput_dict, ssy_list, outcome_list, V_name, a_pol_i, a_pol_pi, a_space


def step_fake_0_2d(back_step_fun, ss):
    # preliminary a: process back_step_funtion
    ssinput_dict, ssy_list, outcome_list, V_list = extract_info2d(back_step_fun, ss)

    # preliminary b: get sparse representation of asset policy rule, then distance between neighboring policy gridpoints
    a_i, a_pi = interpolate_coord_2d(ss['a'], ss['a_grid'])
    b_i_swapped, b_pi_swapped = interpolate_coord_2d(np.swapaxes(ss['b'], 1, 2), ss['b_grid'])
    b_i, b_pi = np.swapaxes(b_i_swapped, 1, 2), np.swapaxes(b_pi_swapped, 1, 2)
    a_space = ss['a_grid'][a_i + 1] - ss['a_grid'][a_i]
    b_space = ss['b_grid'][b_i + 1] - ss['b_grid'][b_i]

    return ssinput_dict, ssy_list, outcome_list, V_list, a_i, a_pi, b_i, b_pi, a_space, b_space


def step_fake_1(back_step_fun, shock_dict, ss, ssinput_dict, ssy_list, outcome_list, V_name, a_pol_i, a_space, T):
    # step 1: compute curlyY and curlyD (backward iteration) for each input i
    curlyYs, curlyDs = dict(), dict()
    for i, shock in shock_dict.items():
        curlyYs[i], curlyDs[i] = backward_iteration(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list,
                                                    V_name, ss['D'], ss['Pi'], a_pol_i, a_space, T)

    return curlyYs, curlyDs


def backward_step(dinput_dict, back_step_fun, ssinput_dict, ssy_list, outcome_list, D, Pi, a_pol_i, a_space, h=1E-4):
    # shock perturbs policies
    curlyV, da, *dy_list = utils.numerical_diff(back_step_fun, ssinput_dict, dinput_dict, h, ssy_list)

    # which affects the distribution tomorrow
    da_pol_pi = -da / a_space
    curlyD = utils.forward_step_shock_1d(D, Pi.T, a_pol_i, da_pol_pi)

    # and the aggregate outcomes today
    curlyY = {name: np.vdot(D, dy) for name, dy in zip(outcome_list, [da] + dy_list)}

    return curlyV, curlyD, curlyY


def backward_iteration(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list, V_name, D, Pi, a_pol_i, a_space, T):
    """Iterate policy steps backward T times for a single shock."""
    # contemporaneous response to unit scalar shock
    curlyV, curlyD, curlyY = backward_step(shock, back_step_fun, ssinput_dict,
                                           ssy_list, outcome_list, D, Pi, a_pol_i, a_space)

    # infer dimensions from this and initialize empty arrays
    curlyDs = np.empty((T,) + curlyD.shape)
    curlyYs = {k: np.empty(T) for k in curlyY.keys()}

    # fill in current effect of shock
    curlyDs[0, ...] = curlyD
    for k in curlyY.keys():
        curlyYs[k][0] = curlyY[k]

    # fill in anticipation effects
    for t in range(1, T):
        curlyV, curlyDs[t, ...], curlyY = backward_step({V_name + '_p': curlyV}, back_step_fun, ssinput_dict,
                                                        ssy_list, outcome_list, D, Pi, a_pol_i, a_space)
        for k in curlyY.keys():
            curlyYs[k][t] = curlyY[k]

    return curlyYs, curlyDs


def step_fake_1_2d(back_step_fun, shock_dict, ss, ssinput_dict, ssy_list, outcome_list, V_list, a_i, a_pi, b_i, b_pi,
                   a_space,
                   b_space, T):
    # step 1: compute curlyY and curlyD (backward iteration) for each input i
    curlyYs, curlyDs = dict(), dict()
    for i, shock in shock_dict.items():
        curlyYs[i], curlyDs[i] = backward_iteration2d(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list,
                                                      V_list,
                                                      ss['D'], ss['Pi'], a_i, b_i, a_pi, b_pi, a_space, b_space, T)

    return curlyYs, curlyDs


def backward_iteration2d(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list, V_list, D, Pi,
                         a_i, b_i, a_pi, b_pi, a_space, b_space, T):
    """Iterate policy steps backward T times for a single shock."""
    # contemporaneous response to unit scalar shock
    curlyVa, curlyVb, curlyD, curlyY = backward_step2d(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list, D,
                                                       Pi,
                                                       a_i, b_i, a_pi, b_pi, a_space, b_space)

    # infer dimensions from this and initialize empty arrays
    curlyDs = np.empty((T,) + curlyD.shape)
    curlyYs = {k: np.empty(T) for k in curlyY.keys()}

    # fill in current effect of shock
    curlyDs[0, ...] = curlyD
    for k in curlyY.keys():
        curlyYs[k][0] = curlyY[k]

    # fill in anticipation effects
    for t in range(1, T):
        curlyVa, curlyVb, curlyDs[t, ...], curlyY = backward_step2d({V_list[0] + '_p': curlyVa,
                                                                     V_list[1] + '_p': curlyVb},
                                                                    back_step_fun, ssinput_dict, ssy_list, outcome_list,
                                                                    D, Pi, a_i, b_i, a_pi, b_pi, a_space, b_space)
        for k in curlyY.keys():
            curlyYs[k][t] = curlyY[k]

    return curlyYs, curlyDs


def backward_step2d(dinput_dict, back_step_fun, ssinput_dict, ssy_list, outcome_list, D, Pi, a_i, b_i, a_pi, b_pi,
                    a_space, b_space, h=1E-4):
    # shock perturbs policies
    curlyVa, curlyVb, da, db, *dy_list = utils.numerical_diff(back_step_fun, ssinput_dict, dinput_dict, h, ssy_list)

    # which affects the distribution tomorrow
    da_pi = -da / a_space
    db_pi = -db / b_space
    curlyDmid = forward_step_policy_shock_2d(D, a_i, b_i, a_pi, b_pi, da_pi, db_pi)
    curlyD = (curlyDmid.T @ Pi).T

    # and the aggregate outcomes today
    curlyY = {name: np.vdot(D, dy) for name, dy in zip(outcome_list, [da, db] + dy_list)}

    return curlyVa, curlyVb, curlyD, curlyY


def step_fake_2(ss, outcome_list, ssy_list, a_pol_i, a_pol_pi, T):
    # step 2: compute prediction vectors curlyP (forward iteration) for each outcome o
    curlyPs = dict()
    for o, ssy in zip(outcome_list, ssy_list[1:]):
        curlyPs[o] = forward_iteration_transpose(ssy, ss['Pi'], a_pol_i, a_pol_pi, T)
    return curlyPs


def forward_iteration_transpose(y_ss, Pi, a_pol_i, a_pol_pi, T):
    """Iterate transpose forward T steps to get full set of prediction vectors for a given outcome."""
    curlyPs = np.empty((T,) + y_ss.shape)
    curlyPs[0, ...] = y_ss
    for t in range(1, T):
        curlyPs[t, ...] = utils.forward_step_transpose_1d(curlyPs[t - 1, ...], Pi, a_pol_i, a_pol_pi)
    return curlyPs


def step_fake_2_2d(ss, outcome_list, ssy_list, a_i, b_i, a_pi, b_pi, T):
    # step 2: compute prediction vectors curlyP (forward iteration) for each outcome o
    curlyPs = dict()
    for o, ssy in zip(outcome_list, ssy_list[2:]):  # outcome list is shorter!
        curlyPs[o] = forward_iteration_transpose2d(ssy, ss['Pi'].T, a_i, b_i, a_pi, b_pi, T)

    return curlyPs


def forward_iteration_transpose2d(y_ss, Pi_T, a_i, b_i, a_pi, b_pi, T):
    """Iterate transpose forward T steps to get full set of prediction vectors for a given outcome."""
    curlyPs = np.empty((T,) + y_ss.shape)
    curlyPs[0, ...] = y_ss
    for t in range(1, T):
        Dmid = (curlyPs[t - 1, ...].T @ Pi_T).T
        curlyPs[t, ...] = forward_step_transpose_2d(Dmid, a_i, b_i, a_pi, b_pi)
    return curlyPs


def build_F(curlyYs, curlyDs, curlyPs):
    T = curlyDs.shape[0]
    F = np.empty((T, T))
    F[0, :] = curlyYs
    F[1:, :] = curlyPs[:T - 1, ...].reshape((T - 1, -1)) @ curlyDs.reshape((T, -1)).T
    return F


def step_fake_3(shock_dict, outcome_list, curlyYs, curlyDs, curlyPs):
    # step 3: make fake news matrix for each outcome-input pair
    F = {o: {} for o in outcome_list}
    for o in outcome_list:
        for i in shock_dict:
            F[o][i] = build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
    return F


def step_fake_3_2d(shock_dict, outcome_list, curlyYs, curlyDs, curlyPs):
    # step 3-4: make fake news matrix and Jacobian for each outcome-input pair
    F = {o: {} for o in outcome_list}
    for o in outcome_list:
        for i in shock_dict:
            F[o][i] = build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
    return F

@njit
def J_from_F(F):
    J = F.copy()
    for t in range(1, J.shape[0]):
        J[1:, t] += J[:-1, t - 1]
    return J


def step_fake_4(F, shock_dict, outcome_list):
    # step 4: turn fake news matrix into jacobian for each outcome-input pair
    J = {o: {} for o in outcome_list}
    for o in outcome_list:
        for i in shock_dict:
            J[o][i] = J_from_F(F[o][i])
    return J


def step_fake_4_2d(F, shock_dict, outcome_list):
    # step 4: turn fake news matrix into jacobian for each outcome-input pair
    J = {o: {} for o in outcome_list}
    for o in outcome_list:
        for i in shock_dict:
            J[o][i] = J_from_F(F[o][i])
    return J


def all_JFs(back_step_fun, ss, T, shock_dict):
    # preliminary a: process back_step_funtion
    ssinput_dict, ssy_list, outcome_list, V_name = extract_info(back_step_fun, ss)

    # preliminary b: get sparse representation of asset policy rule, then distance between neighboring policy gridpoints
    a_pol_i, a_pol_pi = utils.interpolate_coord(ss['a_grid'], ss['a'])
    a_space = ss['a_grid'][a_pol_i + 1] - ss['a_grid'][a_pol_i]

    # step 1: compute curlyY and curlyD (backward iteration) for each input i
    curlyYs, curlyDs = dict(), dict()
    for i, shock in shock_dict.items():
        curlyYs[i], curlyDs[i] = backward_iteration(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list,
                                                    V_name, ss['D'], ss['Pi'], a_pol_i, a_space, T)

    # step 2: compute prediction vectors curlyP (forward iteration) for each outcome o
    curlyPs = dict()
    for o, ssy in zip(outcome_list, ssy_list[1:]):
        curlyPs[o] = forward_iteration_transpose(ssy, ss['Pi'], a_pol_i, a_pol_pi, T)

    # step 3: make fake news matrix and Jacobian for each outcome-input pair
    J = {o: {} for o in outcome_list}
    F = {o: {} for o in outcome_list}
    for o in outcome_list:
        for i in shock_dict:
            F[o][i] = build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
            J[o][i] = J_from_F(F[o][i])

    # remap outcomes to capital letters to avoid conflicts
    for k in list(J.keys()):
        K = k.upper()
        J[K] = J.pop(k)
        F[K] = F.pop(k)

    # report Jacobians
    return J, F


'''Part 4: Direct method to obtaining jacobian of the household block'''


def get_J_direct(household, inputs, outputs, ss, T, cols, h=1E-4):
    # make J as a nested dict where J[o][i] is initially-empty T*T Jacobian
    J_direct = {o: {i: np.empty((T, T)) for i in inputs} for o in outputs}

    # run td once without any shocks to get paths to subtract against (better than subtracting by ss since ss not exact)
    td_noshock = household.td(ss, beta=np.full(T, ss['beta']))

    for i in inputs:
        # simulate with respect to a shock at each date up to T
        for t in cols:
            td_out = household.td(ss, **{i: ss[i] + h * (np.arange(T) == t)})

            # store results as column t of J[o][i] for each outcome o
            for o in outputs:
                J_direct[o][i][:, t] = (td_out[o] - td_noshock[o]) / h

    return J_direct


def ks_backward(ss, backward_iterate, **kwargs):
    # infer T from kwargs, check that all shocks have same length
    shock_lengths = [x.shape[0] for x in kwargs.values()]
    assert shock_lengths[1:] == shock_lengths[:-1], 'Shocks with different length.'
    T = shock_lengths[0]

    # ss dict only with inputs of backward_iterate
    input_names = ['Va_p', 'Pi_p', 'a_grid', 'e_grid', 'r', 'w', 'beta', 'eis']
    ssinput_dict = {}
    for k in input_names:
        if k.endswith('_p'):
            ssinput_dict[k] = ss[k[:-2]]
        else:
            ssinput_dict[k] = ss[k]

    # make new dict of all the ss that are not shocked
    fixed_inputs = {k: v for k, v in ssinput_dict.items() if k not in kwargs}

    # allocate empty arrays to store results
    Va_path, a_path, c_path, D_path = (np.empty((T,) + ss['a'].shape) for _ in range(4))

    # backward iteration
    for t in reversed(range(T)):
        if t == T - 1:
            Va_p = ssinput_dict['Va_p']
        else:
            Va_p = Va_path[t + 1, ...]

        backward_inputs = {**fixed_inputs, **{k: v[t, ...] for k, v in kwargs.items()}, 'Va_p': Va_p}  # order matters
        Va_path[t, ...], a_path[t, ...], c_path[t, ...] = backward_iterate(**backward_inputs)

    return Va_path, a_path, c_path, D_path


def ks_forward(ss, T, Va_path, a_path, c_path, D_path):
    # forward iteration
    Pi_T = ss['Pi'].T.copy()
    D_path[0, ...] = ss['D']
    for t in range(T):
        a_pol_i, a_pol_pi = utils.interpolate_coord(ss['a_grid'], a_path[t, ...])
        if t < T - 1:
            D_path[t + 1, ...] = utils.forward_step_1d(D_path[t, ...], Pi_T, a_pol_i, a_pol_pi)

    # return paths and aggregates
    return {'Va': Va_path, 'a': a_path, 'c': c_path, 'D': D_path,
            'A': np.sum(D_path * a_path, axis=(1, 2)), 'C': np.sum(D_path * c_path, axis=(1, 2))}


def ha_backward(ss, backward_iterate, **kwargs):
    # infer T from kwargs, check that all shocks have same length
    shock_lengths = [x.shape[0] for x in kwargs.values()]
    assert shock_lengths[1:] == shock_lengths[:-1], 'Shocks with different length.'
    T = shock_lengths[0]

    # get steady state inputs
    ssinput_dict, _, _, _ = extract_info(backward_iterate, ss)

    # make new dict of all the ss that are not shocked
    fixed_inputs = {k: v for k, v in ssinput_dict.items() if k not in kwargs}

    # allocate empty arrays to store results
    Va_path, a_path, c_path, n_path, ns_path, D_path = (np.empty((T,) + ss['a'].shape) for _ in range(6))

    # backward iteration
    for t in reversed(range(T)):
        if t == T - 1:
            Va_p = ssinput_dict['Va_p']
        else:
            Va_p = Va_path[t + 1, ...]

        backward_inputs = {**fixed_inputs, **{k: v[t, ...] for k, v in kwargs.items()}, 'Va_p': Va_p}  # order matters
        Va_path[t, ...], a_path[t, ...], c_path[t, ...], n_path[t, ...], ns_path[t, ...] = backward_iterate(
            **backward_inputs)

    return Va_path, a_path, c_path, n_path, ns_path, D_path


def ha_forward(ss, T, Va_path, a_path, c_path, n_path, ns_path, D_path):
    # forward iteration
    Pi_T = ss['Pi'].T.copy()
    D_path[0, ...] = ss['D']
    for t in range(T):
        a_pol_i, a_pol_pi = utils.interpolate_coord(ss['a_grid'], a_path[t, ...])
        if t < T - 1:
            D_path[t + 1, ...] = utils.forward_step_1d(D_path[t, ...], Pi_T, a_pol_i, a_pol_pi)

    # return paths and aggregates
    return {'Va': Va_path, 'a': a_path, 'c': c_path, 'n': n_path, 'ns': ns_path, 'D': D_path,
            'A': np.sum(D_path * a_path, axis=(1, 2)), 'C': np.sum(D_path * c_path, axis=(1, 2)),
            'N': np.sum(D_path * n_path, axis=(1, 2)), 'NS': np.sum(D_path * ns_path, axis=(1, 2))}


def ha2_backward(ss, backward_iterate, **kwargs):
    # infer T from kwargs, check that all shocks have same length
    shock_lengths = [x.shape[0] for x in kwargs.values()]
    assert shock_lengths[1:] == shock_lengths[:-1], 'Shocks with different length.'
    T = shock_lengths[0]

    # ss dict only with inputs of backward_iterate
    ssinput_dict = extract_info(backward_iterate, ss)[0]

    # make new dict of all the ss that are not shocked
    fixed_inputs = {k: v for k, v in ssinput_dict.items() if k not in kwargs}

    # allocate empty arrays to store results
    Va_t, Vb_t, a_t, b_t, c_t, m_t, D_treated, D_control = (np.empty((T,) + ss['a'].shape) for _ in range(8))

    # backward iteration
    for t in reversed(range(T)):
        if t == T - 1:
            Va_p = ssinput_dict['Va_p']
            Vb_p = ssinput_dict['Vb_p']
        else:
            Va_p = Va_t[t + 1, ...]
            Vb_p = Vb_t[t + 1, ...]

        backward_inputs = {**fixed_inputs, **{k: v[t, ...] for k, v in kwargs.items()}, 'Va_p': Va_p, 'Vb_p': Vb_p}
        Va_t[t, ...], Vb_t[t, ...], a_t[t, ...], b_t[t, ...], c_t[t, ...], m_t[t, ...] = backward_iterate(
            **backward_inputs)

    return Va_t, Vb_t, a_t, b_t, c_t, m_t, D_treated, D_control


def ha2_forward(ss, T, Va_t, Vb_t, a_t, b_t, c_t, m_t, D_treated, D_control):
    # sparse representation of ss policy
    a_i_ss, a_pi_ss = interpolate_coord_2d(ss['a'], ss['a_grid'])
    b_i_swapped_ss, b_pi_swapped_ss = interpolate_coord_2d(np.swapaxes(ss['b'], 1, 2), ss['b_grid'])
    b_i_ss, b_pi_ss = np.swapaxes(b_i_swapped_ss, 1, 2), np.swapaxes(b_pi_swapped_ss, 1, 2)

    # forward iteration
    D_treated[0, ...] = ss['D']
    D_control[0, ...] = ss['D']
    for t in range(T):
        if t < T - 1:
            # treated
            a_i, a_pi = interpolate_coord_2d(a_t[t, ...], ss['a_grid'])
            b_i_swapped, b_pi_swapped = interpolate_coord_2d(np.swapaxes(b_t[t, ...], 1, 2), ss['b_grid'])
            b_i, b_pi = np.swapaxes(b_i_swapped, 1, 2), np.swapaxes(b_pi_swapped, 1, 2)
            D_treated[t + 1, ...] = forward_step_2d(D_treated[t, ...], ss['Pi'], a_i, b_i, a_pi, b_pi)

            # control
            D_control[t + 1, ...] = forward_step_2d(D_control[t, ...], ss['Pi'], a_i_ss, b_i_ss, a_pi_ss, b_pi_ss)

    # take out potentially compounding error
    D_t = D_treated - D_control + ss['D']

    # return paths and aggregates
    return {'c': c_t, 'a': a_t, 'b': b_t, 'm': m_t, 'D': D_t,
            'C': np.sum(D_t * c_t, axis=(1, 2, 3)), 'A': np.sum(D_t * a_t, axis=(1, 2, 3)),
            'B': np.sum(D_t * b_t, axis=(1, 2, 3)), 'M': np.sum(D_t * m_t, axis=(1, 2, 3))}


'''Part 5: General purpose functions'''


def extract_info(back_step_fun, ss):
    """Process source code of a one-asset backward iteration function."""
    V_name, *outcome_list = re.findall('return (.*?)\n',
                                       inspect.getsource(back_step_fun))[-1].replace(' ', '').split(',')

    ssy_list = [ss[k] for k in [V_name] + outcome_list]

    input_names = inspect.getfullargspec(back_step_fun).args
    ssinput_dict = {}
    for k in input_names:
        if k.endswith('_p'):
            ssinput_dict[k] = ss[k[:-2]]
        else:
            ssinput_dict[k] = ss[k]

    return ssinput_dict, ssy_list, outcome_list, V_name


def extract_info2d(back_step_fun, ss):
    """Process source code of a two asset backward iteration function."""
    V1_name, V2_name, *outcome_list = re.findall('return (.*?)\n',
                                                 inspect.getsource(back_step_fun))[-1].replace(' ', '').split(',')

    V_list = [V1_name, V2_name]
    ssy_list = [ss[k] for k in V_list + outcome_list]

    input_names = inspect.getfullargspec(back_step_fun).args
    ssinput_dict = {}
    for k in input_names:
        if k.endswith('_p'):
            ssinput_dict[k] = ss[k[:-2]]
        else:
            ssinput_dict[k] = ss[k]

    return ssinput_dict, ssy_list, outcome_list, V_list


@njit
def interpolate_coord_2d(x, xq):
    """Linear interpolation along last dimension of 3-dimensional array.

    Parameters
    ----------
    x    : array(z, b, a), data points increasing along axis=2
    xq   : array(k); query points increasing

    Returns
    -------
    x_i  : array(z, b, a}; index of lower bracketing data point
    x_pi : array(z, b, a); weight on x_i
    """

    nZ, nB, nA = x.shape
    nxq = xq.shape[0]
    x_i = np.empty(x.shape, dtype=np.int64)
    x_pi = np.empty_like(x)

    for iz in range(nZ):
        for ib in range(nB):
            ixp = 1
            xp_last = xq[0]
            xp_cur = xq[1]
            for ia in range(nA):
                # iterate through every point in x
                x_cur = x[iz, ib, ia]

                while ixp < nxq - 1:
                    # now iterate forward on points in xp until we find one greater than x_cur
                    if xp_cur >= x_cur:
                        # we've found an ixp such that xp[ixp-1] <= x[ix] < xp[ixp],
                        # unless we're outside xp[0] and xp[-1]
                        break
                    ixp += 1
                    xp_last = xp_cur
                    xp_cur = xq[ixp]

                # find the fraction assigned to ixp_1 vs ixp
                x_i[iz, ib, ia] = ixp - 1
                x_pi[iz, ib, ia] = (xp_cur - x_cur) / (xp_cur - xp_last)

    return x_i, x_pi


@njit
def forward_step_policy_shock_2d(Dss, ap_i, bp_i, ap_pi, bp_pi, ap_pi_shock, bp_pi_shock):
    """Update distribution of agents with policy functions perturbed around ss."""
    nZ, nB, nA = Dss.shape
    Dnew = np.zeros_like(Dss)
    for iz in range(nZ):
        for ib in range(nB):
            for ia in range(nA):
                ibp = bp_i[iz, ib, ia]
                iap = ap_i[iz, ib, ia]
                beta = bp_pi[iz, ib, ia]
                alpha = ap_pi[iz, ib, ia]

                dalpha = ap_pi_shock[iz, ib, ia] * Dss[iz, ib, ia]
                dbeta = bp_pi_shock[iz, ib, ia] * Dss[iz, ib, ia]

                Dnew[iz, ibp, iap] += dalpha * beta + alpha * dbeta
                Dnew[iz, ibp + 1, iap] += dalpha * (1 - beta) - alpha * dbeta
                Dnew[iz, ibp, iap + 1] += dbeta * (1 - alpha) - beta * dalpha
                Dnew[iz, ibp + 1, iap + 1] -= dalpha * (1 - beta) + dbeta * (1 - alpha)
    return Dnew


@njit
def forward_step_transpose_2d(Dmid, a_i, b_i, a_pi, b_pi):
    """Efficient implementation of D_t =  Lam_{t-1} @ D_{t-1}' using sparsity of Lam_{t-1}."""
    # D = Pi @ D
    nZ, nB, nA = Dmid.shape
    Dnew = np.empty_like(Dmid)
    for iz in range(nZ):
        for ib in range(nB):
            for ia in range(nA):
                ibp = b_i[iz, ib, ia]
                iap = a_i[iz, ib, ia]
                beta = b_pi[iz, ib, ia]
                alpha = a_pi[iz, ib, ia]

                Dnew[iz, ib, ia] = alpha * beta * Dmid[iz, ibp, iap] + alpha * (1 - beta) * Dmid[iz, ibp + 1, iap] + \
                                   (1 - alpha) * beta * Dmid[iz, ibp, iap + 1] + \
                                   (1 - alpha) * (1 - beta) * Dmid[iz, ibp + 1, iap + 1]
    return Dnew


@njit
def forward_step_endo_2d(D, a_i, b_i, a_pi, b_pi):
    """Forward iterate endogenous states: D(z, b, a) to D(z, b', a')."""
    nZ, nB, nA = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(nZ):
        for ib in range(nB):
            for ia in range(nA):
                ibp = b_i[iz, ib, ia]
                iap = a_i[iz, ib, ia]
                beta = b_pi[iz, ib, ia]
                alpha = a_pi[iz, ib, ia]

                Dnew[iz, ibp, iap] += alpha * beta * D[iz, ib, ia]
                Dnew[iz, ibp + 1, iap] += alpha * (1 - beta) * D[iz, ib, ia]
                Dnew[iz, ibp, iap + 1] += (1 - alpha) * beta * D[iz, ib, ia]
                Dnew[iz, ibp + 1, iap + 1] += (1 - alpha) * (1 - beta) * D[iz, ib, ia]
    return Dnew


def forward_step_2d(D, Pi, a_i, b_i, a_pi, b_pi):
    """Full forward iteration in two steps: D(z, b, a) to D(z, b', a') to D(z', b', a')."""
    Dmid = forward_step_endo_2d(D, a_i, b_i, a_pi, b_pi)
    results = (Dmid.T @ Pi).T
    return results
