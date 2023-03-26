import math

import torch as tc
import numpy as np
import BasicFun as bf
import PhysModule as phy
import ComplexTensorFunSJR as ctf
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pure_state_evolution(state, gates, which_where):
    # print(which_where)
    """
    Evolve the state by several gates0.
    :param state: initial state
    :param gates: quantum gates
    :param which_where: [which gate, which spin, which spin]
    :return: evolved state
    Example: which_where = [[0, 1, 2], [1, 0, 1]] means gate 0 on spins
    1 and 2, and gate 1 on spins 0 and 1
    """
    def pure_state_evolution_one_gate(v, g, pos):
        ind = list(range(len(pos), 2*len(pos)))
        v = tc.tensordot(v, g, [pos, ind])
        ind0 = list(range(v.ndimension()))
        for nn in range(len(pos)):
            ind0.remove(pos[nn])
        ind0 += pos
        order = list(np.argsort(ind0))
        # print(order)
        return v.permute(order)

    for n in range(len(which_where)):
        state = pure_state_evolution_one_gate(
             state, gates[which_where[n][0]], which_where[n][1:])
    return state


def gates_from_magnetic_fields(hx, hy, hz, tau, spin, device, dtype):
    # hx（或hy,hz）是一个list，且len(h)=自旋个数
    op = phy.spin_operators(spin, if_list=False, device=device, dtp=tc.complex128)
    length = hx.numel()  # 自旋个数
    gates = list()
    for n in range(length):
        tmp = tau * (hx[n] * op['sx'] + hy[n] * op['sy'] + hz[n] * op['sz'])
        # print(tmp)
        gates.append(tc.matrix_exp(-1j*tmp))
    return gates


# def hamilt2gates(hamilt, tau, d=2):
#     gate_ = [tau*hamilt[1], tau*hamilt[0]]
#     gate2 = ctf.expm(gate_, order=4)
#     gate2 = [gate2[0].reshape([d] * 4), gate2[1].reshape([d] * 4)]
#     return gate2

def hamilt2gates(hamilt, tau, d=2):
    gate_ = -1j*tau*hamilt
    gate2 = tc.matrix_exp(gate_)
    gate2 = gate2.reshape([d] * 4)
    return gate2


def time_evolution_Heisenberg_chain(state=None, para=None):
    para_def = dict()
    para_def['J'] = [0, 0, 1]
    para_def['h'] = None  # tc.Tensor with shape (3 * length * time steps); h[0, 2, 6]代表第二个自旋在t=6时x方向磁场
    para_def['length'] = 6
    para_def['spin'] = 'half'
    para_def['BC'] = 'open'
    para_def['time_tot'] = 10
    para_def['tau'] = 1e-2
    para_def['tau_req'] = 1e-2
    para_def['c_tau'] = 1.0
    # para_def['time_tot'] = para_def['tau']
    para_def['print_dtime'] = 1
    para_def['device'] = bf.choose_device()
    para_def['dtype'] = tc.complex128

    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['d'] = phy.from_spin2phys_dim(para['spin'])
    para['time_it'] = round(para['time_tot'] / para['tau'])
    para['print_time_it'] = round(para['print_dtime'] / para['tau'])
    para['device'] = bf.choose_device(para['device'])
    assert para['time_it'] >= 1, '时间演化步数小于1'

    state = state.reshape([para['d']] * para['length'])
    which_where_gates1 = [[n, n] for n in range(para['length'])]
    M = [
        [1750.26, 0, 0, 0, 0, 0, 0],
        [40.8, 14930.1, 0, 0, 0, 0, 0],
        [1.6, 69.5, 12199.9, 0, 0, 0, 0],
        [8.47, 1.4, 71.04, 17173.7, 0, 0, 0],
        [4.0, 155.6, -1.8, 6.5, 2785.85, 0, 0],
        [6.64, -0.7, 162.9, 3.3, 15.81, 2320.25, 0],
        [128 * 3, -7.1 * 3, 6.6 * 3, -0.9 * 3, 6.9 * 3, -1.7 * 3, 718.487]
    ]
    O = [9500, 9500, 9500, 9500, 1800, 1800, 1800]

    for t in range(para['time_it']):
        hz_ = tc.zeros((1, para['length']), device=para['device'], dtype=tc.float64)
        for i in range(7):
            hz_[0, i] = (M[i][i] - O[i]) * 2 * math.pi

        hx1 = para['h'][0, 0:1, t % para['h'].shape[2]]
        hx2 = para['h'][0, 1:2, t % para['h'].shape[2]]
        hy1 = para['h'][1, 0:1, t % para['h'].shape[2]]
        hy2 = para['h'][1, 1:2, t % para['h'].shape[2]]
        hz1 = hz_[0, 0:1]
        hz2 = hz_[0, 1:2]
        hz3 = hz_[0, 2:3]
        hz4 = hz_[0, 3:4]
        hz5 = hz_[0, 4:5]
        hz6 = hz_[0, 5:6]
        hz7 = hz_[0, 6:7]
        which_where_gates2 = []
        gate2 = []
        gate2_rest = []
        it_slice = int(para['tau'] / para['tau_req'])
        tau_rest = para['tau'] - it_slice * para['tau_req']
        ii = 0
        for i in range(para['length']):
            for j in range(i):
                hamilt_ = phy.hamiltonian_heisenberg(para['spin'], para['J'][0], para['J'][1], 2 * math.pi * M[i][j],
                                                     [0, 0], [0, 0], [0, 0], para['device'], tc.complex128)
                which_where_gates2.append([ii, j, i])
                ii += 1
                if para['tau'] > para['tau_req']:
                    gate2.append(hamilt2gates(hamilt_, para['tau_req']))
                    if tau_rest > 1e-8:
                        gate2_rest.append(hamilt2gates(hamilt_, tau_rest))
                else:
                    gate2.append(hamilt2gates(hamilt_, para['tau']))

        if para['tau'] > para['tau_req']:
            for ts in range(it_slice):
                state = pure_state_evolution(state, gate2, which_where_gates2)
                gates1 = gates_from_magnetic_fields(hx1, hy1, hz1, para['tau_req'], para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates2 = gates_from_magnetic_fields(hx1, hy1, hz2, para['tau_req'], para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates3 = gates_from_magnetic_fields(hx1, hy1, hz3, para['tau_req'], para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates4 = gates_from_magnetic_fields(hx1, hy1, hz4, para['tau_req'], para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates5 = gates_from_magnetic_fields(hx2, hy2, hz5, para['tau_req'], para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates6 = gates_from_magnetic_fields(hx2, hy2, hz6, para['tau_req'], para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates7 = gates_from_magnetic_fields(hx2, hy2, hz7, para['tau_req'], para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates1.extend(gates2)
                gates1.extend(gates3)
                gates1.extend(gates4)
                gates1.extend(gates5)
                gates1.extend(gates6)
                gates1.extend(gates7)
                state = pure_state_evolution(state, gates1, which_where_gates1)
            tau_rest = para['tau'] - it_slice*0.1
            if tau_rest > 1e-6:
                state = pure_state_evolution(state, gate2, which_where_gates2)
                gates1 = gates_from_magnetic_fields(hx1, hy1, hz1, tau_rest, para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates2 = gates_from_magnetic_fields(hx1, hy1, hz2, tau_rest, para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates3 = gates_from_magnetic_fields(hx1, hy1, hz3, tau_rest, para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates4 = gates_from_magnetic_fields(hx1, hy1, hz4, tau_rest, para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates5 = gates_from_magnetic_fields(hx2, hy2, hz5, tau_rest, para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates6 = gates_from_magnetic_fields(hx2, hy2, hz6, tau_rest, para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates7 = gates_from_magnetic_fields(hx2, hy2, hz7, tau_rest, para['spin'],
                                                    device=para['device'], dtype=para['dtype'])
                gates1.extend(gates2)
                gates1.extend(gates3)
                gates1.extend(gates4)
                gates1.extend(gates5)
                gates1.extend(gates6)
                gates1.extend(gates7)
                state = pure_state_evolution(state, gates1, which_where_gates1)
        else:
            state = pure_state_evolution(state, gate2, which_where_gates2)
            gates1 = gates_from_magnetic_fields(hx1, hy1, hz1, para['tau'], para['spin'],
                                                device=para['device'], dtype=para['dtype'])
            gates2 = gates_from_magnetic_fields(hx1, hy1, hz2, para['tau'], para['spin'],
                                                device=para['device'], dtype=para['dtype'])
            gates3 = gates_from_magnetic_fields(hx1, hy1, hz3, para['tau'], para['spin'],
                                                device=para['device'], dtype=para['dtype'])
            gates4 = gates_from_magnetic_fields(hx1, hy1, hz4, para['tau'], para['spin'],
                                                device=para['device'], dtype=para['dtype'])
            gates5 = gates_from_magnetic_fields(hx2, hy2, hz5, para['tau'], para['spin'],
                                                device=para['device'], dtype=para['dtype'])
            gates6 = gates_from_magnetic_fields(hx2, hy2, hz6, para['tau'], para['spin'],
                                                device=para['device'], dtype=para['dtype'])
            gates7 = gates_from_magnetic_fields(hx2, hy2, hz7, para['tau'], para['spin'],
                                                device=para['device'], dtype=para['dtype'])
            gates1.extend(gates2)
            gates1.extend(gates3)
            gates1.extend(gates4)
            gates1.extend(gates5)
            gates1.extend(gates6)
            gates1.extend(gates7)
            # print(gates1)
            state = pure_state_evolution(state, gates1, which_where_gates1)
    return state


def run():
    para = dict()
    time_evolution_Heisenberg_chain(para)


if __name__ == '__main__':
    run()
