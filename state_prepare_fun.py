import PhysModule as PhyM
import BasicFun as bf
import torch as tc
import numpy as np
from torch.optim.lr_scheduler import StepLR
import math
import time
import TimeEvolutionRandomField as ter
import NextsiteTimeEvolutionRandomField as nter
import LFTimeEvolutionRandomField as lter
import copy
from scipy.sparse import linalg
import ED.ExactDiagonalizationAlgorithm as ED
import matplotlib.pyplot as plt
import ComplexTensorFunSJR as ctf
# import test_tau as tt
import os
import text_t as tt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def state_prepare(state_f, state, evol_time, length, it_time, lr, dt_print, tol, para=None):
    """
    :param state_f:final state
    :param state:initial state
    :param evol_time:the evolution of time
    :param length:Spin chain length
    :param it_time:Evolutionary steps
    :param lr:learning rate
    :param dt_print:interval time
    :param tol:The convergence threshold
    :return:h
    """
    paraDef = dict()
    paraDef['record'] = 'record.log'
    paraDef['device'] = 'cpu'
    paraDef['dtype'] = tc.complex128

    if para is None:
        para = dict()
    para = bf.combine_dicts(paraDef, para)
    para['length'] = length
    para['tau'] = 2/2
    # para['time_tot'] = 1e-2
    para['time_tot'] = evol_time
    para['slice'] = round(para['time_tot'] / para['tau'])
    para['J'] = [0, 0, 1]
    para['hp'] = 1e-2
    # x1_axis = list()
    # y_f_layer_0 = list()
    # y_f_layer_1 = list()

    para['h'] = tt.pre_h().data
    # para['h'] = bf.load('./10bit_1.5_32_rand1.data', device=bf.choose_device('cuda'))['h']
    # print(para['h'])
    # para['h'] = para['h'] + para['hp']*tc.randn(para['h'].shape, dtype=para['dtype'])
    # para['h'] = tc.randn((3, length, 2), device=para['device'], dtype=para['dtype'])
    # para['h'] = tc.randn((2, length, para['slice']), device=para['device'], dtype=tc.float64)
    # print(para['h'])
    para['c_tau'] = tc.ones((1, 1), device=para['device'], dtype=para['dtype'])
    para['h'].requires_grad = True
    optimizer = tc.optim.Adam([para['h']], lr=lr)
    scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
    if type(state) is np.ndarray:
        state = tc.from_numpy(state)
    state = state.to(device=para['device'], dtype=tc.complex128)
    if type(state_f) is np.ndarray:
        state_f = tc.from_numpy(state_f)
    state_f = state_f.to(device=para['device'], dtype=tc.complex128)

    # a = tc.rand(4, dtype=tc.float64)
    # a.requires_grad = True
    # optimizer_a = tc.optim.Adam([a], lr=2e-4)
    # scheduler_a = StepLR(optimizer_a, step_size=10000, gamma=0.1)

    # state0 = np.load('2bit0.npy')
    # state0 = tc.from_numpy(state0).to(device=bf.choose_device(), dtype=tc.complex128).reshape(2, 2)
    # state__ = np.load('2bit+.npy')
    # state__ = tc.from_numpy(state__).to(device=bf.choose_device(), dtype=tc.complex128).reshape(2, 2)
    # state1_ = np.load('2bit1.npy')
    # state1_ = tc.from_numpy(state1_).to(device=bf.choose_device(), dtype=tc.complex128).reshape(2, 2)
    # state_ = np.load('2bit_.npy')
    # state_ = tc.from_numpy(state_).to(device=bf.choose_device(), dtype=tc.complex128).reshape(2, 2)

    # a = [2.8016e-01,  1.0761e+00,  8.0897e-08, -2.6501e-08]
    # print(a[0] * state0 + a[1] * state__ + a[2] * state1_ + a[3] * state_)
    # s = a[0] * state0 + a[1] * state__ + a[2] * state1_ + a[3] * state_
    # s = s/tc.norm(s)
    # print(s)
    # print(-(quantum_centual_entropy(s, 1)))
    # input()

    for t in range(it_time):
        para['c_tau'] = tc.ones((1, 1), device=para['device'], dtype=para['dtype'])
        state1 = ter.time_evolution_Heisenberg_chain(state, para)
        # state2 = ter.time_evolution_Heisenberg_chain(state, para)
        # a1 = a

        f = -(quantum_centual_entropy(state1, 5))
        # a1 = a1/tc.norm(a1)
        # state_new = (a[0] * state0 + a[1] * state__ + a[2] * state1_ + a[3] * state_)
        # state_new = state_new/tc.norm(state_new)
        # _norm = tc.sum(state_f * state1.reshape(state_f.shape).conj())

        # state_f = state_f.reshape(64)
        # loss = state1.flatten().dot(state_f.conj()).norm()

        # F = 1 - _norm

        if (t % dt_print) == 0:
            bf.fprint('t = ' + str(t) + ', f = ' + str(f), file=para['record'])
            # bf.fprint('t = ' + str(t) + ', F = ' + str(F), file=para['record'])
            # bf.fprint('t = ' + str(t) + ', a = ' + str(a), file=para['record'])

        f.backward()
        # F.backward()
        optimizer.step()
        # optimizer_a.step()
        optimizer.zero_grad()
        # optimizer_a.zero_grad()
        scheduler.step()
        # scheduler_a.step()

        bf.save('.', 'state.data', [state1], ['state'])
        bf.save('.', 'h.data', [para['h']], ['h'])
    return para['h']


def prepare_Heisenberg():
    """

    """
    para = dict()
    para['lattice'] = 'chain'
    para['BC'] = 'open'
    para['length'] = 10
    para['spin'] = 'half'
    para['jx'] = 1
    para['jy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['k'] = 1

    para = ED.parameters_quickED(para)

    v = ED.quickED_heisenberg(para)[1]
    # print(v.shape)
    # print(v)
    # v = tc.zeros([4096, 1])
    # v[0, 0] = 1
    # v[-1, 0] = 1
    # v = np.load('hs.npy')
    # v = v / tc.norm(v)
    # v = tc.randn([1024, 1])
    # v = v / tc.norm(v)
    # np.save('hs2.npy', v)
    # input()
    # v0 = np.load('v.npy')
    # print(v.shape)
    # v = v.reshape([65536, 1])
    # print(v)
    # print(v.shape)
    # x = bf.load('./state.data', device='cpu')
    # x = x['state']
    # v = x.reshape(1024, 1)
    # v0 = tc.zeros([64, 1])
    # v0[0, 0] = 1
    # v0 = v0.reshape([4096, 1])
    # v0[0, 0] = 1
    # np.save('state0.npy', v0)
    # v0 = v0 + 1e-4*tc.randn(v0.shape)
    v0 = np.load('10bit.npy')

    # v = np.load('z.npy')
    # v0 = np.load('z0.npy')
    if type(v) is np.ndarray:
        v = tc.from_numpy(v).to(device='cpu', dtype=tc.complex128)
    if type(v0) is np.ndarray:
        v0 = tc.from_numpy(v0).to(device='cpu', dtype=tc.complex128)
    # _norm = tc.sum(v * v0)
    # f0 = -tc.log(_norm ** 2)
    # f0 = 1-(_norm ** 2)
    # print(f0.data.item())
    # bf.fprint('f = ' + str(f0.item()))
    para1 = dict()
    para1['length'] = para['length']
    # v0 = v+1e-3*tc.randn(v.shape, device=v.device, dtype=v.dtype)
    # v0 = tc.randn(v.shape, device=v.device, dtype=v.dtype)
    v0 = v0 / tc.norm(v0)
    v = v / tc.norm(v)
    # print(v)
    # np.save('a.npy', v0)
    h = state_prepare(state_f=v, state=v0, evol_time=2, length=para1['length'],
                      it_time=20000, lr=1e-2, dt_print=10, tol=1e-6, para=para1)
    return h


def quantum_centual_entropy(quantum_state, n_site):
    quantum_state = quantum_state.reshape(pow(2, n_site), -1)
    _, s, _ = tc.svd(quantum_state, compute_uv=True)
    ent = entanglement_entropy(s)
    # s_shape = s.shape[0]
    # ent = tc.tensor(0)
    # for n in range(0, s_shape):
    #     s_ = -tc.pow(s[n], 2) * tc.log(tc.pow(s[n], 2))
    #     ent = ent + s_
    return ent


def entanglement_entropy(lm, eps=1e-15):
    lm1 = (lm ** 2 + eps).reshape(-1, )
    if type(lm1) is tc.Tensor:
        return tc.dot(-1 * lm1, tc.log2(lm1))
    else:
        return np.inner(-1 * lm1, np.log2(lm1))
