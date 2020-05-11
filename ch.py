#!/usr/bin/env python
# coding: utf-8

import numpy as onp
import jax.numpy as np
from jax import vmap
from functools import partial
import time
from jax import jit
from jax import grad
import os
from jax.config import config
import dplex
import iminuit
import matplotlib.pyplot as plt
config.update("jax_enable_x64", True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def invm_plus(Pb,Pc):
    Pbc = Pb + Pc
    _Pbc = Pbc * np.array([-1,-1,-1,1])
    return np.sum(Pbc * _Pbc,axis=1)

def invm(Pbc):
    _Pbc = Pbc * np.array([-1,-1,-1,1])
    return np.sum(Pbc * _Pbc,axis=1)

def mcnpz(begin,end):
    amp = onp.load("data/mctruth.npz")
    phif223 = amp['phif223'][begin:end,0:2]
    phif222 = amp['phif222'][begin:end,0:2]
    phif221 = amp['phif221'][begin:end,0:2]
    phif201 = amp['phif201'][begin:end,0:2]
    data_phif2 = np.asarray([phif201,phif221,phif222,phif223])
    phif001 = amp['phif001'][begin:end,0:2]
    phif021 = amp['phif021'][begin:end,0:2]
    data_phif0 = np.asarray([phif001,phif021])
    mom = onp.load("data/mcmom.npz")
    Kp = mom['Kp'][begin:end,:]
    Km = mom['Km'][begin:end,:]
    Pip = mom['Pip'][begin:end,:]
    Pim = mom['Pim'][begin:end,:]
    data_f = Pip + Pim
    data_phi = Kp + Km
    data_phi = invm(data_phi)
    data_f = invm(data_f)
    return data_phif0,data_phi,data_f

def _BW(m_,w_,Sbc):
    gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
    k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
    l = Sbc.shape[0]
    temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
    return dplex.ddivide(k, temp)

def _phase(_theta, _rho):
    return dplex.dconstruct(_rho * np.cos(_theta), _rho * np.sin(_theta))

def BW(phim,phiw,fm,fw,phi,f):
    a = np.moveaxis(vmap(partial(_BW,Sbc=phi))(phim,phiw),1,0)
    b = np.moveaxis(vmap(partial(_BW,Sbc=f))(fm,fw),1,0)
    result = dplex.deinsum('ij,ij->ij',a,b)
    return result

def phase(_theta,_rho):
    result = vmap(_phase)(_theta,_rho)
    return result

def MOD(fm,fw,_const,phif,phi,f):
    phim = np.asarray([1.02])
    phiw = np.asarray([0.01])
    theta = np.asarray([1.])
    rho = np.asarray([1.])
    const = np.append(_const,1.).reshape(2,1)
    ph = np.moveaxis(phase(theta,rho), 1, 0)
    bw = BW(phim,phiw,fm,fw,phi,f)
    _phif = dplex.dtomine(np.einsum('ijk,il->ljk',phif,const))
    _phif = dplex.deinsum('ijk,i->ijk',_phif,ph)
    _phif = dplex.deinsum('ijk,ij->jk',_phif,bw)
    return _phif

def alladd(*mods):
    l = (mods[0].shape)[1]
    sum = onp.zeros(l*2*2).reshape(2,l,2)
    for num in mods:
        sum += num
    return np.sum(dplex.dabs(sum),axis=1)


def weight(args):
    f0m,f0w,const = np.split(args,3)
    d_phif0 = MOD(f0m,f0w,const,data_phif0,data_phi,data_f)
    m_phif0 = MOD(f0m,f0w,const,mc_phif0,mc_phi,mc_f)
    d_tmp = alladd(d_phif0)
    m_tmp = np.average(alladd(m_phif0))
    #print("weight")
    return d_tmp/m_tmp

def likelihood(args):
    f0m,f0w,const = np.split(args,3)
    d_phif0 = MOD(f0m,f0w,const,data_phif0,data_phi,data_f)
    m_phif0 = MOD(f0m,f0w,const,mc_phif0,mc_phi,mc_f)
    d_tmp = alladd(d_phif0)
    m_tmp = np.average(alladd(m_phif0))
    return -np.sum(wt*(np.log(d_tmp) - np.log(m_tmp)))

# f0m = 0.5
# f0w = 0.4
# const1 = 5.5 # 该参数拟合的好

vf0m = []
vf0w = []
vconst = []
ef0m = []
ef0w = []
econst = []
vvalue = []

offset = onp.random.randint(100,600,size=500)
ranm = onp.random.randint(100,size=500) - 50
ranw = onp.random.randint(100,size=500) - 50
ranc = onp.random.randint(100,size=500) - 50
cc = onp.arange(500)

wtarg = onp.asarray([0.5,0.4,5.5])

for i in cc:
    _offset = offset[i] * 1000
    k = ranm[i] / 1000
    l = ranw[i] / 1000
    j = ranc[i] / 1000
    #print('offset',_offset)
    #print(i)
    b1 = _offset
    e1 = b1 + 10000
    b2 = e1
    e2 = b2 + 100000
    data_phif0,data_phi,data_f = mcnpz(b1,e1)
    mc_phif0,mc_phi,mc_f = mcnpz(b2,e2)
    wt = weight(wtarg)
    likelihood(wtarg)
    grad_likelihood = jit(grad(likelihood))
    jit_likelihood = jit(likelihood)
    #print("begin")
    list_arg = np.asarray([0.5+k,0.4+l,5.5+j])
    list_error = (0.1, 0.1,0.1)
    m = iminuit.Minuit.from_array_func(likelihood, list_arg, list_error, errordef=0.5, grad=grad_likelihood)
    # print(m.get_param_states())
    # print(m.migrad())
    m.get_param_states()
    m.migrad()
    fvalue = m.values
    ferror = m.errors
    out = np.asarray([fvalue['x0'],fvalue['x1'],fvalue['x2']])
    value = jit_likelihood(out)
    # print('f0w',fvalue['x1'])
    # print('f0m',ferror['x0'])
    vf0m.append(fvalue['x0'])
    vf0w.append(fvalue['x1'])
    vconst.append(fvalue['x2'])
    vvalue.append(value)
    ef0m.append(ferror['x0'])
    ef0w.append(ferror['x1'])
    econst.append(ferror['x2'])

onp.savez("fit_result",f0m=vf0m,f0w=vf0w,const=vconst,value=vvalue,ef0m=ef0m,ef0w=ef0w,econst=econst)
