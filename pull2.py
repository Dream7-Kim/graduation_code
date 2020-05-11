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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def invm_plus(Pb,Pc):
    Pbc = Pb + Pc
    _Pbc = Pbc * np.array([-1,-1,-1,1])
    return np.sum(Pbc * _Pbc,axis=1)

def invm(Pbc):
    _Pbc = Pbc * np.array([-1,-1,-1,1])
    return np.sum(Pbc * _Pbc,axis=1)

def mcnpz(begin,end):
    amp = onp.load("data/mctruth.npz")
    # phif223 = amp['phif223'][begin:end,0:2]
    # phif222 = amp['phif222'][begin:end,0:2]
    # phif221 = amp['phif221'][begin:end,0:2]
    # phif201 = amp['phif201'][begin:end,0:2]
    # data_phif2 = np.asarray([phif201,phif221,phif222,phif223])
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

def MOD(phim,phiw,fm,fw,const,theta,rho,phif,phi,f):
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
#     print(sum.shape)
    return np.sum(dplex.dabs(sum),axis=1)


def weight(_phim,_phiw,_f0m,_f0w,_const1,_theta,_rho):
    const = np.asarray([[_const1],[1.]])
    rho = np.asarray([_rho])
    theta = np.asarray([_theta])
    phim = np.asarray([_phim])
    phiw = np.asarray([_phiw])
    f0m = np.asarray([_f0m])
    f0w = np.asarray([_f0w])
    d_phif0 = MOD(phim,phiw,f0m,f0w,const,theta,rho,data_phif0,data_phi,data_f)
    m_phif0 = MOD(phim,phiw,f0m,f0w,const,theta,rho,mc_phif0,mc_phi,mc_f)
    d_tmp = alladd(d_phif0)
    m_tmp = np.average(alladd(m_phif0))
    print("weight")
    return d_tmp/m_tmp

def likelihood(_phim,_phiw,_f0m,_f0w,_const1,_theta,_rho):
    const = np.asarray([[_const1],[1.]])
    rho = np.asarray([_rho])
    theta = np.asarray([_theta])
    phim = np.asarray([_phim])
    phiw = np.asarray([_phiw])
    f0m = np.asarray([_f0m])
    f0w = np.asarray([_f0w])
    d_phif0 = MOD(phim,phiw,f0m,f0w,const,theta,rho,data_phif0,data_phi,data_f)
    m_phif0 = MOD(phim,phiw,f0m,f0w,const,theta,rho,mc_phif0,mc_phi,mc_f)
    d_tmp = alladd(d_phif0)
    m_tmp = np.average(alladd(m_phif0))
    return -np.sum(wt*(np.log(d_tmp) - np.log(m_tmp)))

phim = 1.02
phiw = 0.01
f0m = 0.5
f0w = 0.4
f2m = 1.
f2w = 1.
const1 = 5.5 # 该参数拟合的好
const2 = 1
rho = 1.
theta = 0.5
vf0m = []
vf0w = []
vconst = []
ef0m = []
ef0w = []
econst = []
vvalue = []
# offset = onp.random.sample(1000)
offset = onp.random.randint(100,650,size=500)
ranm = onp.random.randint(10,100,size=500) - 50
ranw = onp.random.randint(10,100,size=500) - 50
ranc = onp.random.randint(10,50,size=500) - 25
cc = onp.arange(500)

m=(0,1,2,3,4,5,6)
grad_likelihood = jit(grad(likelihood,argnums=m))
jit_likelihood = jit(likelihood)
par=('phim','phiw','f0m','f0w','const1','rho','theta')

for i in cc:
    _offset = offset[i] * 1000
    k = ranm[i] / 1000
    l = ranw[i] / 1000
    j = ranc[i] / 100
    print('offset',_offset)
    print(i)
    b1 = _offset
    e1 = b1 + 1000
    b2 = e1
    e2 = b2 + 50000
    data_phif0,data_phi,data_f = mcnpz(b1,e1)
    mc_phif0,mc_phi,mc_f = mcnpz(b2,e2)
    wt = weight(phim,phiw,f0m,f0w,const1,theta,rho)
    likelihood(phim,phiw,f0m,f0w,const1,theta,rho)
    print("begin")
    m = iminuit.Minuit(jit_likelihood,
                    forced_parameters=par,
                    phim=1.02,phiw=0.01,f0m=0.5+k,f0w=0.4+l,
                    const1=5.5+j,rho=1.,theta=0.5,
                    fix_phim=True,fix_phiw=True,fix_rho=True,
                    fix_theta=True,
                    #    fix_const1=True,
                grad=grad_likelihood,
                error_phim=0.001,error_phiw=0.001,
                error_f0m=0.1, error_f0w=0.01, error_const1=0.1,
                error_rho=0.1, error_theta=0.1, errordef=0.5)
    # print(m.get_param_states())
    # print(m.migrad())
    # m.get_param_states()
    m.migrad()
    fvalue = m.values
    ferror = m.errors
    value = jit_likelihood(phim,phiw,fvalue['f0m'],fvalue['f0w'],fvalue['const1'],1.,0.5)
    # print('f0w',fvalue['f0w'])
    # print('f0m',ferror['f0m'])
    vf0m.append(fvalue['f0m'])
    vf0w.append(fvalue['f0w'])
    vconst.append(fvalue['const1'])
    vvalue.append(value)
    ef0m.append(ferror['f0m'])
    ef0w.append(ferror['f0w'])
    econst.append(ferror['const1'])


onp.savez("fit_result2",f0m=vf0m,f0w=vf0w,const=vconst,value=vvalue,offset=offset,ef0m=ef0m,ef0w=ef0w,econst=econst)
