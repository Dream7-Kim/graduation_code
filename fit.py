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

config.update("jax_enable_x64", True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def invm_plus(Pb,Pc):
    Pbc = Pb + Pc
    _Pbc = Pbc * np.array([-1,-1,-1,1])
    return np.sum(Pbc * _Pbc,axis=1)

def invm(Pbc):
    _Pbc = Pbc * np.array([-1,-1,-1,1])
    return np.sum(Pbc * _Pbc,axis=1)

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


# MC truth 数据
size1 = 40000
amp = onp.load("data/mctruth.npz")
phif223 = amp['phif223'][0:size1,0:2]
phif222 = amp['phif222'][0:size1,0:2]
phif221 = amp['phif221'][0:size1,0:2]
phif201 = amp['phif201'][0:size1,0:2]
data_phif2 = np.asarray([phif201,phif221,phif222,phif223])

phif001 = amp['phif001'][0:size1,0:2]
phif021 = amp['phif021'][0:size1,0:2]
data_phif0 = np.asarray([phif001,phif021])

mom = onp.load("data/mcmom.npz")
Kp = mom['Kp'][0:size1,:]
Km = mom['Km'][0:size1,:]
Pip = mom['Pip'][0:size1,:]
Pim = mom['Pim'][0:size1,:]

data_f = Pip + Pim
data_phi = Kp + Km

data_phi = invm(data_phi)
data_f = invm(data_f)


# MC truth 数据
size2 = 400000
amp = onp.load("data/mctruth.npz")
a = phif223 = amp['phif223']
phif223 = amp['phif223'][0:size2,0:2]
phif222 = amp['phif222'][0:size2,0:2]
phif221 = amp['phif221'][0:size2,0:2]
phif201 = amp['phif201'][0:size2,0:2]
mc_phif2 = np.asarray([phif201,phif221,phif222,phif223])

phif001 = amp['phif001'][0:size2,0:2]
phif021 = amp['phif021'][0:size2,0:2]
mc_phif0 = np.asarray([phif001,phif021])

mom = onp.load("data/mcmom.npz")
Kp = mom['Kp'][0:size2,:]
Km = mom['Km'][0:size2,:]
Pip = mom['Pip'][0:size2,:]
Pim = mom['Pim'][0:size2,:]

mc_f = Pip + Pim
mc_phi = Kp + Km

mc_phi = invm(mc_phi)
mc_f = invm(mc_f)


def weight(_phim,_phiw,_f0m,_f0w,_const1,_theta,_rho):
    const = np.asarray([[_const1],[1.]])
    rho = np.asarray([_rho])
    theta = np.asarray([_theta])
    phim = np.asarray([_phim])
    phiw = np.asarray([_phiw])
    f0m = np.asarray([_f0m])
    f0w = np.asarray([_f0w])
    d_phif0 = MOD(phim,phiw,f0m,f0w,const,theta,rho,data_phif0,data_phi,data_f)
    print(d_phif0.shape)
    m_phif0 = MOD(phim,phiw,f0m,f0w,const,theta,rho,mc_phif0,mc_phi,mc_f)
    print(m_phif0.shape)
    d_tmp = alladd(d_phif0)
    print(d_tmp.shape)
    m_tmp = np.average(alladd(m_phif0))
    print("weight")
    return d_tmp/m_tmp
    # return d_tmp

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
    return -np.sum(np.log(d_tmp)*wt) + size1*np.log(m_tmp)
    # return -np.sum(np.log(d_tmp)*wt)

phim = 1.02
phiw = 0.01
f0m = 0.5
f0w = 0.3
f2m = 1.
f2w = 1.
const1 = 5.5 # 该参数拟合的好
const2 = 1
rho = 1.
theta = 0.5

import matplotlib.pyplot as plt
wt = weight(phim,phiw,f0m,f0w,const1,theta,rho)
print(wt)
likelihood(phim,phiw,f0m,f0w,const1,theta,rho)
m=(0,1,2,3,4,5,6)
# m = 2
grad_likelihood = jit(grad(likelihood,argnums=m))
jit_likelihood = jit(likelihood)

# cc = np.arange(1000) / 1000
# kk = []
# for y in cc:
#     # kk.append(jit_likelihood(phim,phiw,y,f0w,const1,theta,rho))
#     # kk.append(jit_likelihood(phim,phiw,f0m,y,const1,theta,rho))
#     # kk.append(jit_likelihood(phim,phiw,f0m,f0w,y,theta,rho))
#     # kk.append(grad_likelihood(phim,phiw,y,f0w,const1,theta,rho))
#     # kk.append(grad_likelihood(phim,phiw,f0m,y,const1,theta,rho))
#     # kk.append(grad_likelihood(phim,phiw,f0m,f0w,y,theta,rho))
# plt.plot(cc, kk)
# plt.savefig("theta.png")

# print(iminuit.describe(likelihood))

par=('phim','phiw','f0m','f0w','const1','rho','theta')
print("begin")
m = iminuit.Minuit(jit_likelihood,
                   forced_parameters=par,
                   phim=1.02,phiw=0.01,f0m=0.4,f0w=0.2,
                   const1=5.,rho=1.,theta=0.5,
                   fix_phim=True,fix_phiw=True,fix_rho=True,
                   fix_theta=True,
                #    fix_const1=True,
             grad=grad_likelihood,
             error_phim=0.001,error_phiw=0.001, 
             error_f0m=0.1, error_f0w=0.1, error_const1=0.1, 
             error_rho=0.001, error_theta=0.001, errordef=0.5)
# m = iminuit.Minuit(likelihood,_phim=1.02,_phiw=0.01,_f0m=0.1,
#                     _f0w=0.2,_const1=5.5,_rho=1.,_theta=0.5,
#                    fix__phim=True,fix__phiw=True,fix__rho=True,
#                    fix__theta=True,
#                 #    fix_const1=True,
#                 grad=grad_likelihood,
#                 error__phim=0.001,error__phiw=0.001, 
#                 error__f0m=0.1, error__f0w=0.1, error__const1=0.1, 
#                 error__rho=0.001,error__theta=0.001,errordef=0.5)
print(m.get_param_states())
print(m.migrad())