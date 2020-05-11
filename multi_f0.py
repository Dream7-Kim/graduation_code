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
import time
config.update("jax_enable_x64", True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def invm_plus(Pb,Pc):
    Pbc = Pb + Pc
    _Pbc = Pbc * np.array([-1,-1,-1,1])
    return np.sum(Pbc * _Pbc,axis=1)

def invm(Pbc):
    _Pbc = Pbc * np.array([-1,-1,-1,1])
    return np.sum(Pbc * _Pbc,axis=1)

def mcnpz(begin,end,num):
    amp = onp.load("data/mctruth.npz")
    # phif223 = amp['phif223'][begin:end,0:2]
    # phif222 = amp['phif222'][begin:end,0:2]
    # phif221 = amp['phif221'][begin:end,0:2]
    # phif201 = amp['phif201'][begin:end,0:2]
    # data_phif2 = onp.asarray([phif201,phif221,phif222,phif223])
    phif001 = amp['phif001'][begin:end,0:2]
    phif021 = amp['phif021'][begin:end,0:2]
    data_phif0 = onp.asarray([phif001,phif021])
    mom = onp.load("data/mcmom.npz")
    Kp = mom['Kp'][begin:end,:]
    Km = mom['Km'][begin:end,:]
    Pip = mom['Pip'][begin:end,:]
    Pim = mom['Pim'][begin:end,:]
    data_f = Pip + Pim
    data_phi = Kp + Km
    data_phi = invm(data_phi)
    data_f = invm(data_f)
    data_phif0 = onp.array_split(data_phif0,num,axis=1)
    data_phi = onp.array_split(data_phi,num,axis=0)
    data_f = onp.array_split(data_f,num,axis=0)
    return data_phif0,data_phi,data_f

def _BW(m_,w_,Sbc):
    gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
    k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
    l = Sbc.shape[0]
    temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
    return dplex.ddivide(k, temp)

def _phase(theta):
    return dplex.dconstruct(np.cos(theta), np.sin(theta))

def BW(phim,phiw,fm,fw,phi,f):
    a = _BW(phim,phiw,phi)
    b = np.moveaxis(vmap(partial(_BW,Sbc=f))(fm,fw),1,0)
    result = dplex.deinsum('j,ij->ij',a,b)
    return result

def phase(theta):
    result = vmap(_phase)(theta)
    return result

def MOD(fm,fw,const,theta,phif,phi,f):
    phim = np.asarray([1.02])
    phiw = np.asarray([0.01])
    ph = np.moveaxis(phase(theta), 1, 0)
    bw = BW(phim,phiw,fm,fw,phi,f)
    _phif = dplex.dtomine(np.einsum('ijk,il->ljk',phif,const))
    _phif = dplex.deinsum('ijk,i->ijk',_phif,ph)
    _phif = dplex.deinsum('ijk,ij->jk',_phif,bw)
    return _phif

def alladd(*mods):
    l = (mods[0].shape)[1]
    sum = np.zeros(l*2*2).reshape(2,l,2)
    for num in mods:
        sum += num
    return np.sum(dplex.dabs(sum),axis=1)


def weight(args,data_phif0,data_phi,data_f,mc_phif0,mc_phi,mc_f):
    array_args = args.reshape(4,-1) 
    f0m,f0w,const,theta = np.split(array_args,4,axis=0)
    f0m = np.squeeze(f0m,axis=0)
    f0w = np.squeeze(f0w,axis=0)
    theta = np.squeeze(theta,axis=0)
    const = np.append(np.squeeze(const,axis=0),np.ones(const.shape)).reshape(2,-1)
    d_phif0 = MOD(f0m,f0w,const,theta,data_phif0,data_phi,data_f)
    m_phif0 = MOD(f0m,f0w,const,theta,mc_phif0,mc_phi,mc_f)
    d_tmp = np.sum(dplex.dabs(d_phif0),axis=1)
    m_tmp = np.average(np.sum(dplex.dabs(m_phif0),axis=1))
    return d_tmp/m_tmp

def mods(args,wt,data_phif0,data_phi,data_f,mc_phif0,mc_phi,mc_f):
    args = np.array(args)
    array_args = args.reshape(4,-1) 
    f0m,f0w,const,theta = np.split(array_args,4,axis=0)
    f0m = np.squeeze(f0m,axis=0)
    f0w = np.squeeze(f0w,axis=0)
    theta = np.squeeze(theta,axis=0)
    const = np.append(np.squeeze(const,axis=0),np.ones(const.shape)).reshape(2,-1)
    d_phif0 = MOD(f0m,f0w,const,theta,data_phif0,data_phi,data_f)
    m_phif0 = MOD(f0m,f0w,const,theta,mc_phif0,mc_phi,mc_f)
    d_tmp = np.sum(dplex.dabs(d_phif0),axis=1)
    m_tmp = np.average(np.sum(dplex.dabs(m_phif0),axis=1))
    print('d_tmp',d_tmp.shape)
    print('m_tmp',m_tmp.shape)
    wt_sum = np.sum(wt)
    print('wt', wt.size)
    return -np.sum(wt*(np.log(d_tmp) - np.log(m_tmp))) / np.log(wt_sum)

def Weight(ags):
    return weight(ags,data_phif0,data_phi,data_f,mc_phif0,mc_phi,mc_f)

def likelihood(ags):
    return jit(mods)(ags,wt,data_phif0,data_phi,data_f,mc_phif0,mc_phi,mc_f)

num_ran = 100
ranm = (onp.random.randint(100,size=num_ran) - 50) / 1250
ranw = (onp.random.randint(100,size=num_ran) - 50) / 1250
ranc = (onp.random.randint(100,size=num_ran) - 50) / 250
i = 0
j = 0
event_num = 100
argnum = event_num * 2 # 乘上的是f0的数量
vf0m = onp.zeros(argnum)
vf0w = onp.zeros(argnum)
vconst = onp.zeros(argnum)
vtheta = onp.zeros(argnum)
ef0m = onp.zeros(argnum)
ef0w = onp.zeros(argnum)
econst = onp.zeros(argnum)
etheta = onp.zeros(argnum)

all_data_phif0,all_data_phi,all_data_f = mcnpz(0,500000,event_num)
all_mc_phif0,all_mc_phi,all_mc_f = mcnpz(500000,700000,1)
mc_phif0 = np.squeeze(all_mc_phif0[0],axis=None)
mc_phi = np.squeeze(all_mc_phi[0],axis=None)
mc_f = np.squeeze(all_mc_f[0],axis=None)

# m w c t 分别是质量 宽度 常数项 角度
m = np.array([0.5,0.4])
w = np.array([0.24,0.35])
c = np.array([5.5,6.7])
t = np.array([1.3,0])
wtarg = np.append(np.append(np.append(m,w),c),t)

# print(wtarg.shape)

for rm,rw,rc in zip(ranm,ranw,ranc):
    data_phif0 = np.squeeze(all_data_phif0[i],axis=None)
    data_phi = np.squeeze(all_data_phi[i],axis=None)
    data_f = np.squeeze(all_data_f[i],axis=None)
    i+=1
    wt = Weight(wtarg)
    print(wt.shape)
    m = np.array([0.5,0.4])
    w = np.array([0.3,0.2])
    c = np.array([5.5,6.7])
    t = np.array([1.3,0])
    args_list = onp.append(np.append(np.append(m,w),c),t)
    list_error = tuple(onp.repeat(0.1, args_list.shape, axis=None))
    my_likelihood = likelihood
    grad_likelihood = jit(grad(likelihood))
    # s = time.time()
    m = iminuit.Minuit.from_array_func(my_likelihood, args_list, list_error, fix_x7=True, errordef=0.5, grad=grad_likelihood)
    # m.migrad()
    print(m.migrad())
    # e = time.time()
    # print("time",e-s)
    # print(m.migrad_ok())
    fvalue = m.values
    ferror = m.errors
    vf0m[j] = fvalue['x0']
    vf0m[j+1] = fvalue['x1']
    vf0w[j] = fvalue['x2']
    vf0w[j+1] = fvalue['x3']
    vconst[j] = fvalue['x4']
    vconst[j+1] = fvalue['x5']
    vtheta[j] = fvalue['x6']
    vtheta[j+1] = fvalue['x7']
    ef0m[j] = fvalue['x0']
    ef0m[j+1] = fvalue['x1']
    ef0w[j] = ferror['x2']
    ef0w[j+1] = ferror['x3']
    econst[j] = ferror['x4']
    econst[j+1] = ferror['x5']
    etheta[j] = ferror['x6']
    etheta[j+1] = ferror['x7']
    j+=2
onp.savez("fit_result",f0m=vf0m,f0w=vf0w,const=vconst,theta=vtheta,ef0m=ef0m,ef0w=ef0w,econst=econst,etheta=etheta)