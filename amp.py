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

config.update("jax_enable_x64", True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# # 计算 Pc + Pb 的不变质量
# def invm_plus(Pb,Pc):
#     Pbc = Pb + Pc
#     _Pbc = Pbc * np.array([-1,-1,-1,1])
#     return np.sum(Pbc * _Pbc,axis=1)

# # 计算 Pbc 的不变质量
# def invm(Pbc):
#     _Pbc = Pbc * np.array([-1,-1,-1,1])
#     return np.sum(Pbc * _Pbc,axis=1)

# # 对bw_取绝对值
# def _abs(bw_):
#     conjbw = np.conj(bw_)
#     return np.real(bw_*conjbw)

# # briet-w 公式
# def BW(m_,w_,Sbc):
#     gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
#     k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
#     return k/(m_*m_ - Sbc - m_*w_*1j)

# # \rho * e^{\theta}
# def phase(theta, rho):
# #     return rho * np.exp(theta*1j)
#     return rho * (np.cos(theta)+1j*np.sin(theta))

# 准备数据

amp = onp.load("data/data_amp.npz")
mom = onp.load("data/data_mom.npz")
phif = onp.load("data/data_phif.npz")

phif243 = phif['phif243'][:,0:2]
phif223 = amp['phif223'][:,0:2]
phif222 = amp['phif222'][:,0:2]
phif221 = amp['phif221'][:,0:2]
data_phif2 = np.asarray([phif221,phif222,phif223,phif243])
phif001 = amp['phif001'][:,0:2]
phif021 = amp['phif021'][:,0:2]
data_phif0 = np.asarray([phif001,phif021])
Kp = mom['Kp']
Km = mom['Km']
Pip = mom['Pip']
Pim = mom['Pim']


mc_amp = onp.load("data/mc_amp.npz")
mc_mom = onp.load("data/mc_mom.npz")
mc_phif = onp.load("data/mc_phif.npz")

mc_phif243 = mc_phif['phif243'][:,0:2]
mc_phif223 = mc_amp['phif223'][:,0:2]
mc_phif222 = mc_amp['phif222'][:,0:2]
mc_phif221 = mc_amp['phif221'][:,0:2]
mc_phif2 = np.asarray([mc_phif221,mc_phif222,mc_phif223,mc_phif243])
mc_phif001 = mc_amp['phif001'][:,0:2]
mc_phif021 = mc_amp['phif021'][:,0:2]
mc_phif0 = np.asarray([mc_phif001,mc_phif021])
mc_Kp = mc_mom['Kp']
mc_Km = mc_mom['Km']
mc_Pip = mc_mom['Pip']
mc_Pim = mc_mom['Pim']

data_f = Pip + Pim
data_phi = Kp + Km

mc_f = mc_Pip + mc_Pim
mc_phi = mc_Kp + mc_Km

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
    print(sum.shape)
    return np.sum(dplex.dabs(sum),axis=1)

def likelyhood(phim,phiw,f0m,f0w,f2m,f2w,const1,const2,rho,theta,_data_phif0,_data_phif2,_mc_phif0,_mc_phif2,_data_phi,_data_f,_mc_phi,_mc_f):
    d_phif0 = MOD(phim,phiw,f0m,f0w,const1,theta,rho,_data_phif0,_data_phi,_data_f)
    d_phif2 = MOD(phim,phiw,f2m,f2w,const2,theta,rho,_data_phif2,_data_phi,_data_f)
    m_phif0 = MOD(phim,phiw,f0m,f0w,const1,theta,rho,_mc_phif0,_mc_phi,_mc_f)
    m_phif2 = MOD(phim,phiw,f2m,f2w,const2,theta,rho,_mc_phif2,_mc_phi,_mc_f)
    d_tmp = np.sum(dplex.dabs(d_phif0+d_phif2),axis=1)
    print(d_tmp.shape)
    d_tmp = -np.sum(np.log(d_tmp))
    print(d_tmp)
    return 0

phim = np.array([2.,1.,1.,1.])
phiw = np.array([1.,2.,1.,1.])
f0m = np.array([1.,1.,1.,3.])
f0w = np.array([1.,1.,1.,1.])
f2m = np.array([1.,1.,1.,3.])
f2w = np.array([1.,1.,1.,1.])
const1 = np.array([[2.,1.,1.,1.],[1.,1.,1.,1.]])
const2 = np.array([[2.,1.,1.,1.],[1.,1.,1.,1.],[2.,1.,1.,1.],[1.,1.,1.,1.]])
rho = np.array([1.,1.,1.,1.])
theta = np.array([1.,1.,1.,1.])

data_phi = invm(data_phi)
data_f = invm(data_f)
mc_phi = invm(mc_phi)
mc_f = invm(mc_f)

print(data_phif2.shape)
likelyhood(phim,phiw,f0m,f0w,f2m,f2w,const1,const2,rho,theta,data_phif0,data_phif2,mc_phif0,mc_phif2,data_phi,data_f,mc_phi,mc_f)
