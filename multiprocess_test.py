import multiprocessing as mp
from multiprocessing import Process, Queue
import os
import jax.numpy as np
import numpy as onp
from jax import vmap
from functools import partial
import time
from jax import jit
from jax import grad
import dplex
from jax.config import config

config.update("jax_enable_x64", True)

# def invm_plus(Pb,Pc):
#     Pbc = Pb + Pc
#     _Pbc = Pbc * onp.array([-1,-1,-1,1])
#     return onp.sum(Pbc * _Pbc,axis=1)

# def invm(Pbc):
#     _Pbc = Pbc * onp.array([-1,-1,-1,1])
#     return onp.sum(Pbc * _Pbc,axis=1)

# def _abs(bw_):
#     conjbw = onp.conj(bw_)
#     return onp.real(bw_*conjbw)

# size = 800000
# Kp = onp.random.sample(size*4).reshape(size,4)
# Km = onp.random.sample(size*4).reshape(size,4)
# Pip = onp.random.sample(size*4).reshape(size,4)
# Pim = onp.random.sample(size*4).reshape(size,4)
# phif001 = onp.random.sample(size*2).reshape(size,2)
# phif021 = onp.random.sample(size*2).reshape(size,2)

# phi = invm_plus(Kp,Km)
# f0 = invm_plus(Pip,Pim)
# phif0 = onp.array([phif001,phif021])

# phim = onp.array([2.,1.,1.,1.])
# phiw = onp.array([1.,2.,1.,1.])
# f0m = onp.array([1.,1.,1.,3.])
# f0w = onp.array([1.,1.,1.,1.])
# const = onp.array([[2.,1.,1.,1.],[1.,1.,1.,1.]])
# rho = onp.array([1.,1.,2.,1.])
# theta = onp.array([1.,1.,1.,1.])

class Card(Process):
    def __init__(self, card, qdata, qdata_out, gradnums):
        Process.__init__(self)
        self.card = card
        self.qdata = qdata
        self.qout = qdata_out
        os.environ["CUDA_VISIBLE_DEVICES"] = self.card
        print("process " + str(os.getpid()) + ": " + "GPU" + card)
        self.id = "GPU" + self.card
        self.res = jit(grad(self.test_pw, argnums=gradnums))
        print(self.id, 'Initialization finished:', os.environ["CUDA_VISIBLE_DEVICES"])
    
    def invm_plus(self,Pb,Pc):
        
        Pbc = Pb + Pc
        _Pbc = Pbc * onp.array([-1,-1,-1,1])
        return onp.sum(Pbc * _Pbc,axis=1)

    def invm(self,Pbc):
        
        _Pbc = Pbc * onp.array([-1,-1,-1,1])
        return onp.sum(Pbc * _Pbc,axis=1)

    def _abs(self,bw_):
        conjbw = onp.conj(bw_)
        return onp.real(bw_*conjbw)

    def BW(self,m_,w_,Sbc):
        
        gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
        k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
        l = Sbc.shape[0]
        temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
        return dplex.ddivide(k, temp)

    def phase(self,theta, rho):
        
        return dplex.dconstruct(rho * np.cos(theta), rho * np.sin(theta))

    def BW_f0(self,phim,phiw,f0m,f0w,phi,f0):
        a = np.moveaxis(vmap(partial(self.BW,Sbc=phi))(phim,phiw),1,0)
        b = np.moveaxis(vmap(partial(self.BW,Sbc=f0))(f0m,f0w),1,0)
        result = dplex.deinsum('ij,ij->ij',a,b)
        
        return result

    def phase_f0(self,theta_,rho_):
        result = vmap(self.phase)(theta_,rho_)
        
        return result

    def test_pw(self,phim,phiw,f0m,f0w,const,rho,theta,phif0,phi,f0):
        print(self.id + ': test_pw is called')
        ph = np.moveaxis(self.phase_f0(theta,rho), 1, 0)
        bw = self.BW_f0(phim,phiw,f0m,f0w,phi,f0)
        _phif0 = dplex.dtomine(np.einsum('ijk,il->ljk',phif0,const))
        _phif0 = dplex.deinsum('ijk,i->ijk',_phif0,ph)
        _phif0 = dplex.deinsum('ijk,ij->jk',_phif0,bw)
        _phif0 = np.real(np.sum(dplex.dabs(_phif0),axis=1))
        
        return -np.sum(np.log(_phif0))

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.card
        size = 800000
        Kp = onp.random.sample(size*4).reshape(size,4)
        Km = onp.random.sample(size*4).reshape(size,4)
        Pip = onp.random.sample(size*4).reshape(size,4)
        Pim = onp.random.sample(size*4).reshape(size,4)
        phif001 = onp.random.sample(size*2).reshape(size,2)
        phif021 = onp.random.sample(size*2).reshape(size,2)

        phi = self.invm_plus(Kp,Km)
        f0 = self.invm_plus(Pip,Pim)
        phif0 = onp.array([phif001,phif021])

        phim = onp.array([2.,1.,1.,1.])
        phiw = onp.array([1.,2.,1.,1.])
        f0m = onp.array([1.,1.,1.,3.])
        f0w = onp.array([1.,1.,1.,1.])
        const = onp.array([[2.,1.,1.,1.],[1.,1.,1.,1.]])
        rho = onp.array([1.,1.,2.,1.])
        theta = onp.array([1.,1.,1.,1.])
        
        print(self.id, 'Variable initialization is finished')        
        
        while(True):
            var = self.qdata.get()
            #start = time.time()
            #print(self.id, self.res(var[0],var[1],var[2],var[3],var[4],var[5],var[6],phif0,phi,f0))
            self.qout.put(self.res(var[0],var[1],var[2],var[3],var[4],var[5],var[6],phif0,phi,f0))
            # print('exec time on ' + self.id + ':', float(time.time()-start))


print('***************************************************************************')
qdata_p = Queue()
qdata_q = Queue()
qdata_p_out = Queue()
qdata_q_out = Queue()
p = Card("0", qdata_p, qdata_p_out, (0, 1, 2, 3, 4, 5, 6))
q = Card("1", qdata_q, qdata_q_out, (0, 1, 2, 3, 4, 5, 6))
p.start()
q.start()

phim = onp.array([2.,1.,1.,1.])
phiw = onp.array([1.,2.,1.,1.])
f0m = onp.array([1.,1.,1.,3.])
f0w = onp.array([1.,1.,1.,1.])
const = onp.array([[2.,1.,1.,1.],[1.,1.,1.,1.]])
rho = onp.array([1.,1.,2.,1.])
theta = onp.array([1.,1.,1.,1.])

var = [phim,phiw,f0m,f0w,const,rho,theta]

for i in range(10):
    qdata_p.put(var)
    qdata_q.put(var)
    start = time.time()
    print('GPU0:', qdata_p_out.get())
    print('exec time of GPU1:', float(time.time() - start))
    start = time.time()
    print('GPU1:', qdata_q_out.get())
    print('exec time of GPU2:', float(time.time() - start))

p.join()
q.join()



