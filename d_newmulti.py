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
from jax import jacfwd
import dplex
from jax.config import config

config.update("jax_enable_x64", True)

class Card(Process):
    def __init__(self, card, part, qdata, qdata_out):
        Process.__init__(self)
        self.card = card
        self.part = part
        self.qdata = qdata
        self.qout = qdata_out
        os.environ["CUDA_VISIBLE_DEVICES"] = self.card
        print("process " + str(os.getpid()) + ": " + "GPU" + card)
        self.id = "GPU" + self.card

        self.res = np.add

        print(self.id, 'Initialization finished:', os.environ["CUDA_VISIBLE_DEVICES"])

    def invm_plus(self, Pb,Pc):
        Pbc = Pb + Pc
        _Pbc = Pbc * np.array([-1,-1,-1,1])
        return np.sum(Pbc * _Pbc,axis=1)

    def invm(self, Pbc):
        _Pbc = Pbc * np.array([-1,-1,-1,1])
        return np.sum(Pbc * _Pbc,axis=1)

    def mcnpz(self,begin,end):
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
        data_phi = self.invm(data_phi)
        data_f = self.invm(data_f)
        return data_phif0,data_phi,data_f

    def _BW(self,m_,w_,Sbc):
        gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
        k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
        l = Sbc.shape[0]
        temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
        return dplex.ddivide(k, temp)

    def _phase(self,_theta, _rho):
        return dplex.dconstruct(_rho * np.cos(_theta), _rho * np.sin(_theta))

    def BW(self,phim,phiw,fm,fw,phi,f):
        a = np.moveaxis(vmap(partial(self._BW,Sbc=phi))(phim,phiw),1,0)
        b = np.moveaxis(vmap(partial(self._BW,Sbc=f))(fm,fw),1,0)
        result = dplex.deinsum('ij,ij->ij',a,b)
        return result

    def phase(self,_theta,_rho):
        result = vmap(self._phase)(_theta,_rho)
        return result

    def MOD(self,fm,fw,_const,phif,phi,f):
        phim = np.asarray([1.02])
        phiw = np.asarray([0.01])
        theta = np.asarray([1.])
        rho = np.asarray([1.])
        const = np.append(_const,1.).reshape(2,1)
        ph = np.moveaxis(self.phase(theta,rho), 1, 0)
        bw = self.BW(phim,phiw,fm,fw,phi,f)
        _phif = dplex.dtomine(np.einsum('ijk,il->ljk',phif,const))
        _phif = dplex.deinsum('ijk,i->ijk',_phif,ph)
        _phif = dplex.deinsum('ijk,ij->jk',_phif,bw)
        return _phif

    def alladd(self, *mods):
        l = (mods[0].shape)[1]
        sum = onp.zeros(l*2*2).reshape(2,l,2)
        for num in mods:
            sum += num
        return np.sum(dplex.dabs(sum),axis=1)

    def weight(self, args):
        f0m,f0w,const = np.split(args,3)
        d_phif0 = self.MOD(f0m,f0w,const,self.data_phif0,self.data_phi,self.data_f)
        m_phif0 = self.MOD(f0m,f0w,const,self.mc_phif0,self.mc_phi,self.mc_f)
        d_tmp = self.alladd(d_phif0)
        m_tmp = np.average(self.alladd(m_phif0))
        #print("weight")
        return d_tmp/m_tmp

    def likelihood(self,args):
        f0m,f0w,const = np.split(args,3)
        d_phif0 = self.MOD(f0m,f0w,const,self.data_phif0,self.data_phi,self.data_f)
        m_phif0 = self.MOD(f0m,f0w,const,self.mc_phif0,self.mc_phi,self.mc_f)
        d_tmp = self.alladd(d_phif0)
        m_tmp = np.average(self.alladd(m_phif0))
        return -np.sum(self.wt*(np.log(d_tmp) - np.log(m_tmp)))

    def part1(self, args):
        f0m,f0w,const = np.split(args,3)
        d_phif0 = self.MOD(f0m,f0w,const,self.data_phif0,self.data_phi,self.data_f)
        d_tmp = self.alladd(d_phif0)
        return np.log(d_tmp)

    def part2(self, args):
        f0m,f0w,const = np.split(args,3)
        m_phif0 = self.MOD(f0m,f0w,const,self.mc_phif0,self.mc_phi,self.mc_f)
        m_tmp = np.average(self.alladd(m_phif0))
        return np.log(m_tmp)

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.card

        print(self.id, 'Variable initialization is finished')

        self.data_phif0,self.data_phi,self.data_f = self.mcnpz(0,1000000)
        self.mc_phif0,self.mc_phi,self.mc_f = self.mcnpz(0,1000000)

        wtarg = onp.asarray([0.5,0.4,5.5])

        self.wt = self.weight(wtarg)
        if self.part == 1:
            self.res = jit(jacfwd(self.part1))
        else:
            self.res = jit(jacfwd(self.part2))

        while(True):
            var = self.qdata.get()
            if var.shape[0] == 3:
                start = time.time()
                result = self.res(var)
                # self.qout.put(result)
                print('process ID -', self.id, ':', float(time.time()-start))
                self.qout.put(result)
            else:
                self.qout.put(0)
                break

def gradAll(grad1, grad2):
    # wtt = onp.random.rand(10000,3)
    return -np.sum(wtt*(grad1 - grad2), 0)

print('***************************************************************************')
wtt = onp.random.rand(713947,3)
jitGradAll = jit(gradAll)
qdata_p = Queue()
qdata_q = Queue()
qdata_p_out = Queue()
qdata_q_out = Queue()
p = Card("1", 1, qdata_p, qdata_p_out)
q = Card("0", 2, qdata_q, qdata_q_out)
p.start()
q.start()

var = onp.asarray([0.5,0.4,5.5])
# print(var.shape)

for i in range(10):
    qdata_p.put(var)
    qdata_q.put(var)
    start = time.time()
    t = time.time()
    s = time.time()
    a = qdata_p_out.get()
    print('p cal time:', float(time.time() - s))
    s = time.time()
    b = qdata_q_out.get()
    print('q cal time:', float(time.time() - s))
    print('total cal time:', float(time.time() - t))
    s = time.time()
    c = jitGradAll(a,b)
    print('integration time:', float(time.time() - s))
    print('exec time:', float(time.time() - start))
    print('grad:', c)

var = onp.asarray([0.,])
qdata_p.put(var)
qdata_q.put(var)

p.join()
q.join()



