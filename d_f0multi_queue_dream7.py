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

    def mcnpz(self,begin,end,num):
        amp = onp.load("data/mctruth.npz")
        # phif223 = amp['phif223'][begin:end,0:2]
        # phif222 = amp['phif222'][begin:end,0:2]
        # phif221 = amp['phif221'][begin:end,0:2]
        # phif201 = amp['phif201'][begin:end,0:2]
        # data_phif2 = np.asarray([phif201,phif221,phif222,phif223])
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
        data_phi = self.invm(data_phi)
        data_f = self.invm(data_f)
        data_phif0 = onp.array_split(data_phif0,num,axis=1)
        data_phi = onp.array_split(data_phi,num,axis=0)
        data_f = onp.array_split(data_f,num,axis=0)
        return data_phif0,data_phi,data_f

    def _BW(self,m_,w_,Sbc):
        gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
        k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
        l = Sbc.shape[0]
        temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
        return dplex.ddivide(k, temp)

    def _phase(self,theta):
        return dplex.dconstruct(np.cos(theta), np.sin(theta))

    def BW(self,phim,phiw,fm,fw,phi,f):
        a = self._BW(phim,phiw,phi)
        b = np.moveaxis(vmap(partial(self._BW,Sbc=f))(fm,fw),1,0)
        result = dplex.deinsum('j,ij->ij',a,b)
        return result

    def phase(self,theta):
        result = vmap(self._phase)(theta)
        return result

    def MOD(self,fm,fw,const,theta,phif,phi,f):
        phim = np.asarray([1.02])
        phiw = np.asarray([0.01])
        ph = np.moveaxis(self.phase(theta), 1, 0)
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

    def weight(self,args,data_phif0,data_phi,data_f,mc_phif0,mc_phi,mc_f):
        array_args = args.reshape(4,-1) 
        f0m,f0w,const,theta = np.split(array_args,4,axis=0)
        f0m = np.squeeze(f0m,axis=0)
        f0w = np.squeeze(f0w,axis=0)
        theta = np.squeeze(theta,axis=0)
        const = np.append(np.squeeze(const,axis=0),np.ones(const.shape)).reshape(2,-1)
        d_phif0 = self.MOD(f0m,f0w,const,theta,data_phif0,data_phi,data_f)
        m_phif0 = self.MOD(f0m,f0w,const,theta,mc_phif0,mc_phi,mc_f)
        d_tmp = np.sum(dplex.dabs(d_phif0),axis=1)
        m_tmp = np.average(np.sum(dplex.dabs(m_phif0),axis=1))
        return d_tmp/m_tmp

    def mods(self,args,wt,data_phif0,data_phi,data_f,mc_phif0,mc_phi,mc_f):
        args = np.array(args)
        array_args = args.reshape(4,-1) 
        f0m,f0w,const,theta = np.split(array_args,4,axis=0)
        f0m = np.squeeze(f0m,axis=0)
        f0w = np.squeeze(f0w,axis=0)
        theta = np.squeeze(theta,axis=0)
        const = np.append(np.squeeze(const,axis=0),np.ones(const.shape)).reshape(2,-1)
        d_phif0 = self.MOD(f0m,f0w,const,theta,data_phif0,data_phi,data_f)
        m_phif0 = self.MOD(f0m,f0w,const,theta,mc_phif0,mc_phi,mc_f)
        d_tmp = np.sum(dplex.dabs(d_phif0),axis=1)
        m_tmp = np.average(np.sum(dplex.dabs(m_phif0),axis=1))
        wt_sum = np.sum(wt)
        return -np.sum(wt*(np.log(d_tmp) - np.log(m_tmp))) / np.log(wt_sum)

    def part11(self,args,wt,data_phif0,data_phi,data_f,mc_phif0,mc_phi,mc_f):
        args = np.array(args)
        array_args = args.reshape(4,-1) 
        f0m,f0w,const,theta = np.split(array_args,4,axis=0)
        f0m = np.squeeze(f0m,axis=0)
        f0w = np.squeeze(f0w,axis=0)
        theta = np.squeeze(theta,axis=0)
        const = np.append(np.squeeze(const,axis=0),np.ones(const.shape)).reshape(2,-1)
        d_phif0 = self.MOD(f0m,f0w,const,theta,data_phif0,data_phi,data_f)
        d_tmp = np.sum(dplex.dabs(d_phif0),axis=1)
        return np.sum(wt*np.log(d_tmp))

    def part22(self,args,wt,data_phif0,data_phi,data_f,mc_phif0,mc_phi,mc_f):
        args = np.array(args)
        array_args = args.reshape(4,-1) 
        f0m,f0w,const,theta = np.split(array_args,4,axis=0)
        f0m = np.squeeze(f0m,axis=0)
        f0w = np.squeeze(f0w,axis=0)
        theta = np.squeeze(theta,axis=0)
        const = np.append(np.squeeze(const,axis=0),np.ones(const.shape)).reshape(2,-1)
        m_phif0 = self.MOD(f0m,f0w,const,theta,mc_phif0,mc_phi,mc_f)
        m_tmp = np.average(np.sum(dplex.dabs(m_phif0),axis=1))
        return np.sum(wt*np.log(m_tmp))

    def Weight(self,ags):
        return self.weight(ags,self.data_phif0,self.data_phi,self.data_f,self.mc_phif0,self.mc_phi,self.mc_f)

    def likelihood(self,ags):
        return jit(self.mods)(ags,self.wt,self.data_phif0,self.data_phi,self.data_f,self.mc_phif0,self.mc_phi,self.mc_f)

    def part1(self,ags):
        return jit(self.part11)(ags,self.wt,self.data_phif0,self.data_phi,self.data_f,self.mc_phif0,self.mc_phi,self.mc_f)

    def part2(self,ags):
        return jit(self.part22)(ags,self.wt,self.data_phif0,self.data_phi,self.data_f,self.mc_phif0,self.mc_phi,self.mc_f)

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.card

        print(self.id, 'Variable initialization is finished')

        event_num = 100
        all_data_phif0,all_data_phi,all_data_f = self.mcnpz(0,500000,event_num)
        all_mc_phif0,all_mc_phi,all_mc_f = self.mcnpz(500000,700000,1)
        self.mc_phif0 = np.squeeze(all_mc_phif0[0],axis=None)
        self.mc_phi = np.squeeze(all_mc_phi[0],axis=None)
        self.mc_f = np.squeeze(all_mc_f[0],axis=None)
        
        t_ = 278
        m = onp.random.rand(t_)
        w = onp.random.rand(t_)
        c = onp.random.rand(t_)
        t = onp.random.rand(t_)
        wtarg = np.append(np.append(np.append(m,w),c),t)

        i=0
        self.data_phif0 = np.squeeze(all_data_phif0[i],axis=None)
        self.data_phi = np.squeeze(all_data_phi[i],axis=None)
        self.data_f = np.squeeze(all_data_f[i],axis=None)

        self.wt = self.Weight(wtarg)        

        if self.part == 1:
            self.res = jit(grad(self.part1))
            self.qout.put(self.wt)
        else:
            self.res = jit(grad(self.part2))
        
        while(True):
            var = self.qdata.get()
            print(var.shape)
            if var.shape[0] == t_*4:

                start = time.time()
                result = self.res(var)
                # print('process ID -',self.id,result)
                # self.qout.put(result)
                print('process ID -', self.id+' part'+str(self.part), '(time):', float(time.time()-start))
                self.qout.put(result)

            else:
                self.qout.put(0)
                break

# # GPU1-part1 GPU2-part2
# def gradAll(grad1, grad2):
#     return -np.sum(wt*(grad1 - grad2.reshape(1,8)), 0) / np.log(np.sum(wt))

# GPU1-part2 GPU2-part1
def gradAll(grad1, grad2):
    return (grad1 - grad2) / np.log(np.sum(wt))

print('***************************************************************************')
qdata_p = Queue()
qdata_q = Queue()
qdata_p_out = Queue()
qdata_q_out = Queue()
p = Card("0", 2, qdata_p, qdata_p_out)
q = Card("1", 1, qdata_q, qdata_q_out)
p.start()
q.start()

# wt = onp.random.rand(5000,8)
wt = qdata_q_out.get()
print(wt.size)
jitGradAll = jit(gradAll)

t_ = 278
m = onp.random.rand(t_)
w = onp.random.rand(t_)
c = onp.random.rand(t_)
t = onp.random.rand(t_)
wtarg = np.append(np.append(np.append(m,w),c),t)

etime = []
for i in range(10):
    qdata_p.put(wtarg)
    qdata_q.put(wtarg)
    start = time.time()
    t = time.time()
    s = time.time()
    a = qdata_p_out.get()
    print('\tp cal time:', float(time.time() - s))
    s = time.time()
    b = qdata_q_out.get()
    print('\tq cal time:', float(time.time() - s))
    print('\ttotal cal time:', float(time.time() - t))
    s = time.time()
    res = jitGradAll(a,b)
    print('\tintegration time:', float(time.time() - s))
    etime_ = float(time.time() - start)
    print('exec time:', float(time.time() - start))
    etime.append(etime_)
    print('grad:', res.shape)

var = onp.asarray([0.,])
qdata_p.put(var)
qdata_q.put(var)

print('average cal time:', onp.average(etime[1:]))

p.join()
q.join()



