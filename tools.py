import seaborn as sns
import matplotlib.pyplot as plt
import numpy as onp
import jax
import jax.numpy as np
import time
import scipy.optimize as opt
import ROOT as rt
import uproot_methods.classes.TLorentzVector
from jax import device_put


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
    n = end - begin
    return data_phif0,data_phi,data_f,n



def read(string):
    lines = open(string).readlines()
    row = int(len(lines) / 4)
    lists = []
    for line in lines:
        str = line.replace('[', '')
        str = str.replace(']', '')
        str = str.strip()
        tmp = float(str)
        lists.append(tmp)
    array = onp.array(lists).reshape(row, 4)
    return array


def readtxt():
    phif001 = read('data/phif001MC.txt')[0:20000, 0:2]
    phif021 = read('data/phif021MC.txt')[0:20000, 0:2]

    phif001MC = read('data/phif001MC.txt')[20000:100000, 0:2]
    phif021MC = read('data/phif021MC.txt')[20000:100000, 0:2]

    Kp = read('data/KpMC.txt')[0:20000, :]
    Km = read('data/KmMC.txt')[0:20000, :]
    Pip = read('data/PipMC.txt')[0:20000, :]
    Pim = read('data/PimMC.txt')[0:20000, :]

    KpMC = read('data/KpMC.txt')[20000:100000, :]
    KmMC = read('data/KmMC.txt')[20000:100000, :]
    PipMC = read('data/PipMC.txt')[20000:100000, :]
    PimMC = read('data/PimMC.txt')[20000:100000, :]
    return phif001, phif021, phif001MC, phif021MC, Kp, Km, Pip, Pim, KpMC, KmMC, PipMC, PimMC


def readroot(name):
    # rt.ROOT.EnableImplicitMT()
    f = rt.TFile(name)
    # tree = f.Get("tr")
    tree = f.tr
    # entries = tree.GetEntriesFast()
    mom = onp.asarray([[tree.Kp1X, tree.Kp1Y, tree.Kp1Z, tree.Kp1E,
                        tree.Km1X, tree.Km1Y, tree.Km1Z, tree.Km1E,
                        tree.Kp2X, tree.Kp2Y, tree.Kp2Z, tree.Kp2E,
                        tree.Km2X, tree.Km2Y, tree.Km2Z, tree.Km2E
                        ] for event in tree])
    fastor = onp.asarray([[tree.phif001X, tree.phif001Y, tree.phif021X, tree.phif021Y] for event in tree])
    phif001 = (fastor[0:120000,0:2])
    phif021 = (fastor[0:120000,2:4])
    phif001MC = (fastor[80000:400000,0:2])
    phif021MC = (fastor[80000:400000,2:4])
    Kp = (mom[0:120000,0:4])
    Km = (mom[0:120000,4:8])
    Pip = (mom[0:120000,8:12])
    Pim = (mom[0:120000,12:16])
    KpMC = (mom[80000:400000,0:4])
    KmMC = (mom[80000:400000,4:8])
    PipMC = (mom[80000:400000,8:12])
    PimMC = (mom[80000:400000,12:16])
    print("have get tensor")
    return phif001, phif021, phif001MC, phif021MC, Kp, Km, Pip, Pim, KpMC, KmMC, PipMC, PimMC


def array2lorentzvector(arr):
    px = arr[:, 0]
    py = arr[:, 1]
    pz = arr[:, 2]
    E = arr[:, 3]
    flat = uproot_methods.classes.TLorentzVector.TLorentzVectorArray(
        px, py, pz, E)
    return flat


# likelihood(var,const,phif001,phif021,phif001MC,phif021MC,phi,f0,phiMC,f0MC)


# cc = np.arange(1000) / 100
# kk = []
# for y in cc:
#     var = np.array([M1,W1,M2,W2,M3,W3,y])
#     kk.append(likelihood(var))
# plt.plot(cc,kk)
# plt.grid()
# plt.savefig('mutil_rho.png')


# grad = jax.jit(jax.grad(likelihood))
# x = np.array([1.04, 0.05, 1.3, 0.44, 1.6, 0.122, 1.7])
# res = opt.minimize(likelihood, x, method='BFGS',
#                    jac=grad, options={'disp': True})
# np.set_printoptions(precision=16)
# print('\n\nResult: ', res.x)

    # Kp1 = onp.asarray([[tree.Kp1X, tree.Kp1Y, tree.Kp1Z, tree.Kp1E]
    #                    for event in tree])
    # Km1 = onp.asarray([[tree.Km1X, tree.Km1Y, tree.Km1Z, tree.Km1E]
    #                    for event in tree])
    # Kp2 = onp.asarray([[tree.Kp2X, tree.Kp2Y, tree.Kp2Z, tree.Kp2E]
    #                    for event in tree])
    # Km2 = onp.asarray([[tree.Km2X, tree.Km2Y, tree.Km2Z, tree.Km2E]
    #                    for event in tree])

    # all_ = onp.asarray([[tree.Kp1X, tree.Kp1Y, tree.Kp1Z, tree.Kp1E,
    #                      tree.Km1X, tree.Km1Y, tree.Km1Z, tree.Km1E,
    #                      tree.Kp2X, tree.Kp2Y, tree.Kp2Z, tree.Kp2E,
    #                      tree.Km2X, tree.Km2Y, tree.Km2Z, tree.Km2E,
    #                      tree.phif001X, tree.phif001Y, tree.phif001Z, tree.phif001E, tree.phif021X, tree.phif021Y, tree.phif021Z, tree.phif021E
    #                      ] for event in tree])
    # phif001 = onp.asarray(
    #     [[tree.phif001X, tree.phif001Y, tree.phif001Z, tree.phif001E] for event in tree])
    # phif021 = onp.asarray(
    #     [[tree.phif021X, tree.phif021Y, tree.phif021Z, tree.phif021E] for event in tree])
