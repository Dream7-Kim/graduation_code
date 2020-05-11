import jax.numpy as np
import logging

def deinsum(subscript, aa, bb):
    real = np.einsum(subscript, aa[0], bb[0]) - np.einsum(subscript, aa[1], bb[1])
    imag = np.einsum(subscript, aa[0], bb[1]) + np.einsum(subscript, aa[1], bb[0])
    return np.stack([real, imag], axis=0)

def deinsum_ord(subscript, aa, bb):
    real = np.einsum(subscript, aa, bb[0])
    imag = np.einsum(subscript, aa, bb[1])
    return np.stack([real, imag], axis=0)

def dabs(aa):
    return aa[0]**2 + aa[1]**2 # 因为是纵向叠加所以aa[0]是第一行

def dconj(aa):
    return dplex(aa.val[0], -aa.val[1])

def dtomine(aa):
    return np.stack([np.real(aa), np.imag(aa)], axis=0)

def dconstruct(aa, bb):
    return np.stack([aa, bb], axis=0) # 纵向叠加数组 

def ddivide(a, bb):
    real = a * bb[0] / dabs(bb)
    imag = -a * bb[1] / dabs(bb)
    return np.stack([real, imag], axis=0)