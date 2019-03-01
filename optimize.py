import numpy as np
import torch
import random

from potentials import *
from distances import *
from sampling import *

def dist_torch(x): 
    """Calculates distance vectors and distances (euclidian norm of vecs)
    
    Arguments:
        x (float): position vectors (dim = N x 3)
    
    Output:
        dist (float): distances between particle pairs (dim = N x N)
        vecs (float): distance vectors between particle pairs (dim = N x N x 3)
    """
    x = torch.Tensor(x)
    vecs = x[None, :, :] - x[:, None, :]       
    return torch.norm(vecs, dim=-1), vecs

def gradLJ_t(x, sig=1, eps=1):
    dist, vecs = dist_torch(x)
    dist[dist!=0] = 1/dist[dist!=0]
    D_att = 6 * sig**6 * dist**8
    D_rep = -12 * sig**12 * dist**14
    D = 4*(eps*(D_att + D_rep))[:, :, None]*vecs
    return torch.sum(D, dim=-2)

def descent( x, q, grad, a=1e-4, prec=1e-10, maxst=1e6, k=.1, boxsize=(0, 1) ):
    """Gradient Descent
    
    Arguments:
        x    (float): position vectors (dim = n x 3)
        q: charge
        a    (float): 'learning rate' alpha = 1e-4
        prec (float): difference between steps, precision = 1e-10
        maxst  (int): max # of steps, maxst = 1e6
        k: factor harmonic pot
    
    Output:
        x: position array,
        step: # of steps needed to convergence
    """
    x = x[None, :, :]
    step = 0
    vecs = vectors(x[-1], boxsize)
    f = grad(x[-1], q)
    x1 = x[-1] - a * f
    while step < maxst and np.linalg.norm(x[-1] - x1) > prec:
        x = np.append(x, x1[None, :, :], axis=0)
        vecs = vectors(x[-1], boxsize)
        f = grad(x[-1], q)
        x1 = x[-1] - a * f
        step += 1
    return x, step

