import numpy as np

from potentials import gradients
from distances import vectors

def gradient_HO_LJ( x, q, boxsize ):
    vecs = vectors(x, boxsize)
    return gradients.harmonic(x, boxsize) + gradients.LJ(vecs)
  
def gradient_HO_LJ_Cou( x, q, boxsize ):
    vecs = vectors(x, boxsize)
    return gradients.harmonic(x, boxsize) + gradients.LJ(vecs) - gradients.coulomb(vecs, q)

def descent( x, q, gradient, a=1e-4, prec=1e-10, maxst=1e6, k=.1, boxsize=(0,1) ):
    """Gradient Descent
    
    Arguments:
        x     (float): position vectors (n x 3)
        q       (int): charge (n)
        gradient     : function calculating a gradient
        a     (float): 'learning rate' alpha
        prec  (float): difference between steps, precision
        maxst   (int): max # of steps
        k     (float): factor for harmonic trap
    
    Output:
        x     (float): position matrix of all steps (n x dim x step-1)
        step-1  (int): # of steps needed to convergence
    """
    x = x[None, :, :]
    step = 0
    vecs = vectors(x[-1], boxsize)
    f = grad(x[-1], q, boxsize)
    x1 = x[-1] - a * f
    while step < maxst and np.linalg.norm(x[-1] - x1) > prec:
        x = np.append(x, x1[None, :, :], axis=0)
        vecs = vectors(x[-1], boxsize)
        f = grad(x[-1], q, boxsize)
        x1 = x[-1] - a * f
        step += 1
    return x, step-1

