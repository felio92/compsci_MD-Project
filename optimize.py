import numpy as np
from distances import *

def descent(x, grad, a=1e-4, prec=1e-10, maxst=1e5, boxsize=(0, 1), pbc=False, ewald=False ):
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
    x_min, x_max = boxsize[0], boxsize[1]
    step = 0
    vecs = vectors(x[-1], boxsize, pbc=pbc^ewald)
    f = grad(x[-1])
    x1 = x[-1] - a * f
    while step < maxst and np.linalg.norm(x[-1] - x1) > prec:
        if pbc:
            x1 = x1 - (x1 > x_max)*(x_max-x_min)
            x1 = x1 + (x1 < x_min)*(x_max-x_min)
        x = np.append(x, x1[None, :, :], axis=0)
        vecs = vectors(x[-1], boxsize, pbc=pbc^ewald)
        f = grad(x[-1])
        x1 = x[-1] - a * f
        step += 1
    return x, step

