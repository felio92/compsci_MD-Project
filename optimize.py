import numpy as np
from distances import *

def descent(x, grad, a=1e-4, prec=1e-10, maxst=1e5, boxsize=(0, 1), pbc=False, save_config=True, ewald=False ):
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
    if save_config:
        config = np.zeros((maxst,)+x.shape)
        config[0] = x
    x_min, x_max = boxsize[0], boxsize[1]
    step = 0
    vecs = vectors(x, boxsize, pbc=pbc^ewald)
    f = grad(x)
    x1 = x - a * f
    step+=1
    while step < maxst and np.linalg.norm(x - x1) > prec:
        if pbc:
            x1 = x1 - (x1 > x_max)*(x_max-x_min)
            x1 = x1 + (x1 < x_min)*(x_max-x_min)
        if save_config: config[step] = x1
        vecs = vectors(x1, boxsize, pbc=pbc^ewald)
        f = grad(x1)
        x1 = x1 - a * f
        step += 1
    if save_config:
        return config[:step], step
    return x1, step