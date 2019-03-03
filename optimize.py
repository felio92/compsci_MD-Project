import numpy as np

def descent( x, grad, a=1e-4, prec=1e-6, maxst=1e5, boxsize=(0,1), pbc=False, save_config=False):
    """Gradient Descent

    Arguments:
        coord    (float): position vectors (dim = n x 3)
        grad  (function): gradient calculation
        a        (float): alpha, 'learning rate', influence of gradient per step
        prec     (float): precision, difference between steps
        maxst      (int): max # of steps
        boxsize (double): tuple, box is quadratic/cubic
        pbc == True: enables periodic boundaries
        save_config == True: saves all positions

    Output:
        x    (float): tensor of position vectors at each step (dim = n x 3 x step)
        step: # of steps needed to converge"""
    assert int(maxst) % maxst == 0
    # Initialize:
    step = 0
    if save_config:
        config = np.zeros((int(maxst),) + x.shape)
        config[step] = x
    x1 = x - a * grad(x)
    if pbc:
        xmin, xmax = boxsize[0], boxsize[1]
        assert xmin < xmax
        L = xmax - xmin
        x1[x1 < xmin] = xmax - (xmin - x1[x1 < xmin]) %  L
        x1[x1 > xmax] = xmin + (x1[x1 > xmax] - xmax) %  L
    # Calculate gradient until convergence:
    while step+1 < maxst and np.linalg.norm(x - x1) > prec:
        step += 1
        if save_config: config[step] = x1
        x = x1
        x1 = x - a * grad(x)
        if pbc:
            x1[x1 < xmin] = xmax - (xmin - x1[x1 < xmin]) %  L
            x1[x1 > xmax] = xmin + (x1[x1 > xmax] - xmax) %  L
    if save_config:
        return config[:step+1], step
    return x1

