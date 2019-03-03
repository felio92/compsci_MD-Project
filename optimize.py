def descent( coord, grad, a=1e-4, prec=1e-6, maxst=1e5, boxsize=(0,1), pbc=False, ewald=False ):
    """Gradient Descent

    Arguments:
        coord   (float): position vectors (dim = n x 3)
        grad (function): gradient calculation
        a       (float): alpha, 'learning rate', influence of gradient per step
        prec    (float): precision, difference between steps
        maxst     (int): max # of steps

    Output:
        x    (float): tensor of position vectors at each step (dim = n x 3 x step)
        step: # of steps needed to converge"""
    import numpy as np
    from distances import vectors
    
    assert int(maxst) % maxst == 0
    
    step = 0
    x = np.zeros((int(maxst),) + coord.shape)
    x[step] = coord
    x1 = x[step] - a * grad(x[step])
    if pbc:
        xmin, xmax = boxsize[0], boxsize[1]
        assert xmin < xmax
        L = xmax - xmin
        x1[x1 < xmin] = xmax - (xmin - x1[x1 < xmin]) %  L
        x1[x1 > xmax] = xmin + (x1[x1 > xmax] - xmax) %  L
    while step+1 < maxst and np.linalg.norm(x[step] - x1) > prec:
        step += 1
        x[step] = x1
        x1 = x[step] - a * grad(x[step])
        if pbc:
            x1[x1 < xmin] = xmax - (xmin - x1[x1 < xmin]) %  L
            x1[x1 > xmax] = xmin + (x1[x1 > xmax] - xmax) %  L
    return x[:step], step

