import numpy as np
def vvJ(potential_gradient, position_init, velocity_init, mass, T, time_step):
    size = int(T/time_step)        # number of time steps
    n = len(position_init)         # number of particles
    dim = position_init.shape[-1]  # number of dimension
    m = mass
    # creating positiion, velocity and acceleration container
    position_matrix, velocity_matrix, acceleration_matrix = np.zeros((size, n, dim)), np.zeros((size, n, dim)), np.zeros((size, n, dim))
    # initialization by adding the start configuration
    position_matrix[0], velocity_matrix[0], acceleration_matrix[0] = position_init, velocity_init, potential_gradient(position_init)
    # time iteration
    for t in range(1, size):
        # rename # kostet das rechenleistung ? bennenung lieber von anfang anders?
        p = position_matrix[t-1]
        v = velocity_matrix[t-1]
        a = acceleration_matrix[t]
        gp = potential_gradient(p)
        # vv approximation
        p_new = p + time_step*v - (time_step**2)/(2*m)*gp
        gp_new = potential_gradient(p_new) 
        v_new = v - time_step/(2*m) * (gp + gp_new)
        a = gp_new
        #v_new[:,1][p_new[:,1]<0]*=-1 ???
        # write in pos and vel container
        #position_matrix[t], velocity_matrix[t] = p_new, v_new
        position_matrix[t], velocity_matrix[t], acceleration_matrix[t] = p_new, v_new, a
    # returning position and velocety container
    return position_matrix, velocity_matrix, acceleration_matrix
