import numpy as np

def euler(potential_gradient, position_init, velocity_init, mass, T, time_step, boxsize=(0,1)):
    """
    This function realise the Euler integration scheme. This scheme approximating the 
    integral with the finite sum by the Taylor expansion, whereby the quadratic and 
    higher-order terms are ignored.
    
    Arguments:
        potential_gradient (function): computes potential gradient for particle-positions,
                                       choose between open and closed system
        position_init (numpy.ndarray(n, dim)): initial configuration in dim dimensions
        velocity_init (numpy.ndarray(n, dim)): initial velocity in dim dimensions
        mass (numpy.ndarray(n)): mass of each particle
        T (int): total time of integration
        time_step (float): step size for integration 
        boxsize (double): first right maximum value, second left minimum value

        
    Returns:
        position_matrix (numpy.ndarray(size, n, dim)): configuraiton trajectory
        velocity_matrix (numpy.ndarray(size, n, dim)): velocity trajectory
        acceleration_matrix (numpy.ndarray(size, n, dim)): acceleration trajectory
       
    """
    size = int(T/time_step)  
    n = len(position_init)   
    dim = position_init.shape[-1]
    m = mass
    position_matrix, velocity_matrix, acceleration_matrix = np.zeros((size, n, dim)), np.zeros((size, n, dim)), np.zeros((size, n, dim))
    position_matrix[0], velocity_matrix[0], acceleration_matrix[0] = position_init, velocity_init, - 1/m * potential_gradient(position_init, boxsize=boxsize)
    for t in range(1, size):
        p = position_matrix[t-1]
        v = velocity_matrix[t-1]
        a = acceleration_matrix[t]
        gp = potential_gradient(p, boxsize)
        p_new = p + time_step*v
        v_new = v - time_step/m * gp
        a = - potential_gradient(p_new, boxsize)/m
        position_matrix[t], velocity_matrix[t], acceleration_matrix[t] = p_new, v_new, a
    return position_matrix, velocity_matrix, acceleration_matrix

def vv(potential_gradient, position_init, velocity_init, mass, T, time_step, boxsize=(0,1), pbc=False):
    
    """
    The function vv realise the integration scheme of the Velocity-Verlet Algorithm. 
    The numerical procedure for solving ODEs is based on approximating the integral by
    Taylor expansion and ignoring the cubic and higher-order terms.
    
    Arguments:
        potential_gradient (function): computes potential gradient for particle-positions 
        position_init (numpy.ndarray(n, dim)): initial configuration in dim dimensions
        velocity_init (numpy.ndarray(n, dim)): initial velocity in dim dimensions
        mass (numpy.ndarray(n)): mass of each particle
        T (int): total time of integration
        time_step (float): step size for integration 
        boxsize (double): first right maximum value, second left minimum value
        pbc (boolean): periodic boundary conditions (True) or not periodic (False)
        
    Returns:
        position_matrix (numpy.ndarray(size, n, dim)): configuraiton trajectory
        velocity_matrix (numpy.ndarray(size, n, dim)): velocity trajectory
        acceleration_matrix (numpy.ndarray(size, n, dim)): acceleration trajectory
       
    """
    
    size = int(T/time_step)        
    n = len(position_init)         
    dim = position_init.shape[-1]  
    m = mass
    p_min, p_max = boxsize[0], boxsize[1]
    position_matrix, velocity_matrix, acceleration_matrix = np.zeros((size, n, dim)), np.zeros((size, n, dim)), np.zeros((size, n, dim))
    position_matrix[0], velocity_matrix[0], acceleration_matrix[0] = position_init, velocity_init, potential_gradient(position_init, boxsize=boxsize)
    for t in range(1, size):
        p = position_matrix[t-1]
        v = velocity_matrix[t-1]
        a = acceleration_matrix[t]
        gp = potential_gradient(p, boxsize)
        p_new = p + time_step*v - (time_step**2)/(2*m)*gp
        if pbc:
            p_new = p_new - (p_new > p_max)*(p_max-p_min)
            p_new = p_new + (p_new < p_min)*(p_max-p_min)
        gp_new = potential_gradient(p_new, boxsize) 
        v_new = v - time_step/(2*m) * (gp + gp_new)
        a = - gp_new/m                              # useless for algorithm! needed for visualisation?
        position_matrix[t], velocity_matrix[t], acceleration_matrix[t] = p_new, v_new, a
    return position_matrix, velocity_matrix, acceleration_matrix
 
def langevin(potential_gradient, position_init, velocity_init, mass, total_time, time_step, damping, beta, temp, boxsize, pbc=False):
    
    """
    This function realise the integration scheme of Langevin dynamics with the BAOAB-Algorithm. 
    This scheme includes a thermostat thus solving ODEs leads to a MD-Simulation in the NVT-ensemble.
    
    Arguments:
        potential_gradient (function): computes potential gradient for particle-positions 
        position_init (numpy.ndarray(n, dim)): initial configuration in dim dimensions
        velocity_init (numpy.ndarray(n, dim)): initial velocity in dim dimensions
        mass (numpy.ndarray(n)): mass of each particle
        total_time (int): total time of integration
        time_step (float): step size for integration 
        damping (float): isotopic friction constant (couple the system to the bath), zero for not coupled
        beta (float): inverse temperature
        temp (float): tempering parameter, choose positive value to warm up the system,
                      negative to cooling for each time step and zero no tempering
        boxsize (double): first right maximum value, second left minimum value
        pbc (boolean): periodic boundary conditions (True) or not periodic (False)
        
    Returns:
        position_matrix (numpy.ndarray(size, n, dim)): configuraiton trajectory
        velocity_matrix (numpy.ndarray(size, n, dim)): velocity trajectory
        acceleration_matrix (numpy.ndarray(size, n, dim)): acceleration trajectory
       
    """
    size = int(total_time/time_step)    
    n = len(position_init)              
    dim = position_init.shape[-1]       
    m = mass
    p_min, p_max = boxsize[0], boxsize[1]
    position_matrix, velocity_matrix, acceleration_matrix = np.zeros((size, n, dim)), np.zeros((size, n, dim)), np.zeros((size, n, dim))
    position_matrix[0], velocity_matrix[0], acceleration_matrix[0] = position_init, velocity_init, potential_gradient(position_init, boxsize)
    R_t = np.random.randn(n, dim)
    fri = np.exp(-damping*time_step)
    noi = np.sqrt((1-fri**2)/(beta*m))
    for t in range(1, size):
        p = position_matrix[t-1]
        v = velocity_matrix[t-1]
        a = acceleration_matrix[t]
        gp = potential_gradient(p, boxsize)
        v_new = v - time_step/(2*m) * gp                     
        p_new = p + time_step/2 *v_new                        
        if pbc:
            p_new = p_new - (p_new > p_max)*(p_max-p_min)
            p_new = p_new + (p_new < p_min)*(p_max-p_min)
        v_new = fri*v_new + noi* R_t                          
        p_new = p_new + time_step/2 *v_new                    
        if pbc:
            p_new = p_new - (p_new > p_max)*(p_max-p_min)
            p_new = p_new + (p_new < p_min)*(p_max-p_min)
        gp = potential_gradient(p_new, boxsize)
        v_new = v_new - time_step/(2*m) * gp                  
        a = - gp/m
       # beta = beta + temp*t
        position_matrix[t], velocity_matrix[t], acceleration_matrix[t] = p_new, v_new, a
    return position_matrix, velocity_matrix, acceleration_matrix
