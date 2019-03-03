import numpy as np
from numba import jit

#creates initial velocities. Mean value given by energy, spread given by variance
def initial_velocities(n_atoms, dim, energy, mass, variance):
    plus_minus = np.array([np.random.normal(np.random.choice([1,-1])*np.sqrt(2*energy/mass),variance) for i in range(n_atoms*dim)])
    return plus_minus.reshape(n_atoms,dim)
    

def new_config(coord,stepsize, boxsize, pbc=False):
    proposal = np.random.normal(coord,stepsize)
    min_a, max_a = boxsize
    if pbc:
        #in case of pbc, shift the values outside of the box accordingly
        proposal += (proposal > max_a)*max_a*(-1) + (proposal < min_a)*max_a
    else:
        #if proposed values lie outside of box, we pull them back in with the next command
        #go back to edge of box if outside
        proposal += (proposal > max_a)*(-1)*(proposal-max_a) + (proposal < min_a)*(min_a-proposal)
    return proposal
@jit
def mcmc(potential, n_atoms,dim,n_steps,stepsize=10000, beta=1, boxsize = (0,1),pbc=False, save_config=False, init_config=None):
    min_a, max_a = boxsize
    coord = (np.random.uniform(min_a, max_a, size=(n_atoms, dim)) if init_config == None else init_config)
    if save_config:
        config = np.zeros((n_steps, n_atoms, dim))
        config[0] = coord
    for i in range(1,n_steps):
        #calculate the sum of potentials
        sumpot = np.sum(potential(coord, pbc))
        #propose new configuration (normally distributed around the old coordinates, variance given by stepsizei)
        proposal = new_config(coord, stepsize, boxsize, pbc)
        proposed_sumpot = np.sum(potential(proposal, pbc))
        if sumpot >= proposed_sumpot or np.exp((sumpot-proposed_sumpot)*beta) > np.random.uniform(0,1):
            coord = proposal
        if save_config: config[i] = coord
    if save_config:
        return config
    return coord
