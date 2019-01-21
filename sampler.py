#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from distances import *
from Potentials import *


# In[64]:


#creates initial velocities. Mean value given by energy, spread given by variance
def initial_velocities(n_atoms, dim, energy, mass, variance):
    plus_minus = np.array([np.random.normal(np.random.choice([1,-1])*np.sqrt(2*energy/mass),variance) for i in range(n_atoms*dim)])
    return plus_minus.reshape(n_atoms,dim)
    

def new_config(coord,stepsize,pbc=False):
    proposal = np.random.normal(coord,stepsize)
    if pbc:
        #in case of pbc, shift the values outside of the box accordingly
        proposal += (proposal > max_a)*max_a*(-1) + (proposal < min_a)*max_a
    else:
        #if proposed values lie outside of box, we pull them back in with the next command
        #go back to edge of box if outside
        proposal += (proposal > max_a)*(-1)*(proposal-max_a) + (proposal < min_a)*(min_a-proposal)
    return proposal


# In[55]:


#Variables to be defined: init_coord, LJ:sigma, LJ:epsilon, Coulomb:epsilon charges, potentials, boxsize, n_atoms
@jit
def mcmc(n_atoms,dim,n_steps,stepsize,pbc=False):
    coord = np.random.rand(n_atoms,dim)*5
    for i in range(n_steps):
        dist=distances(vectors(coord,pbc))
        #calculate the sum of potentials
        sumpot = np.sum(LJ(dist,sigma,eps))
        #propose new configuration (normally distributed around the old coordinates, variance given by stepsizei)
        proposal = new_config(coord,stepsize,pbc)
        proposed_sumpot = np.sum(LJ(distances(vectors(proposal,pbc)),sigma,eps))
        prob = np.exp((sumpot-proposed_sumpot)*beta)
        if prob > np.random.uniform(0,1):
            coord = proposal
    return coord


# In[47]:





# In[67]:





# In[ ]:





# In[ ]:





# In[ ]:




