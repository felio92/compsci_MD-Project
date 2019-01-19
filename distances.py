
# coding: utf-8

# # Distances

# ### TO DISCUSS:
# ####  1) At what point will we turn distances > cutoff into NaN values?
# ####  2) What units will we use (length, time, energy...)?
# ####  3) Which variables should be globalized?
# ####  4) Forget about symmetrization?
# ####  5) Plotting with respect to PBC is an actual piece of work (include acceleration vectors) 
# 
# 
# 
# 
# ### NOTES TO SELF:
# #### 1) The largest possible distance between two points with PBC is   $\Large \frac{Boxlength}{\sqrt{2}}$
# 

# In[1]:


import numpy as np
import random


# In[5]:


#random (grid-)position sampler in a box with side length max_a
max_a = 5
min_a = 0
#Number of gridpoints per axis
n_grid_x = 300
n_grid_y = 300
#Array with all possible x/y gridpoints, on which positions will be sampled
possible_x, possible_y = np.linspace(min_a,max_a,n_grid_x+1), np.linspace(min_a,max_a,n_grid_y+1)
#number of randomly sampled particles on grid points
n_atoms = 10
coord = np.array([[random.choice(possible_x),random.choice(possible_y)] for i in range(0,n_atoms)])


# In[6]:


#The following function computes connecting vectors between particles with or without respect to PBC
#INPUT: coordinate array of particles of shape (n,2)
#OUTPUT: array of connecting vectors of all particles of shape (n,n,2)
def vectors(coord, pbc=False):
    vecs = coord[:, None, :] - coord[None, :, :]
    if not pbc:
        return vecs
    elif pbc:
        vecs += (vecs<-0.5*max_a)*max_a - (vecs>0.5*max_a)*max_a
        return vecs
    
#Euclidean distance calculator
#INPUT: array of connecting vectors of all particles of shape (n,n,2)
#OUTPUT: array of distances between all particles of shape (n,n)
def distances(vectors):
    return np.linalg.norm(vectors,axis=-1)

#Normalized vectors are needed for the integrator to calculate forces
#This function will normalize the list of connecting vectors obtained from function 'vectors'

#INPUT: array of connecting vectors of all particles of shape (n,n,2) and the norm of all those vectors of shape (n,n)
#OUTPUT: array of normalized connecting vectors of all particles of shape (n,n,2)
def normalize(vectors,distances):
    #set all elements with zero distance to 1 to calculate the norm (avoid ZeroDivisionError)
    return vectors/((distances==0)*1 + distances)[:,:,None]


# In[7]:



