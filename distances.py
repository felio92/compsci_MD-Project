
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

import numpy as np
import random

#The following function computes connecting vectors between particles with or without respect to PBC
#INPUT: coordinate array of particles of shape (n,2)
#OUTPUT: array of connecting vectors of all particles of shape (n,n,2)
def vectors(coord, boxsize, pbc=False):
    vecs = coord[:, None, :] - coord[None, :, :]
    if not pbc:
        return vecs
    elif pbc:
        L = boxsize[1] - boxsize[0] #calculate boxlength
        vecs += (vecs<-0.5*L)*L - (vecs>0.5*L)*L
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

