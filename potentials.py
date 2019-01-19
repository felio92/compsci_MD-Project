
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import distances
#%load_ext line_profiler


# In[2]:


class potentials(object):
        
    def coulomb(coord, q, eps0=1, pbc=False):
        vectors = distances.vectors(coord, pbc)
        dist = distances.distances(vectors)
        if dist.ndim!=0:
            dist[dist!=0] = 1/dist[dist!=0]
        else:
            dist = 1/dist
        return 1/(4*np.pi*eps0) * np.dot(dist, q)
    
    def LJ(coord, eps=1, sig=1, pbc=False):
        vectors = distances.vectors(coord, pbc)
        dist = distances.distances(vectors)
        if dist.ndim!=0:
            dist[dist!=0] = 1/dist[dist!=0]
        else:
            dist = 1/dist
        pot_rep = np.dot(eps*sig**12,dist**12) 
        pot_atr = np.dot(eps*sig**6,dist**6)
        pot = pot_rep - pot_atr
        return np.sum(pot, axis=-1)
    
    def harmonic(coord, r0, k, pbc=False):
        vectors = distances.vectors(coord, pbc)
        return k/2*(np.linalg.norm(r - r0, axis=-1))**2


# In[3]:


class gradients(object):

    def coulomb(coord, q, eps0=1, pbc=False):
        vectors = distances.vectors(coord, pbc)
        dist = distances.distances(vectors)
        dist[dist!=0] = 1/dist[dist!=0]**3
        D = dist[:,:,None]*vectors
        return q[:, None]*np.einsum("ijk, j",D, q)
    
    def LJ(coord, sig=1, eps=1, pbc=False):
        vectors = distances.vectors(coord, pbc)
        dist = distances.distances(vectors)
        dist[dist!=0] = 1/dist[dist!=0]
        D_att = 6 * sig**6 * dist**8
        D_rep = -12 * sig**12 * dist**14
        D = 4*(eps*(D_att + D_rep))[:, :, None]*vectors
        return np.sum(D, axis=-2)

