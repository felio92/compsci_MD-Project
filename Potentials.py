#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def coulomb(dist, q, eps0=1):
    dist[dist!=0] = 1/dist[dist!=0]
    pot = 1/(4*np.pi*eps0) * np.dot(dist, q)
    return pot


# In[1]:


def LJ(dist, eps, sig):
    dist[dist!=0] = 1/dist[dist!=0]
    pot_rep = np.dot(np.multiply(eps,sig**12),dist**12) 
    pot_atr = np.dot(np.multiply(eps,sig**6),dist**6)
    pot = pot_rep - pot_atr
    return pot


# In[6]:


def harmonic(r, r0, k):
    return k*(r - r0)


# In[ ]:




