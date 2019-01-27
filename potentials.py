import numpy as np
import matplotlib.pyplot as plt
import distances

class potentials(object):
        
    def coulomb(dist, q, eps0=1):
        if dist.ndim!=0:
            dist[dist!=0] = 1/dist[dist!=0]
        else:
            dist = 1/dist
        return 1/(4*np.pi*eps0) * np.dot(dist, q)
    
    def LJ(dist, eps=1, sig=1):
        if dist.ndim!=0:
            dist[dist!=0] = 1/dist[dist!=0]
        else:
            dist = 1/dist
        pot_rep = np.dot(eps*sig**12,dist**12) 
        pot_atr = np.dot(eps*sig**6,dist**6)
        pot = pot_rep - pot_atr
        return np.sum(pot, axis=-1)
    
    def harmonic(coord, boxsize, pbc=False, r0=0, k=1):
        vecs = coord - r0
        max_a = boxsize[1] - boxsize[0]
        if pbc:
            vecs += (vecs<-0.5*max_a)*max_a - (vecs>0.5*max_a)*max_a
        return k/2*(np.linalg.norm(vecs, axis=-1))**2

class gradients(object):

    def coulomb(vecs, q, eps0=1):
        dist = distances.distances(vecs)
        dist[dist!=0] = 1/dist[dist!=0]**3
        D = dist[:,:,None]*vecs
        return q[:, None]*np.einsum("ijk, j",D, q)
    
    def LJ(vecs, sig=1, eps=1):
        dist = distances.distances(vecs)
        dist[dist!=0] = 1/dist[dist!=0]
        D_att = 6 * sig**6 * dist**8
        D_rep = -12 * sig**12 * dist**14
        D = 4*(eps*(D_att + D_rep))[:, :, None]*vecs
        return np.sum(D, axis=-2)
    
    def harmonic(coord, boxsize, pbc=False, r0=0, k=1):
        max_a = boxsize[1] - boxsize[0]
        vecs = coord - r0
        if pbc:
            vecs += (vecs<-0.5*max_a)*max_a - (vecs>0.5*max_a)*max_a
        return k*vecs

