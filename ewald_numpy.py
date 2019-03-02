import numpy as np
from distances import *
from scipy.special import erfc

class Ewald(object):
    #To Do: Better documentation so that retracing the programming steps becomes easier.
    def __init__(self, charges, boxsize, alpha, r_cutoff, n_max, k_cutoff):
        #probably includes redundant parameters, will remove some in the future
        self.boxsize = boxsize
        self.L = boxsize[1] - boxsize[0]
        self.vol = self.L**3
        self.a = alpha
        self.nm = n_max
        self.rc = r_cutoff
        self.kc = k_cutoff
        self.n_max = n_max
        self.q = charges
        self.k_array = np.asarray([[i,j, k] 
                                   for i in range(int(self.kc)+1) 
                                   for j in range(int((self.kc**2-i**2)**(1/2))+1)
                                   for k in range(int((self.kc**2-i**2-j**2)**(1/2)+1)) if (i,j,k)!=(0,0,0)])
        self.k_array = np.concatenate([[i,j,1]*self.k_array for i in [-1, 1] for j in [-1, 1]])
        self.n_array = np.mgrid[-n_max:n_max+1,-n_max:n_max+1,-n_max:n_max+1].reshape(3,-1).T
   
    @property
    def params(self):
        return self.a, self.rc, self.nm, self.kc
    
    @params.setter
    def params(self, params):
        self.a = params[0]
        self.rc = params[1]
        if self.nm != params[2]:
            self.nm = params[2]
            self.n_array = np.mgrid[-self.nm:self.nm+1,-self.nm:self.nm+1,-self.nm:self.nm+1].reshape(3,-1).T
        if self.kc != params[3]:
            self.kc = params[3]
            self.k_array = np.asarray([[i,j, k]
                                   for i in range(int(self.kc)+1)
                                   for j in range(int((self.kc**2-i**2)**(1/2))+1)
                                   for k in range(int((self.kc**2-i**2-j**2)**(1/2)+1)) if (i,j,k)!=(0,0,0)])
            self.k_array = np.concatenate([[i,j,1]*self.k_array for i in [-1, 1] for j in [-1, 1]])
              
    def pot_sr(self, coord):
        i_dvec = self.n_array[:,None,None]-vectors(coord, self.boxsize)
        dist = distances(i_dvec) #calculates the pairwise particle distances, including images
        mask = dist>=self.rc #particle-particle interactions are ignored if interparticle distance exceeds cutoff radius
        dist = np.ma.masked_array(dist, dist==0) #division by zero is prevented
        dist_ma = np.ma.masked_array(dist, mask) #applies cutoff mask to distance matrix
        return np.sum(1/2*self.q[:,None]*self.q[None,:]*1/dist_ma*erfc(self.a*dist_ma))
    
    def S(self,coord):
        #calculates the absolute squared of S(k) (structure factor inside imaginary part of Ewald sum)
        k_r = np.tensordot(self.k_array,coord, axes=(-1, -1)) #calculates the scalar product of all k-vectors with the position vectors k_r[i,j] = dot(k[i],r[j])
        s_re = np.tensordot(self.q, np.cos(2*np.pi*k_r), axes=(-1, -1))
        s_im = np.tensordot(self.q, np.sin(2*np.pi*k_r), axes=(-1, -1))
        return s_re**2 + s_im**2
       
    def pot_lr(self, coord):
        #long-ranged potential that is calculated in reciprocal space. Cutoff is defined by k_cutoff (see init)
        k_abs = np.linalg.norm(self.k_array,axis=-1)
        return 1/(np.pi*self.vol)*np.sum((np.exp(-(np.pi*k_abs/self.a)**2)/k_abs**2)*self.S(coord))
    
    def pot_self(self):
        #self-interaction potential, that has to be corrected for in the Ewald sum
        return self.a/np.sqrt(np.pi)*np.linalg.norm(self.q)**2
    
    def pot(self, coord):
        return np.ma.sum((self.pot_sr(coord),self.pot_lr(coord), - self.pot_self()))
    
    def force_sr(self, coord):
        i_dvec = self.n_array[:,None,None]-vectors(coord, self.boxsize)
        dist = distances(i_dvec)
        mask = dist>=self.rc #particle-particle interactions are ignored if interparticle distance exceeds cutoff radius
        dist = np.ma.masked_array(dist, mask)
        dist = np.ma.masked_array(dist, dist==0) #division by zero is prevented
        s1, s2 = erfc(self.a*dist), 2*self.a/np.sqrt(np.pi)*dist*np.exp(-(self.a*dist)**2)
        return self.q[:, None] * np.ma.sum((self.q[None,:]/(dist**3)*(s1 + s2))[:,:,:,None] * i_dvec, axis=(0, 2))
    
    def force_lr(self, coord):
        dvec = -vectors(coord, self.boxsize)
        k_abs = np.linalg.norm(self.k_array,axis=-1)
        k_r = np.tensordot(self.k_array,dvec, axes=(-1, -1))
        f1, f2 = np.pi/(self.a*self.L), 2*np.pi/self.L
        f3=(1/k_abs**2*np.exp(-(f1*k_abs)**2))
        f4=f3[:,None,None]*np.sin(f2*k_r)
        f5=(self.k_array[:,:,None,None]*f4[:,None,:])
        return 4*self.q[:, None]/self.L * np.sum(f5, axis=(0, -1)).T
    
    def force(self, coord):
        return self.force_lr(coord) + self.force_sr(coord)
        