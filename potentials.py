import numpy as np
import matplotlib.pyplot as plt
import distances

class potentials(object):
    '''Potentials class containing class methods to calculate a number of different potentials relevant to MD-simulations. The potentials available are:
    -coulomb(dist, q) 
    -LJ(dist, eps=1, sig=1)
    -harmonic(coord, boxsize, pbc=False, r0=0, k=1)
    All inputs are assumed to be dimensionless, with the parameters being expressed in atomic units. More information can be found in the docstring of the respective potential.'''
    def pot_barrier(p, boxsize):
        '''
        darüber müssen wir nachdenken! ich denke das
        '''
        p_min, p_max = boxsize[0], boxsize[1]
        L = p_max - p_min
        V_r = 1/2*(p - (p_max - L/4))**2 * (p>=p_max)
        V_l = 1/2*(p - (p_min + L/4))**2 * (p<=p_min)
        V = V_r + V_l 
        return np.sum(V, axis=-1)*5  
    def coulomb(dist, q):
        '''Calculate the scalar coulomb potential in atomic units of n charged particles. Input values for this function are:
        -dist, the pairwise distances of the particles as a (n x n) numpy array, expressed in bohr radii.
        -q, a numpy array containing the particle charges.
        Output is the total coulomb potential expressed in hartree.'''
        d = np.copy(dist)
        if d.ndim!=0:
            #Prevent division by zero.
            d[d!=0] = 1/d[d!=0]
        else:
            d = 1/d
        return np.dot(d, q)
    
    def LJ(dist, eps=1, sig=1):
        '''Calculate the scalar Lennard Jones potential in atomic units of n particles. Input values for this function are:
        -dist, the pairwise distances of the particles as a (n x n) numpy array expressed in bohr radii.
        -eps, depth of the potential well expressed in hartree. Can be either a scalar or (n x n) numpy array.
        -sig, root of potential expressed in bohr radii. Can be either a scalar or a (n x n) numpy array.
        Output is the total LJ potential in hartree.'''
        d = np.copy(dist)
        if d.ndim!=0:
            #Prevent division by zero.
            d[d!=0] = 1/d[d!=0]
        else:
            d = 1/d
        pot_rep = np.dot(eps*sig**12,d**12) 
        pot_atr = np.dot(eps*sig**6,d**6)
        pot = 4*(pot_rep - pot_atr)
        return np.sum(pot, axis=-1)
    
    def LJ_cut(dist, eps=1, sig=1, cutoff=2.5):
        '''Calculate the shifted scalar Lennard Jones potential in atomic units of n particles with a cutoff being applied. Input values for this function are:
        -dist, the pairwise distances of the particles as a (n x n) numpy array expressed in bohr radii.
        -eps, depth of the potential well expressed in hartree. Can be either a scalar or (n x n) numpy array.
        -sig, root of potential expressed in bohr radii. Can be either a scalar or a (n x n) numpy array.
        -cutoff, sclar multiple of sig. The potential is shifted by the potential value at point cutoff*sig and is set to zero beyond this point.
        Output is the total LJ potential in hartree.'''
  
        def LJ_pot(dist, eps=1, sig=1):
            if dist.ndim!=0:
                #Prevent division by zero.
                dist[dist!=0] = 1/dist[dist!=0]
            else:
                dist = 1/dist
            pot_rep = np.dot(eps*sig**12,dist**12) 
            pot_atr = np.dot(eps*sig**6,dist**6)
            pot = 4*(pot_rep - pot_atr)
            return pot
        
        d = np.copy(dist)
        r = sig*cutoff
        f = lambda dist: LJ_pot(dist, eps, sig) - LJ_pot(np.asarray([r]), eps, sig)
        res = np.zeros_like(d)
        res += np.piecewise(d, [d<=r, d>r], [f, 0])
        return np.sum(res, axis=-1)
    
    
    def harmonic(coord, boxsize, pbc=False, r0=0, k=1):
        '''Calculate the potential of n particles inside a harmonic potential. Input values for this function are:
        -coord, the particle positions given in bohr radii.
        -boxsize, a tuple consisting of the starting value of the box edge and the end value, given in bohr radii. This parameter only has an effect in the case of periodic boundary conditions.
        -pbc, boolean defining, if the boundary conditions are periodic or not.
        -r0, origin of the harmonic well in bohr radii. Can be either a scalar or a vector with the same dimensions as the particle coordinates.
        -k, strength of the harmonic potential expressed in [hartree]/[bohr radius]^2
        Output is the total harmonic potential in hartree.'''
        vecs = coord - r0
        max_a = boxsize[1] - boxsize[0]
        if pbc:
            vecs += (vecs<-0.5*max_a)*max_a - (vecs>0.5*max_a)*max_a
        return k/2*(np.linalg.norm(vecs, axis=-1))**2

class gradients(object):
    '''Gradients class containing class methods to calculate a number of different forces relevant to MD-simulations. The forces available are:
        -coulomb(vecs, q) 
        -LJ(vecs, eps=1, sig=1)
        -harmonic(coord, boxsize, pbc=False, r0=0, k=1)
        All inputs are assumed to be dimensionless, with the parameters being expressed in atomic units. More information can be found in the docstring of the respective gradient.'''
    
    def pot_barrier(p, boxsize):
        '''
        Calculate the gradient of the barrier potential of the box. 
        Input
        p (numpy.ndarray(n, dim)): initial configuration in dim dimensions
        boxsize (douple): min box edges, max box edges
        Output
        V (): gradient of barrier potential, for stronger potential (needed for small boxes with few particles) multiplication with constant.
        '''
        p_min, p_max = boxsize[0], boxsize[1]
        L = p_max - p_min
        V_r = (p - (p_max - L/4))*(p>=p_max)
        V_l = (p - (p_min + L/4))*(p<=p_min)
        V = V_r+ V_l 
        return V *5 
    
    def coulomb(vecs, q):
        '''Calculate the coulomb force in atomic units for n charged particles. Input values for this function are:
        -vecs, the pairwise distance vectors of the particles as a (n x n x dim) numpy array, expressed in bohr radii.
        -q, a numpy array containing the particle charges, expressed in elementary charges.
        Output is a (n x dim) numpy array containing the forces acting on each particle, expressed in [hartree]/[bohr radius].'''
        dist = distances.distances(vecs)
        dist[dist!=0] = 1/dist[dist!=0]**3 #prevent division by zero
        D = dist[:,:,None]*vecs
        return q[:, None]*np.einsum("ijk, j",D, q)
    
    def LJ(vecs, eps=1, sig=1):
        '''Calculate the Lennard Jones force in atomic units for n particles. Input values for this function are:
-vecs, the pairwise distance vectors of the particles as a (n x n x dim) numpy array, expressed in bohr radii.
-eps, depth of the potential well expressed in hartree. Can be either a scalar or (n x n) numpy array.
        -sig, root of potential expressed in bohr radii. Can be either a scalar or a (n x n) numpy array.
Output is a (n x dim) numpy array containing the forces acting on each particle, expressed in [hartree]/[bohr radius].'''
        dist = distances.distances(vecs)
        dist[dist!=0] = 1/dist[dist!=0]
        D_att = 6 * sig**6 * dist**8
        D_rep = -12 * sig**12 * dist**14
        D = 4*(eps*(D_att + D_rep))[:, :, None]*vecs
        return np.sum(D, axis=-2)
    def LJ_cut(vecs, eps=1, sig=1, cutoff=2.5):
        '''Calculate the Lennard Jones force in atomic units for n particles with a cutoff. Input values for this function are:
-vecs, the pairwise distance vectors of the particles as a (n x n x dim) numpy array, expressed in bohr radii.
-eps, depth of the potential well expressed in hartree. Can be either a scalar or (n x n) numpy array.
        -sig, root of potential expressed in bohr radii. Can be either a scalar or a (n x n) numpy array.
        -cutoff, scalar multiple of sig. The potential gradient is set to zero beyond the point sig*cutoff.
Output is a (n x dim) numpy array containing the forces acting on each particle, expressed in [hartree]/[bohr radius].'''
        def LJ_grad(vecs, eps=1, sig=1):
            dist = distances.distances(vecs)
            if dist.ndim!=0:
                #Prevent division by zero.
                dist[dist!=0] = 1/dist[dist!=0]
            else:
                dist = 1/dist
            D_att = 6 * sig**6 * dist**8
            D_rep = -12 * sig**12 * dist**14
            D = 4*(eps*(D_att + D_rep))[ :, None]*vecs
            return D
    
        r = sig*cutoff
        dist = distances.distances(vecs)
        f = lambda vecs: LJ_grad(vecs, eps, sig)
        res = np.zeros_like(vecs)
        res += np.piecewise(vecs, [dist<=r, dist>r], [f, 0])
        return np.sum(res, axis=-2)

    def harmonic(coord, boxsize, pbc=False, r0=0, k=1):
        '''Calculate the harmonic force in atomic units for n particles. Input values for this function are:
        -coord, the particle positions given in bohr radii.
        -boxsize, a tuple consisting of the starting value of the box edge and the end value, given in bohr radii. This parameter only has an effect in the case of periodic boundary conditions.
        -pbc, boolean defining, if the boundary conditions are periodic or not.
        -r0, origin of the harmonic well in bohr radii. Can be either a scalar or a vector with the same dimensions as the particle coordinates.
        -k, strength of the harmonic potential expressed in [hartree]/[bohr radius]^2
        Output is a (n x dim) numpy array containing the forces acting on each particle, expressed in [hartree]/[bohr radius].
        '''
        max_a = boxsize[1] - boxsize[0]
        vecs = coord - r0
        if pbc:
            vecs += (vecs<-0.5*max_a)*max_a - (vecs>0.5*max_a)*max_a
        return k*vecs

