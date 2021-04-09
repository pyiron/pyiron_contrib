import numpy as np

def G(r, poissons_ratio, shear_modulus, min_distance=0):
    """ Green's function """
    r = np.array(r).reshape(-1, 3)
    R = np.linalg.norm(r, axis=-1)
    vorfaktor = 1/(16*shear_modulus*np.pi*(1-poissons_ratio))
    return vorfaktor*((3-4*poissons_ratio)*np.eye(3)+np.einsum('ni,nj,n->nij', r, r, 1/R**2))/R

def dG(x, poissons_ratio, shear_modulus, min_distance=0):
    """ first derivative of the Green's function """
    E = np.eye(3)
    r = np.array(x).reshape(-1, 3)
    R = np.linalg.norm(r, axis=-1)
    distance_condition = R<min_distance
    R[distance_condition] = 1
    r = np.einsum('ni,n->ni', r, 1/R)
    B = 1/(16*np.pi*shear_modulus*(1-poissons_ratio))
    A = (3-4*poissons_ratio)*B
    v = -A*np.einsum('ik,nj->nijk', E, r)
    v += B*np.einsum('ij,nk->nijk', E, r)
    v += B*np.einsum('jk,ni->nijk', E, r)
    v -= 3*B*np.einsum('ni,nj,nk->nijk', r, r, r)
    v = np.einsum('nijk,n->nijk', v, 1/R**2)
    v[distance_condition] *= 0
    return v

def ddG(x, poissons_ratio, shear_modulus, min_distance=0):
    """ Second derivative of the Green's function """
    E = np.eye(3)
    r = np.array(x).reshape(-1, 3)
    R = np.linalg.norm(r, axis=-1)
    distance_condition = R<min_distance
    R[distance_condition] = 1
    r = np.einsum('ni,n->ni', r, 1/R)
    A = (3-4*poissons_ratio)/(16*np.pi*shear_modulus*(1-poissons_ratio))
    B = 1/(16*np.pi*shear_modulus*(1-poissons_ratio))
    v = -A*np.einsum('ik,jl->ijkl', E, E)
    v = v+3*A*np.einsum('ik,nj,nl->nijkl', E, r, r)
    v = v+B*np.einsum('il,jk->ijkl', E, E)
    v -= 3*B*np.einsum('il,nj,nk->nijkl', E, r, r)
    v = v+B*np.einsum('ij,kl->ijkl', E, E)
    v -= 3*B*np.einsum('ni,nj,kl->nijkl', r, r, E)
    v -= 3*B*np.einsum('ij,nk,nl->nijkl', E, r, r)
    v -= 3*B*np.einsum('jk,ni,nl->nijkl', E, r, r)
    v -= 3*B*np.einsum('jl,ni,nk->nijkl', E, r, r)
    v += 15*B*np.einsum('ni,nj,nk,nl->nijkl', r, r, r, r)
    v = np.einsum('nijkl,n->nijkl', v, 1/R**3)
    v[distance_condition] *= 0
    return v

def displacement_field(r, dipole_tensor, poissons_ratio, shear_modulus, min_distance=0):
    g_tmp = dG(r, poissons_ratio=poissons_ratio, shear_modulus=shear_modulus, min_distance=min_distance)
    if dipole_tensor.shape==(3,):
        return -np.einsum(
            'nijk,kj->ni', g_tmp, dipole_tensor*np.eye(3)
        ).reshape(r.shape)
    elif dipole_tensor.shape==(3,3,):
        return -np.einsum(
            'nijk,kj->ni', g_tmp, dipole_tensor
        ).reshape(r.shape)
    elif dipole_tensor.shape==(r.shape+(3,)):
        return -np.einsum(
            'nijk,nkj->ni', g_tmp, dipole_tensor.reshape(-1, 3, 3)
        ).reshape(r.shape)
    else:
        raise ValueError('dipole tensor must be a 3d vector 3x3 matrix or Nx3x3 matrix')

def strain_field(r, dipole_tensor, poissons_ratio, shear_modulus, min_distance=0):
    g_tmp = ddG(r, poissons_ratio=poissons_ratio, shear_modulus=shear_modulus, min_distance=min_distance)
    if dipole_tensor.shape==(3,):
        v = -np.einsum(
            'nijkl,kl->nij', g_tmp, dipole_tensor*np.eye(3)
        )
    elif dipole_tensor.shape==(3,3,):
        v = -np.einsum(
            'nijkl,kl->nij', g_tmp, dipole_tensor
        )
    elif dipole_tensor.shape==(r.shape+(3,)):
        v = -np.einsum(
            'nijkl,nkl->nij', g_tmp, dipole_tensor.reshape(-1,3,3)
        )
    else:
        raise ValueError('dipole tensor must be a 3d vector 3x3 matrix or Nx3x3 matrix')
    v = 0.5*(v+np.einsum('nij->nji', v))
    return v.reshape(r.shape+(3,))

