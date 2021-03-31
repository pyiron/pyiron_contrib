import numpy as np

def G(r, poissons_ratio, shear_modulus):
    """
    Green's function
    """
    r = np.array(r).reshape(-1, 3)
    R = np.linalg.norm(r, axis=-1)
    vorfaktor = 1/(16*shear_modulus*np.pi*(1-poissons_ratio))
    return vorfaktor*np.squeeze(
        ((3-4*poissons_ratio)*np.eye(3)+np.einsum('ni,nj,n->nij', r, r, 1/R**2))/R
    )

def dG(x, poissons_ratio, shear_modulus):
    """
    first derivative of the Green's function
    """
    E = np.eye(3)
    r = np.array(x).reshape(-1, 3)
    R = np.linalg.norm(r, axis=-1)
    r = np.einsum('ni,n->ni', r, 1/R)
    B = 1/(16*np.pi*shear_modulus*(1-poissons_ratio))
    A = (3-4*poissons_ratio)*B
    v = -A*np.einsum('ik,nj->nijk', E, r)
    v += B*np.einsum('ij,nk->nijk', E, r)
    v += B*np.einsum('jk,ni->nijk', E, r)
    v -= 3*B*np.einsum('ni,nj,nk->nijk', r, r, r)
    return np.einsum('nijk,n->nijk', v, 1/R**2)

def ddG(x, poissons_ratio, shear_modulus):
    """
    Second derivative of the Green's function
    """
    E = np.eye(3)
    r = np.array(x).reshape(-1, 3)
    R = np.linalg.norm(r, axis=-1)
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
    return np.einsum('nijkl,n->nijkl', v, 1/R**3)

def displacement_field(r, dipole_tensor, poissons_ratio, shear_modulus):
    return -np.einsum('nijk,kj->ni', dG(r, poissons_ratio=poissons_ratio, shear_modulus=shear_modulus), dipole_tensor).squeeze()

def strain_field(r, dipole_tensor, poissons_ratio, shear_modulus):
    return -np.einsum('nijkl,kl->nij', ddG(r, poissons_ratio=poissons_ratio, shear_modulus=shear_modulus), dipole_tensor).squeeze()
