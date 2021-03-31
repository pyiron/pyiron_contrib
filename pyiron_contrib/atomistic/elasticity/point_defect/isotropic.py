import numpy as np

def displacement_field(
    x, dipole_tensor, lame_coefficient, shear_modulus
):
    """
    displacement field according to the isotropic linear point defect theory.
    """
    r = np.atleast_2d(x)
    R = np.linalg.norm(r, axis=-1)[:,None,None]
    p = [
        (lame_coefficient+3*shear_modulus)/(lame_coefficient+2*shear_modulus),
        (lame_coefficient+shear_modulus)/(lame_coefficient+2*shear_modulus)
    ]
    G = p[1]*np.eye(3)+p[2]*np.einsum('ni,nj->nij', r, r)
    u = np.eye(3)[None,:,:]-r[:,:,None]*r[:,None,:]/R**2
    R = R[:,:,:,None]
    u = p[1]*np.einsum('nik,njk->nijk', u, u)/R**2
    u += np.einsum('nk,nijk,nij->nijk', r, 1/R**3, G)
    u /= 8*np.pi*shear_modulus*R
    u = -np.einsum('nijk,jk->ni', u, dipole_tensor)
    return np.squeeze(u)
