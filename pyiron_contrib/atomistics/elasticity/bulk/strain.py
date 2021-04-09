import numpy as np

def get_strain(structure, ref_structure, num_neighbors=None):
    if num_neighbors is None:
        cna = ref_structure.analyse.pyscal_cna_adaptive()
        bulk = np.asarray([k for k in cna.keys()])[np.argmax([v for v in cna.values()])]
        if bulk=='bcc':
            num_neighbors = 8
        elif bulk=='fcc' or bulk=='hcp':
            num_neighbors = 12
        else:
            raise ValueError('Crystal structure not recognized')
    ref_frame = ref_structure.get_neighbors(num_neighbors=num_neighbors).vecs[0]
    neigh = structure.get_neighbors(num_neighbors=num_neighbors)
    indices = np.argmin(np.linalg.norm(neigh.vecs[:,:,None,:]-ref_frame[None,None,:,:], axis=-1), axis=-1)
    D = np.einsum('ij,ik->jk', ref_frame, ref_frame)
    D = np.linalg.inv(D)
    J = np.einsum('nij,nik->njk', ref_frame[indices], neigh.vecs)
    J = np.einsum('ij,njk->nik', D, J)
    return 0.5*(np.einsum('nij,nkj->nik', J, J)-np.eye(3))

