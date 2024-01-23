import numpy as np

from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar, root
from scipy.integrate import cumtrapz
from scipy.signal import argrelextrema
from scipy.constants import physical_constants
KB = physical_constants['Boltzmann constant in eV/K'][0]

class GenerateBonds():
    """
    """
    
    def __init__(self, b_0, basis, long_disp_lims, t1_disps_lims, t2_disps_lims, n_shells=1, 
                 n_bins_long=100, n_bins_t1=50, n_bins_t2=50):
        self.b_0 = b_0
        self.basis = np.array(basis)
        self.long_disps = np.array(long_disps)
        self.t1_disps = np.array(t1_disps)
        self.t2_disps = np.array(t2_disps)
        self.n_bins_long = n_bins_long
        self.n_bins_t1 = n_bins_t1
        self.n_bins_t2 = n_bins_t2
        
class MeanField2():
    """
    !!! This class is currently in an experimental state and undergoes frequent updates. 
        Please be cautious when using it -rads !!!
    
    Performs the mean-field approximation on an FCC bond lattice. Provided a bonding potential (for ex. r_potential, or
    (r_potential + t1_potential + t2_potential), apply symmetry operations and correlations to generate an 'effective' bonding potential.
    The Boltzmann distribution of this effective bonding potential is the 'bond density' which can be used to obtain useful thermodynamic properties like anharmonic internal and free energies, and lattice expansion.
    
    Currently implemented only for the 1st nearest neighbor (NN) shell. For FCC, we have m=12 equilvalent NN bonds and l=[0, m].
    
    Paper reference: https://doi.org/10.1103/PhysRevB.102.100101

    Parameters:
    - b_0 (float): The bond length for the NN shell.
    - basis (list): The basis vectors corresponding to the NN shell.
    - rotations (list): The rotation matrices which transform the reference bond vector (l=0) to the bond vectors (k) in the same shell.
    - alphas (list): The correlation matrices \alpha_{pl} between different bond pairs.
    - r_potential (callable): Function for the ij (or 'radial' or 'central') potential (we take this to be the same as the longitudinal potential).
    - t1_potential (callable): Function for the t1 potential, if any. Defaults to None.
    - t2_potential (callable): Function for the t2 potential, if any. Defaults to None.
    - shells (int): Number of NN shells. Defaults to 1, just the 1NN shell.
    - energy_list (list/np.ndarray): Energies of the energy-volume curve. Defaults to None.
    - strain_list (list/np.ndarray): Strains corresponding to the volumes of the energy-volume curve. Defaults to None.
    """
    
    def __init__(self, b_0, basis, rotations, r_potentials, t1_potentials=None, t2_potentials=None, shells=1, 
                 alphas=None, energy_list=None, strain_list=None):
        self.b_0 = b_0
        self.basis = np.array(basis)
        self.rotations = np.array(rotations)
        self.alphas = alphas
        self.r_potentials = r_potentials
        self.t1_potentials = t1_potentials
        self.t2_potentials = t2_potentials
        self.shells = shells
        self.energy_list = energy_list
        self.strain_list = strain_list
        self._rho_1s = None
        self._dV1 = None

        self.check_inputs()

    def check_inputs(self):
        """
        Check if the inputs are correctly defined.
        """
        self.b_0 = np.array([self.b_0]) if isinstance(self.b_0, (int, float)) else np.array(self.b_0)

        for attr_name in ['r_potentials', 't1_potentials', 't2_potentials']:
            inp = getattr(self, attr_name)
            if not isinstance(inp, list):
                setattr(self, attr_name, [inp])
        
        inputs = [self.b_0, self.basis, self.rotations, self.r_potentials, self.t1_potentials, self.t2_potentials]
        inputs_str = ['b_0', 'basis', 'rotations', 'r_potentials', 't1_potentials', 't2_potentials']
        for inp, inp_str in zip(inputs, inputs_str):
            if len(inp) != self.shells:
                raise TypeError(f"Length of {inp_str} is not equal to shells.")
                
        if self.alphas is not None:
            if len(self.alphas) != self.shells:
                raise TypeError(f"Length of alphas is not equal to shells.")
        
    def get_meshes(self, bonds, shell=0):
        """
        Generate meshes based on the given bond vectors.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.
            
        Returns:
            long_mesh (ndarray): (n_long x n_t1 x n_t2) mesh generated using the longitudinal (long) basis vector.
            t1_mesh (ndarray): (n_long x n_t1 x n_t2) mesh generated using the 1st transversal (t1) basis vector.
            t2_mesh (ndarray): (n_long x n_t1 x n_t2) mesh generated using the 2nd transversal (t2) basis vector.
        """
        long_mesh = np.dot(bonds, self.basis[shell][0])
        t1_mesh = np.dot(bonds, self.basis[shell][1])
        t2_mesh = np.dot(bonds, self.basis[shell][2])
        return long_mesh, t1_mesh, t2_mesh
        
    def V1(self, bonds, rotation=np.eye(3), shell=0):
        """
        Calculate the bonding potential.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            rotation (3 x 3, optional): Rotation matrix. Defaults to the identity matrix.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.

        Returns:
            v1 (ndarray): The bonding potential.
        """
        r = np.linalg.norm(bonds, axis=-1)
        v1 = self.r_potentials[shell](r)
        if self.t1_potentials[shell] is not None:
            t1 = np.dot(bonds, self.basis[shell][1]@rotation)
            v1 += self.t1_potentials[shell](t1)
        if self.t2_potentials[shell] is not None:
            t2 = np.dot(bonds, self.basis[shell][2]@rotation)
            v1 += self.t2_potentials[shell](t2)
        return v1
        
    def dV1(self, bonds, meshes, shell=0):
        """
        Calculate the gradient of the bonding potential.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            meshes (ndarray): 3 (n_long x n_t1 x n_t2) meshgrids along the long, t1 and t2 directions.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.

        Returns:
            gradients (ndarray): Gradients of the bonding potential along the long, t1 and t2 directions.
        """
        V = self.V1(bonds=bonds, shell=shell)
        long_mesh, t1_mesh, t2_mesh = meshes
        gradients = np.gradient(V, long_mesh[:, 0, 0], t1_mesh[0, :, 0], t2_mesh[0, 0, :], edge_order=2)
        return gradients

    def Vmf_component(self, bonds, rotation, alpha=None, eps=1., shell=0):
        """
        Calculate the component 'l' of the mean-field effective potential: The bonds will be rotated according to the rotation provided and passed to self.V1 to generate the compoment.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            rotation (ndarray): Rotation matrix between the reference direction (l=0) and the desired direction (k) in the shell.
            alpha (3 x 3, optional): Correlation matrix, \alpha_{pl}. Defaults to None, to preserve backwards compatibility.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.

        Returns:
            vmf_component (ndarray): The 'l'th component of the mean-field effective potential.
        """
        # l = reference direction, k = desired direction
        b_l = bonds
        a_l = self.b_0[0]*eps*self.basis[0][0]
        a_k = self.b_0[shell]*eps*(self.basis[shell][0]@rotation)
        alpha = np.eye(3) if alpha is None else alpha  # set it to identity if alpha is None
        # compatability criteon, with alpha as the scaling matirx to db
        b_k = (b_l-a_l)@alpha+a_k
        return self.V1(b_k, rotation=rotation, shell=shell)

    def Vmf(self, bonds, eps=1.):
        """
        Calculate the mean-field effective potential, which includes all the 'm' components generated from the reference component (l=0) using the symmetry operations (rotations) between l=0 and l=[0, m].

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

        Returns:
            vmf (float): The complete mean-field effective potential which include the 'm' components.
        """
        vmf = 0
        for shell in range(self.shells):
            for rot in self.rotations[shell]:
                vmf += 0.5*self.Vmf_component(bonds=bonds, rotation=rot, alpha=None, eps=eps, shell=shell)
        return vmf

    def Vmfc(self, bonds, eps=1.):
        """
        Calculate the correlated mean-field effective potential. The anti-parallel bond (l=1) is highly correlated to the reference bond (l=0). We omit this bond for the model and consider the reference bond twice (factor 1 instead of (1/2)) to get the 'correlated' mean-field effective potential. This is mathematically consistent, as elaborated in the supplementary material of the referenced paper.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

        Returns:
            vmfc (float): The mean-field correlated effective potential.
        """
        vmfc = 0
        for shell in range(self.shells):
            if self.alphas is not None:
                for alpha in self.alphas[shell]:
                    for rot, al in zip(self.rotations[shell], alpha):
                        vmfc += 0.5*self.Vmf_component(bonds=bonds, rotation=rot, alpha=al, eps=eps, shell=shell)
            else:
                for i, rot in enumerate(self.rotations[shell]):
                    factor = 1. if ((i == 0) and (shell == 0)) else 0.5 if ((i != 1) or (shell != 0)) else 0.
                    vmfc += factor*self.Vmf_component(bonds, rotation=rot, alpha=None, eps=eps, shell=shell)  
        return vmfc
    
    def get_rho(self, bonds, long_mesh, temperature=100., eps=1., lm=0., Veff=None, shell=0):
        """
        Calculate the bond density from the effective bonding potential. 
        
        In order to maintain volume invariance, we include a Lagrange multiplier scalar term. This is fitted automatically in the 'nvt' and 'npt' routines (as get_rho is part of their objective functions).

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            long_mesh (ndarray): (n_long x n_t1 x n_t2) mesh generated using the longitudinal (long) basis vector.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            lm (float, optional): Lagrange multiplier. Defaults to 0.
            Veff (float, optional): The mean-field effective potential. If not provided, it is calculated internally.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.

        Returns:
            rho (ndarray): Normalized single bond density.
        """
        if Veff is None:
            Veff = self.Vmfc(bonds=bonds, eps=eps)
        Veff -= Veff.min()
        lm += 1e-10
        rho_1 = np.exp(-(Veff+lm*(long_mesh-self.b_0[shell]*eps))/KB/temperature)
        rho_1 /= rho_1.sum()
        num_offset = np.min(rho_1[rho_1 > 0])  # add a numerical offset to avoid divide by zero errors
        return rho_1+num_offset
    
    def Veff_fix(self, meshes, Veff, temperature=100., eps=1., shell=0):
        """
        Upon analysis of the MD bond density we observe 2 things:
        
        1. The t1 and t2 potentials are theoretically not equivalent to each other, as t1 is along the 1NN direction and t2 along the 2NN direction. This inequivalence is included in the mean-field effective potential. However, we find from the effective potential generated from the MD bond density that for the 'effective' potential, t1 and t2 are equivalent.
        
        2. The 'attractive' components of the effective potential in the transversal and longitudinal directions closely resemble each other. The 'repulsive' components in the transversal directions mirror their attractive components (since the effective potential is harmonic in the transversal directions).
        
        3. It is also the case that the 'attractive' longitudinal component of the mean-field effective potetnial develops an inflection point as it scales with temperature.
        
        To address these observations, we apply a fix to the mean-field effective potential.
        
        The fix does the following to the effective potential with the volume constraint (Lagrange multiplier term) pre-imposed. 
        
        The effective potential is projected to the longitudinal direction.
        The gradient of this projection, up to its inflection point is fit to a CubicSpline. 
        Beyond this point, the effective potential increases linearly.
        The transversal projections are set to the attractive-part (db > 0) of the truncated longitudinal projection, by fitting to a 2nd order polynomial.
        A 'fixed' effective potential is generated.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            meshes (ndarray): 3 (n_long x n_t1 x n_t2) meshgrids along the long, t1 and t2 directions.
            Veff (ndarray): The unfixed mean-field effective potential.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.

        Returns:
            new_Veff (ndarray): 'Fixed' effective potential.
        """
        long_mesh, t1_mesh, t2_mesh = meshes
        # mid-point of meshes along t1 and t2. Exception, if it starts from 0
        mp_t1 = int(t1_mesh.shape[2]/2) if (t1_mesh[0, :, 0][0] < 0.) else 0
        mp_t2 = int(t2_mesh.shape[2]/2) if (t2_mesh[0, 0, :][0] < 0.) else 0

        # determine the inflection point (ip) of the gradient of the effective potential...
        if self.alphas is not None:
            ip = long_mesh[:, mp_t1, mp_t2][-1]
        else:
            long_grad = np.gradient(Veff[:, mp_t1, mp_t2], long_mesh[:, mp_t1, mp_t2], edge_order=2)
            long_grad_fit = CubicSpline(long_mesh[:, mp_t1, mp_t2], long_grad)
            fine_long = np.linspace(long_mesh[:, mp_t1, mp_t2][0], long_mesh[:, mp_t1, mp_t2][-1], 10000)
            ip = fine_long[argrelextrema(long_grad_fit(fine_long), np.greater, order=1)[0][0]]

        # ...and set a cutoff for db
        db = long_mesh[:, mp_t1, mp_t2]-self.b_0[shell]*eps
        cutoff = ip-self.b_0[shell]*eps
        cutoff *= -1 if (cutoff < 0) else 1

        # fit new long potential truncated at the cutoff 
        sel_1 = (db <= cutoff)
        # for boundary conditions of the CubicSpline
        trunc_grad = np.gradient(Veff[:, mp_t1, mp_t2][sel_1], long_mesh[:, mp_t1, mp_t2][sel_1], edge_order=2)
        long_fit = CubicSpline(long_mesh[:, mp_t1, mp_t2][sel_1], Veff[:, mp_t1, mp_t2][sel_1], 
                               bc_type=((1, trunc_grad[0]), (1, trunc_grad[-1])))

        # fit new t1 and t2 potentials, which are quadratic, and set them equal to each other
        sel_2 = ((db >= 0.) & (db <= cutoff))
        t_db = np.concatenate((-np.flip(db[sel_2]), db[sel_2]))
        t_pot = np.concatenate((np.flip(Veff[:, mp_t1, mp_t2][sel_2]), Veff[:, mp_t1, mp_t2][sel_2]))
        t2_fit = np.poly1d(np.polyfit(t_db, t_pot, deg=2))
        t1_fit = t2_fit

        # generate new Veff
        new_Veff = long_fit(long_mesh)+t1_fit(t1_mesh)+t2_fit(t2_mesh)
            
        return new_Veff
    
    def get_epsilon_pressure(self, eps=1., shell=0):
        """
        From the energy_list and strain_list inputs, determine the pressure and energy offsets from the mean-field model.
        
        Parameters:
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.
            
        Returns:
            P_offset (float/array): Pressure offsets.
            E_offset (float/array): Energy offsets.
        """
        if (np.any(self.energy_list) == None) and (np.any(self.strain_list) == None):
            if isinstance(eps, (int, float)):
                return 0., 0.
            else:
                return np.array([0.]*len(eps)), np.array([0.]*len(eps))
        else:
            # from static calculations
            strains = 1+self.strain_list
            u_md_fit_eqn = np.poly1d(np.polyfit(strains, self.energy_list, deg=4))
            du_md_fit = u_md_fit_eqn.deriv(m=1)(eps)
            # from mean-field model
            u_mf = 6.*self.r_potentials[shell](strains*self.b_0[shell])
            u_mf_fit_eqn = np.poly1d(np.polyfit(strains, u_mf, deg=4))
            du_mf_fit = u_mf_fit_eqn.deriv(m=1)(eps)
            P_offset = -4./(3.*eps**2*(np.sqrt(2)*self.b_0[shell])**3)*(du_md_fit-du_mf_fit)
            E_offset = (u_md_fit_eqn(eps)-u_mf_fit_eqn(eps))
            return P_offset, E_offset

    def find_virial_quant(self, bonds, meshes, temperature=100., eps=1., lm=0., fix_Veff=True, Veff=None, return_rho_1=False, shell=0):
        """
        Calculate virial temperature, pressure and also the equilibrium bond at the given strain.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            meshes (ndarray): 3 (n_long x n_t1 x n_t2) meshgrids along the long, t1 and t2 directions.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            lm (float, optional): Lagrange multiplier. Defaults to 0.
            fix_Veff (Boolean): Mean-field effective potential fix. Defaults to True.
            Veff (float, optional): The mean-field effective potential. If not provided, it is calculated internally.
            return_rho_1 (boolean): Whether or not to return the bond density.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.

        Returns:
            T_vir (float): Virial temperature of the mean-field model.
            P_vir (float): Virial pressure of the mean-field model.
            b_eps (float): Equilibrium bond at the given strain.
        """
        long_mesh, t1_mesh, t2_mesh = meshes
        rho_1 = self.get_rho(bonds=bonds, long_mesh=long_mesh, temperature=temperature, eps=eps, lm=lm, Veff=Veff, shell=shell)
        if fix_Veff:
            Veff_w_lm = -KB*temperature*np.log(rho_1)
            fixed_Veff = self.Veff_fix(meshes=meshes, temperature=temperature, eps=eps, Veff=Veff_w_lm, shell=shell)
            rho_1 = self.get_rho(bonds=bonds, long_mesh=long_mesh, temperature=temperature, Veff=fixed_Veff, eps=eps, lm=0., shell=shell)
        b_eps = (long_mesh*rho_1).sum()
        dV = self._dV1
        db_dV = (long_mesh-b_eps)*dV[0]+t1_mesh*dV[1]+t2_mesh*dV[2]
        a_dV = b_eps*dV[0]+(t1_mesh*rho_1).sum()*dV[1]+(t2_mesh*rho_1)*dV[2]
        N_by_V = 4./(b_eps*np.sqrt(2))**3 
        T_vir = 2./KB*(db_dV*rho_1).sum()
        P_vir = -2.*N_by_V*(a_dV*rho_1).sum()+self.get_epsilon_pressure(eps=eps)[0]
        if return_rho_1:
            return T_vir, P_vir, b_eps, rho_1
        return T_vir, P_vir, b_eps
    
    def run_nvt(self, bonds, temperature=100., eps=1., lm=0., fix_Veff=True, fix_T=False, shell=0):
        """
        Consider the NVT case. If strain (eps) != 1., then the structure is strained, and the calculation is done at this 'strained' bond.
        The optimization converges the virial temperature to the target temperature and bond length (b_0*eps) to the target bond length.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            lm (float, optional): Lagrange multiplier. Defaults to 0.
            fix_Veff (Boolean): Mean-field effective potential fix. Defaults to True.
            fix_T (Boolean): If True, normlaize the mean-field model temperature to the input temperature. Defaults to True.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.
            
        Returns:
            T_vir (float): Virial temperature of the mean-field model.
            P_vir (float): Virial pressure of the mean-field model.
            b_eps (float): Equilibrium bond at the given strain.
            T_eff (float): Re-normalized temperature.
            eps (float, optional): Strain on bond b_0.
            lm (float): Lagrange multiplier.
        """
        Veff = self.Vmfc(bonds=bonds, eps=eps)
        meshes = self.get_meshes(bonds, shell=shell)
        self._dV1 = self.dV1(bonds=bonds, meshes=meshes, shell=shell)
        if fix_T:
            def virial(args):
                T_vir, _, b_eps = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=args[0], eps=eps, lm=args[1], 
                                                         fix_Veff=fix_Veff, Veff=Veff, shell=shell)
                return [abs(temperature-T_vir), np.abs(self.b_0[shell]*eps-b_eps)]
            solver = root(virial, x0=(temperature, lm), method='lm', tol=1e-20)
            eff_temp, lm = solver.x
        else:
            def virial(args):
                _, _, b_eps = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=temperature, eps=eps, lm=args, 
                                                     fix_Veff=fix_Veff, Veff=Veff)
                return np.abs(self.b_0[shell]*eps-b_eps)
            solver = root_scalar(virial, x0=lm, x1=lm+0.001, rtol=1e-20)
            eff_temp, lm = temperature, solver.root
        virial_q = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=eff_temp, eps=eps, lm=lm, fix_Veff=fix_Veff, 
                                          Veff=Veff, return_rho_1=True, shell=shell)
        print('T: {}\nT_eff: {}\nT_vir: {}\nP_vir: {}\neps: {}\nlm: {}\n'.format(temperature, eff_temp, *virial_q[:2], eps, lm))
        return *virial_q, eff_temp, eps, lm
    
    def run_npt(self, bonds, temperature=100., pressure=1e-4, eps=1., lm=0., fix_Veff=True, fix_T=False, shell=0):
        """
        Consider the NPT case. The optimization converges the virial temperature to the target temperature, the virial pressure to the target pressure and the bond length (b_0*eps) to the target bond length. The converged value of 'eps' is then the strain at the target temperture and pressure. 

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            temperature (float, optional): Temperature value. Defaults to 100.
            pressure (float, optional): Pressure value. Defaults to 1e-4 (1 atm).
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            lm (float, optional): Lagrange multiplier. Defaults to 0.
            fix_Veff (Boolean): Mean-field effective potential fix. Defaults to True.
            fix_T (Boolean): If True, normlaize the mean-field model temperature to the input temperature. Defaults to True.
            shell (int): The nearest neighbor shell under consideration. Defaults to 0, the 1NN shell.
            
        Returns:
            T_vir (float): Virial temperature of the mean-field model.
            P_vir (float): Virial pressure of the mean-field model.
            b_eps (float): Equilibrium bond at the given strain.
            T_eff (float): Re-normalized temperature.
            eps (float, optional): Strain on bond b_0.
            lm (float): Lagrange multiplier.
        """
        meshes = self.get_meshes(bonds, shell=shell)
        self._dV1 = self.dV1(bonds=bonds, meshes=meshes, shell=shell)
        if fix_T:
            def virial(args):
                T_vir, P_vir, b_eps = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=args[0], eps=args[1], 
                                                             lm=args[2], fix_Veff=fix_Veff, shell=shell)
                return [abs(temperature-T_vir), (pressure-P_vir), np.abs(self.b_0[shell]*args[1]-b_eps)]
            solver = root(virial, x0=(temperature, eps, lm), method='lm', tol=1e-5)
            eff_temp, eps, lm = solver.x
        else:
            def virial(args):
                _, P_vir, b_eps = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=temperature, eps=args[0], 
                                                         lm=args[1], fix_Veff=fix_Veff, shell=shell)
                return [abs(pressure-P_vir), np.abs(self.b_0[shell]*args[0]-b_eps)]
            solver = root(virial, x0=(eps, lm), method='lm', tol=1e-20)
            eff_temp, eps, lm = temperature, solver.x[0], solver.x[1]
        virial_q = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=eff_temp, eps=eps, lm=lm, fix_Veff=fix_Veff, return_rho_1=True,
                                          shell=shell)
        print('T: {}\nT_eff: {}\nT_vir: {}\nP_vir: {}\neps: {}\nlm: {}\n'.format(temperature, eff_temp, *virial_q[:2], eps, lm))
        return *virial_q, eff_temp, eps, lm
    
    @staticmethod
    def _validate(bonds, temperatures, pressures, epsilons):
        """
        Helper function to make sure inputs to run_nve and run_npt are correct.
        """
        if isinstance(temperatures, (int, float)):
            temperatures = [temperatures]
            bonds = [bonds]
        if (isinstance(pressures, (int, float))) or (pressures is None):
            pressures = [pressures]
            if len(pressures) != len(temperatures):
                pressures *= len(temperatures)
        if isinstance(epsilons, (int, float)):
            epsilons = [epsilons]
            if len(epsilons) != len(temperatures):
                epsilons *= len(temperatures)
        return bonds, temperatures, pressures, epsilons

    def get_ah_U(self, bonds, temperatures, pressures=None, epsilons=None, fix_Veff=True, fix_T=False):
        """
        Collect the mean-filed approximated properties for a given set of temperatures. If pressures is not None, then
        the NPT case is considered. Ohterwise, considers the NVT case (at the strain (epsilons) value or at no 
        strain if epsilons=None). 

        Parameters:
            bonds (ndarray): Array of bond vectors.
            temperatures (list or float): List of temperatures or a single temperature value.
            pressures (list or float): List of pressures corresponding to the temperatures or a single pressure value. Defaults to None.
            epsilons (list or float): List of strains corresponding to the temperatures or a single strain value. Defaults to None.
            fix_Veff (Boolean): Mean-field effective potential fix. Defaults to True.
            fix_T (Boolean): If True, normlaize the mean-field model temperature to the input temperature. Defaults to True.

        Returns:
            dict: Dictionary containing the properties: 
                  'ah_U': Anharmonic internal energy,
                  'T_vir': Virial temperature,
                  'T_eff': Re-normalized temperature,
                  'P_vir': Virial pressure,
                  'epsilon': Strain,
                  'lattice': Lattice constant,
                  'lm': Lagrange multiplier.
        """
        ensemble = 'npt' if pressures is not None else 'nvt'
        epsilons = 1. if epsilons is None else epsilons
        
        # validate
        bonds, temperatures, pressures, epsilons = self._validate(bonds=bonds, temperatures=temperatures, pressures=pressures,
                                                                  epsilons=epsilons)
        
        # do this little bit to set the eps and lm values from the previous temperature as the starting point for the next one
        outs = []
        epsi, lm = epsilons[0], 0.
        for i, temp in enumerate(temperatures):
            if ensemble == 'nvt':
                epsi = epsilons[i]
            out = self.run_ensemble(bonds=bonds[i], temperature=temp, pressure=pressures[i], eps=epsi, lm=lm, 
                                    ensemble=ensemble, fix_Veff=fix_Veff, fix_T=fix_T)
            epsi, lm = out[5], out[6]
            outs.append(out)
        T_virs, P_virs, b_eps, rho_1s, T_effs, eps, lms = zip(*outs)
    
        # save for visualization
        self._rho_1s = rho_1s
        
        # energy_volume_offsets
        _, energy_offsets = self.get_epsilon_pressure(eps=np.array(eps))

        # collect anharmonic internal energy in meV/atom
        ah_U = []
        for i, temp in enumerate(T_virs):
            per_bond_energy = (self.V1(bonds=bonds[i])*rho_1s[i]).sum()
            per_atom_energy = 6*per_bond_energy
            ah_U.append((per_atom_energy+energy_offsets[i]-1.5*KB*temp)*1000)

        return {
            'ah_U': np.array(ah_U),
            'T_vir': np.array(T_virs),
            'T_eff': np.array(T_effs),
            'P_vir': np.array(P_virs),
            'epsilon': np.array(eps),
            'lattice': np.array(eps)*self.b_0[0]*np.sqrt(2),
            'lm': np.array(lms)
        }

    def run_ensemble(self, bonds, temperature, pressure, eps, lm, ensemble, fix_Veff, fix_T):
        """
        Run the appropriate ensemble simulation based on the ensemble type.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            temperature (float): Target temperature.
            pressure (float): Target pressure.
            eps (float): Strain on bond b_0.
            lm (float): Lagrange multiplier.
            ensemble (str): Ensemble type ('nvt' or 'npt').
            fix_Veff (Boolean): Mean-field effective potential fix. Defaults to True.
            fix_T (Boolean): If True, normlaize the mean-field model temperature to the input temperature. Defaults to True.

        Returns:
            tuple: Tuple containing the properties: rho_1, epsilon, lm.
        """
        if ensemble == 'nvt':
            return self.run_nvt(bonds=bonds, temperature=temperature, eps=eps, lm=lm, fix_Veff=fix_Veff, fix_T=fix_T)
        elif ensemble == 'npt':
            return self.run_npt(bonds=bonds, temperature=temperature, pressure=pressure, eps=eps, lm=lm, fix_Veff=fix_Veff, fix_T=fix_T)

    @staticmethod
    def get_ah_F(ah_U, temperatures, n_fine_samples=10000):
        """
        Interpolate and integrate the anharmonic internal energy to obtain the anharmonic free energy.

        Parameters:
            ah_U (ndarray): Anharmonic internal energies (ah_U).
            temperatures (ndarray): Array of temperatures.
            n_fine (int, optional): Number of points for fine temperature grid. Defaults to 10000.

        Returns:
            tuple: Tuple containing the fine temperatures and the integrated anharmonic free energy.
        """
        fine_temperatures = np.linspace(temperatures[0], temperatures[-1], n_fine_samples, endpoint=True)
        ah_U_eqn = CubicSpline(x=temperatures, y=ah_U)
        return fine_temperatures[1:], cumtrapz(ah_U_eqn(fine_temperatures), 1/fine_temperatures)*fine_temperatures[1:]