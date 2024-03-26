import numpy as np
import os

from scipy.optimize import root, root_scalar
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
from scipy.constants import physical_constants
KB = physical_constants['Boltzmann constant in eV/K'][0]

from BondModeFunctions import DebyeModel

def get_meshes(bond_grid, basis):
    """
    Generate longitudinal and 2 transversal meshes from the bond grid.

    Parameters:
        bond_grid (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates.
        basis (list/np.ndarray): 3 x 3 array of basis vectors.
        
    Returns:
        lo_mesh (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) mesh generated along the longitudinal direction.
        t1_mesh (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) mesh generated along the first transversal direction.
        t2_mesh (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) mesh generated along the second transversal direction.
    """
    lo_mesh = np.dot(bond_grid, basis[0])
    t1_mesh = np.dot(bond_grid, basis[1])
    t2_mesh = np.dot(bond_grid, basis[2])
    return lo_mesh, t1_mesh, t2_mesh

def get_anharmonic_F(anharmonic_U, temperatures, n_fine_samples=10000):
    """
    Interpolate and integrate the anharmonic internal energies (U) over inverse temperature to obtain the anharmonic free energy (F).

    Parameters:
        anharmonic_U (list/np.ndarray): Anharmonic internal energies.
        temperatures (list/np.ndarray): Array of temperatures.
        n_fine (int, optional): Number of points for the fine temperature interpolation. Defaults to 10000.

    Returns:
        tuple: Tuple containing the fine temperatures and the integrated anharmonic free energy.
    """
    fine_temperatures = np.linspace(temperatures[0], temperatures[-1], n_fine_samples, endpoint=True)
    ah_U_eqn = CubicSpline(x=temperatures, y=anharmonic_U)
    return fine_temperatures[1:], cumtrapz(ah_U_eqn(fine_temperatures), 1/fine_temperatures)*fine_temperatures[1:]

class GenerateBonds():
    """
    Class that takes a reference bond length b_0, basis and displacements along the longitudinal and 2 transversal directions and generates a grid of bond vectors in Cartesian xyz coordinates.
    The shape of the bond grid is (len(lo_disps), len(t1_disps), len(t2_disps), 3).

    Args:
        b_0 (float): Reference bond length.
        basis (list/np.ndarray): 3 x 3 array of basis vectors.
        lo_disps (list/np.ndarray): 1D array of displacements along the longitudinal direction.
        t1_disps (list/np.ndarray): 1D array of displacements along the first transversal direction.
        t2_disps (list/np.ndarray): 1D array of displacements along the second transversal direction.

    Returns:
        bond_grid (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates.
    """
    
    def __init__(self, b_0, basis, lo_disps, t1_disps, t2_disps):
        self.b_0 = b_0
        self.basis = np.array(basis)
        self.lo_disps = np.array(lo_disps)
        self.t1_disps = np.array(t1_disps)
        self.t2_disps = np.array(t2_disps)

        self.check_inputs()

    def check_inputs(self):
        assert isinstance(self.b_0, float), 'b_0 must be a float.'
        assert self.basis.shape == (3, 3), 'basis must be 3 x 3.'
        assert self.lo_disps.ndim == 1, 'lo_disps must be 1D.'
        assert self.t1_disps.ndim == 1, 't1_disps must be 1D.'
        assert self.t2_disps.ndim == 1, 't2_disps must be 1D.'
        
        self.basis = self.basis/np.linalg.norm(self.basis, axis=-1, keepdims=True)
    
    def get_meshes(self):
        lo_mesh, t1_mesh, t2_mesh = np.meshgrid(self.lo_disps, self.t1_disps, self.t2_disps, indexing='ij')
        return lo_mesh, t1_mesh, t2_mesh

    def get_bond_grid(self):
        lo_mesh, t1_mesh, t2_mesh = self.get_meshes()
        bond_grid = np.tensordot(lo_mesh, self.basis[0], axes=0)+np.tensordot(t1_mesh, self.basis[1], axes=0)+np.tensordot(t2_mesh, self.basis[2], axes=0)
        return bond_grid
    
class MeanFieldModel():
    """
    Class that computes the mean-field effective potential.

    Args:
        ref_shell (int): Reference shell with respect to which the mean-field model is defined.
        shells (int): Number of shells up to which the input b_0s, potentials, rotations, alphas are specified.
        l_order (int): Number of shells to consider as neighbors to build the mean-field model. Defaults to 1, the 1st nearest neighbor shell.
        bond_grid (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates. Must be defined with respect to the reference shell.
        b_0s (list/np.array): The equilibrium bond length of each shell. Must be of length shells.
        basis (list/np.ndarray): (shells x 3 x 3) array of unit vectors for each shell. Must be of length shells.
        rotations (list/np.ndarray): Rotation matrices which transform the reference bond vector (l=0) to other bond vectors (k) in the same shell, for each shell. Must be of length shells.
        alphas (list/np.ndarray): Correlation matrices \alpha_{pl} between different bond pairs, defined for each shell with respect to a reference bond in that shell. Must be of length shells.
        
        r_potentials (list of functions): Function for the ij (or 'radial' or 'central') potential (we take this to be the same as the longitudinal (lo) potential). Must be of length shells.
        t1_potentials (list of functions, optional): Function for the t1 potential, if any. Defaults to None. Must be of length shells.
        t2_potentials (list of functions, optional): Function for the t2 potential, if any. Defaults to None.  Must be of length shells.
    """

    def __init__(self, bond_grid, b_0s, basis, rotations, alphas, alphas_rot_ids, r_potentials, t1_potentials=None, t2_potentials=None, ref_shell=0, shells=1, l_order=1):
        self.bond_grid = bond_grid
        self.b_0s = b_0s
        self.basis = basis
        self.rotations = rotations
        self.alphas = alphas
        self.alphas_rot_ids = alphas_rot_ids
        self.r_potentials = r_potentials
        self.t1_potentials = t1_potentials
        self.t2_potentials = t2_potentials
        self.shells = shells
        self.ref_shell = ref_shell
        self.l_order = l_order

        self.meshes = get_meshes(self.bond_grid, self.basis[ref_shell])
        self.check_inputs()
        self.initialize_db()

    def check_inputs(self):
        """
        Check if the inputs are properly defined.
        """
        assert isinstance(self.shells, int), 'shells must be an integer.'
        assert self.shells > 0, 'shells must be greater than 0.'
        assert isinstance(self.l_order, int), 'l_order must be an integer.'
        assert self.l_order <= self.shells, 'l_order must be lesser than shells.'
        assert isinstance(self.ref_shell, int), 'ref_shell must be an integer.'
        assert self.ref_shell <= self.shells, 'ref_shell must be less than shells.'
              
        # enusre that these inputs have the same lengths
        inputs = [self.b_0s, self.basis, self.rotations, self.r_potentials, self.t1_potentials, self.t2_potentials]
        inputs_str = ['b_0s, basis', 'rotations', 'r_potentials', 't1_potentials', 't2_potentials']
        for inp, inp_str in zip(inputs, inputs_str):
            if len(inp) != self.shells:
                raise TypeError(f'length of {inp_str} is not equal to shells.')
        #assert len(self.alphas) == self.l_order, 'length of alphas is not equal to l_order.'
        #assert len(self.alphas_rot_ids) == self.l_order, 'length of alphas is not equal to l_order.'

    def V1(self, bond_grid=None, shell=None):
        """
        Calculate the bonding potential.

        Parameters:
            bond_grid (np.ndarray, optional): Modified bond grid. Defaults to None, uses self.bond_grid.
            shell (int, optional): Shell for which the bonding potential is calculated. Defaults to None, the reference shell.
            rotation (3 x 3, optional): Rotation matrix. Defaults to the identity matrix.

        Returns:
            v1 (np.ndarray): The bonding potential.
        """
        bond_grid = self.bond_grid if bond_grid is None else bond_grid
        shell = self.ref_shell if shell is None else shell
        v1 = 0.
        r = np.linalg.norm(bond_grid, axis=-1)
        v1 = self.r_potentials[shell](r)
        if (self.t1_potentials[shell] is not None): # and (shell == self.ref_shell):
            t1 = np.dot(bond_grid, self.basis[shell][1])
            v1 += self.t1_potentials[shell](t1)
        if (self.t2_potentials[shell] is not None): # and (shell == self.ref_shell):
            t2 = np.dot(bond_grid, self.basis[shell][2])
            v1 += self.t2_potentials[shell](t2)
        return v1

    def V1_gradient(self):
        """
        Calculate the gradient of the bonding potential for the reference shell.

        Returns:
            dv1 (3, np.ndarray): Gradients of the bonding potential along the long, t1, and t2 directions.
        """
        V = self.V1()
        lo_mesh, t1_mesh, t2_mesh = self.meshes
        gradient = np.gradient(V, lo_mesh[:, 0, 0], t1_mesh[0, :, 0], t2_mesh[0, 0, :], edge_order=2)
        return np.array(gradient)
    
    def initialize_db(self):
        """
        Initialize the displacement from equilibrium bond length for each shell.
        """
        print('Initializing {}NN db...'.format(self.ref_shell+1))
        self._db_s = []
        self._a_k_s = []
        for shell in range(self.l_order):
            # reference db for the shell
            b_l = self.bond_grid
            a_l = self.b_0s[self.ref_shell]*self.basis[self.ref_shell][0]
            self._a_k_s.append(self.b_0s[shell]*self.basis[shell][0])
            self._db_s.append(b_l-a_l)

    def Veff(self, eps=1.):
        """
        Calculate the correlated mean-field effective potential.

        Parameters:
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

        Returns:
            vmfc (np.ndarray): The correlated mean-field effective potential.
        """
        vmfc = 0.
        for shell in range(self.l_order):
            # here, alpha scales the displacement from equilibrium, i.e. db = (b_l-a_l). 
            for alphas, alphas_rot_ids in zip(self.alphas[shell], self.alphas_rot_ids[shell]):
                # we scale first, rotate it, and then add the new NN a_k
                # Note that the rotation is done on db and not a_k! This lets us circumvent the rotation that needs to be made for the transveral components in V1.
                # for future optimization.
                for alpha, rot in zip(alphas, self.rotations[shell][alphas_rot_ids]):
                    b_k_s = self._db_s[shell]@alpha@rot.T+self._a_k_s[shell]
                    vmfc += 0.5*self.V1(bond_grid=b_k_s*eps, shell=shell)
        return vmfc
    
class VirialQuantities():
    """
    Class that calculates the virial quantities.

    Args:
        bond_grid (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates defined with respect to a shell.
        b_0 (list/np.array): The equilibrium bond length of the shell.
        basis (list/np.ndarray): (3 x 3) array of unit vectors of the shell.
        crystal (str): Crystal structure. Defaults to 'fcc'.
    """

    def __init__(self, bond_grid, b_0, basis, crystal='fcc'):
        self.bond_grid = bond_grid
        self.b_0 = b_0
        self.basis = basis
        self.crystal = crystal

        self.meshes = get_meshes(self.bond_grid, self.basis)
        self._rho_1_temp = None

        if self.crystal == 'fcc':
            # set the number of bonds per shell.
            self._n_bonds_per_shell = np.array([12, 6, 24, 12, 24, 8])
            # factors to obtain the correct volume from the bond length of each shell
            self._factors = np.array([np.sqrt(2), 1., np.sqrt(2/3), np.sqrt(1/2), np.sqrt(2/5), np.sqrt(1/3)])
        elif self.crystal == 'bcc':
            self._n_bonds_per_shell = np.array([8, 6, 12, 24, 8, 6, 24, 24])
            self._factors = np.array([2/np.sqrt(3), 1., 1/np.sqrt(2), 2/np.sqrt(11), 1/np.sqrt(3), 1/2, 2/np.sqrt(19), 1/np.sqrt(5)])
        else:
            raise TypeError('crystal must be fcc or bcc.')


    def get_rho(self, Veff, temperature=100., eps=1., lm=0.):
        """
        Calculate the bond density from the effective bonding potential. 
        
        In order to maintain volume invariance, we include a Lagrange multiplier scalar term. This is fitted automatically in the 'nvt' and 'npt' routines (as get_rho is part of their objective functions).

        Parameters:
            Veff (float): The correlated mean-field effective potential.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            lm (float, optional): Lagrange multiplier. Defaults to 0.

        Returns:
            rho_1 (np.ndarray): Normalized single bond density.
        """
        Veff -= Veff.min()
        lm += 1e-20
        rho_1 = np.exp(-(Veff+lm*(self.meshes[0]-self.b_0*eps))/KB/temperature)
        rho_1 /= rho_1.sum()
        if not np.isnan(rho_1).any():
            self._rho_1_temp = rho_1
        else:
            rho_1 = self._rho_1_temp+np.min(self._rho_1_temp[self._rho_1_temp>0])
        num_offset = np.min(rho_1[rho_1>0])  # add a numerical offset to avoid divide by zero errors
        return rho_1+num_offset

    def get_virial_quantities(self, Veff, dV1, temperature=100., eps=1., lm=0., shell=0, return_rho_1=False):
        """
        Calculate virial temperature, pressure, and equilibrium bond at the given strain.

        Parameters:
            Veff (float): The correlated mean-field effective potential.
            dV1 (3, np.ndarray): The gradient of the single bonding potential along the longitudinal and 2 transversal directions.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            lm (float, optional): Lagrange multiplier. Defaults to 0.
            shell (int, optional): Shell with respect to which the volume will be calculated. Defaults to 0, the 1st shell.
            return_rho_1 (boolean): Whether or not to return the bond density.

        Returns:
            T_vir (float): Virial temperature of the mean-field model.
            P_vir (float): Virial pressure of the mean-field model.
            b_eps (float): Equilibrium bond at the given strain.
            rho_1 (ndarray): Bond density.
        """
        lo_mesh, t1_mesh, t2_mesh = self.meshes
        rho_1 = self.get_rho(Veff=Veff, temperature=temperature, eps=eps, lm=lm)
        b_eps = (lo_mesh*rho_1).sum()
        db_dV1 = (lo_mesh-b_eps)*dV1[0]+t1_mesh*dV1[1]+t2_mesh*dV1[2]
        T_vir = self._n_bonds_per_shell[shell]/6/KB*(db_dV1*rho_1).sum()
        
        if self.crystal == 'fcc':
            N_by_V = 4/(b_eps*self._factors[shell])**3
        elif self.crystal == 'bcc':
            N_by_V = 2/(b_eps*self._factors[shell])**3
        else:
            raise TypeError('crystal must be fcc or bcc.')
        a_dV1 = b_eps*dV1[0]
        P_vir = -N_by_V*(a_dV1*rho_1).sum()*self._n_bonds_per_shell[shell]/6

        if not return_rho_1:
            rho_1 = np.array([None])  
        return T_vir, P_vir, b_eps, rho_1
    
class Optimizer():
    """
    Class for NVT and NPT for mean-field calculations.
    
    Args:
        b_0s (list/np.array): The equilibrium bond length for each shell. Must be of length shells.
        mfm_instances (list): List of MeanFieldModel instances. Must be of length shells.
        vq_instances (list): List of VirialQuantities instances. Must be of length shells.
        energy_list (list/np.array): Energies of the energy-volume curve. Defaults to None, where no correction terms are added.
        strain_list (list/np.array): Strains corresponding to the volumes of the energy-volume curve. Defaults to None, where no correction terms are added.
    """
    def __init__(self, mfm_instances, vq_instances, energy_list=None, strain_list=None, r_order=1):
        self.mfm_instances = mfm_instances
        self.vq_instances = vq_instances
        self.energy_list = energy_list
        self.strain_list = strain_list
        self.r_order = r_order

        self.check_inputs()
        self._ev_fit_initialized = False

    def check_inputs(self):
        """
        Check if the inputs are properly defined.
        """
        
        for inp, inp_str in zip([self.mfm_instances, self.vq_instances], ['mfm_instances', 'vq_instances']):
            if len(inp) != self.r_order:
                raise TypeError(f"Length of {inp_str} is not equal to shell_order.")
            
        if (self.energy_list is not None) and (self.strain_list is not None):
            assert len(self.energy_list) == len(self.strain_list), 'length of energy_list and strain_list must be equal.'
            self.energy_list = np.array(self.energy_list)
            self.strain_list = np.array(self.strain_list)
        
        for vq, mfm in zip(self.vq_instances, self.mfm_instances):
            assert isinstance(vq, VirialQuantities), 'vq_instance must be a VirialQuantities instance.'
            assert isinstance(mfm, MeanFieldModel), 'mfm_instance must be a MeanFieldModel instance.'

        self.b_0s = self.mfm_instances[0].b_0s
        self.basis = self.mfm_instances[0].basis
        self._crystal = self.vq_instances[0].crystal
        self._n_bonds_per_shell = self.vq_instances[0]._n_bonds_per_shell
        self._factors = self.vq_instances[0]._factors
        
        
    def get_epsilon_pressure(self, eps=1.):
        """
        From the energy_list and strain_list inputs, determine the pressure and energy offsets from the mean-field model.
        
        Parameters:
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            
        Returns:
            P_offset (float/np.array): Pressure offsets.
            E_offset (float/np.array): Energy offsets.
        """
        if (np.any(self.energy_list) == None) and (np.any(self.strain_list) == None):
            if isinstance(eps, (int, float)):
                return 0., 0.
            else:
                return np.array([0.]*len(eps)), np.array([0.]*len(eps))
        else:
            if not self._ev_fit_initialized:
                strains = 1+self.strain_list
                self._u_md_fit_eqn = np.poly1d(np.polyfit(strains, self.energy_list, deg=4))
                self._du_md_fit_eqn = np.poly1d(np.polyfit(strains, np.gradient(self.energy_list, strains, edge_order=2), deg=4))

                u_mf = np.zeros(len(strains))
                for shell in range(self.r_order):
                    s_b_xyz = self.basis[shell][0]*(strains*self.b_0s[shell])[:, np.newaxis]
                    u_mf += self._n_bonds_per_shell[shell]*self.mfm_instances[shell].V1(s_b_xyz)/2
                self._u_mf_fit_eqn = np.poly1d(np.polyfit(strains, u_mf, deg=4))
                self._du_mf_fit_eqn = np.poly1d(np.polyfit(strains, np.gradient(u_mf, strains, edge_order=2), deg=4))

                if self._crystal == 'fcc':
                    self._N_by_V_0 = 4/(self.b_0s[0]*self._factors[0])**3
                elif self._crystal == 'bcc':
                    self._N_by_V_0 = 2/(self.b_0s[0]*self._factors[0])**3
                else:
                    raise TypeError('crystal must be fcc or bcc.')
                
                self._ev_fit_initialized = True

            E_offset = self._u_md_fit_eqn(eps)-self._u_mf_fit_eqn(eps)
            P_offset = -self._N_by_V_0/(3*eps**2)*(self._du_md_fit_eqn(eps)-self._du_mf_fit_eqn(eps))
            return P_offset, E_offset

    def collect_properties(self, temperature, lms, eps, Veffs=None, dV1s=None, return_rho_1=False):
        """
        Collects the properties of the system.
        
        Parameters:
            temperature (float): Temperature of the system.
            lms (list/np.array): Lagrange multipliers for each shell.
            eps (float): Strain on the system.
            Veffs (list/np.array, optional): Effective potentials for each shell. Defaults to None.
            dV1s (list/np.array, optional): Gradients of the bonding potential for each shell. Defaults to None.
            return_rho_1 (boolean): Whether or not to return the bond density.
        
        Returns:
            T_vir (np.array): Array of virial temperatures, calculated for each shell.
            P_vir (np.array): Array of virial pressures, calculated for each shell.
            P_offset (float): Offset pressure from the energy-volume curve, corresponding to the strain on the system. 
            E_offset (float): Offset energy from the energy-volume curve, corresponding to the strain on the system. 
            b_eps (np.array): Strained bond lengths, calculated for each shell.
            eps (float): Strain on the system.
            lms (np.array): Lagrange multipliers for each shell.
            rho_1 (np.ndarray): Bond density, if return_rho_1 is True.
        """
        T_vir = []
        P_vir = []
        b_eps = []
        rho_1s = []
        for shell in range(self.r_order):
            Veff = self.mfm_instances[shell].Veff(eps=eps) if Veffs is None else Veffs[shell]
            dV1 = self.mfm_instances[shell].V1_gradient() if dV1s is None else dV1s[shell]    
            t, p, b, r = self.vq_instances[shell].get_virial_quantities(Veff=Veff, dV1=dV1, temperature=temperature, eps=eps, lm=lms[shell], shell=shell, return_rho_1=return_rho_1)
            T_vir.append(t)
            P_vir.append(p)
            b_eps.append(b)
            rho_1s.append(r)
        P_offset, E_offset = self.get_epsilon_pressure(eps=eps)
        print('T: {}\nT_vir: {}\nP_vir: {}\nP_offset: {}\nE_offset: {}\neps: {}\nlm: {}\n'.format(temperature, np.array(T_vir), np.array(P_vir), P_offset, E_offset, eps, lms))
        return np.array(T_vir), np.array(P_vir), P_offset, E_offset, np.array(b_eps), eps, lms, np.array(rho_1s)

    def run_nvt(self, temperature=100., eps=1., return_rho_1=False, minimize_T=False):
        """
        Runs the mean-field NVT minimization for a given input temperature and strain.
        
        Parameters:
            temperature (float): Input temperature of the system.
            eps (float): Target strain of the system.
            return_rho_1 (boolean): Whether or not to return the bond density.
        
        Returns:
            T_vir (np.array): Array of virial temperatures calculated for each shell by the model. May not equal to the input temperature.
            P (np.array): Array of pressures calculated for each shell by the model.
            P_offset (float): Offset pressure from the energy-volume curve, corresponding to the strain on the system. 
            b_eps (np.array): Strained bond lengths calculated for each shell by the model.
            eps (float): Strain on the system.
            lms (np.array): Lagrange multipliers for each shell.
            rho_1 (np.ndarray): Bond density, if return_rho_1 is True.
        """
        # For each shell, find a lagrange multiplier that enforces bond length.
        # For each shell, find a lagrange multiplier that enforces bond length.
        lms = []
        Veffs = []
        dV1s = []
        print('T: {}\neps: {}\n'.format(temperature, eps))
        if not minimize_T:
            for shell in range(self.r_order):
                print('Optimizing shell {}...'.format(shell))
                Veff = self.mfm_instances[shell].Veff(eps=eps)
                dV1 = self.mfm_instances[shell].V1_gradient()
                def objective_function(args):
                    _, _, b_eps, _ = self.vq_instances[shell].get_virial_quantities(Veff=Veff, dV1=dV1, temperature=temperature, eps=eps, lm=args, shell=shell)
                    return np.abs(self.b_0s[shell]*eps-b_eps)
                solver = root_scalar(objective_function, x0=0., x1=0.001, rtol=1e-8)
                print('Optimization complete.')
                lms.append(solver.root)
                Veffs.append(Veff)
                dV1s.append(dV1)
            lms = np.array(lms)
        else:
            Veffs = []  
            for shell in range(self.r_order):
                print('Building Veff for shell {}...'.format(shell))
                Veffs.append(self.mfm_instances[shell].Veff(eps=eps))
            Veffs = np.array(Veffs)
            dV1s = np.array([self.mfm_instances[shell].V1_gradient() for shell in range(self.r_order)])
            def objective_function(args):
                T_virs = []
                b_eps_diff = []
                for shell in range(self.r_order):
                    t_vir, _, b_eps, _ = self.vq_instances[shell].get_virial_quantities(Veff=Veffs[shell], dV1=dV1s[shell], temperature=args[-1], eps=eps, lm=args[shell], shell=shell)
                    T_virs.append(t_vir)
                    b_eps_diff.append(self.b_0s[shell]-b_eps)
                T_diff = np.abs(np.sum(T_virs)-temperature)
                return np.concatenate((np.array(b_eps_diff), np.array([T_diff])))
            print('Optimizing all shells at once...')
            x0 = np.concatenate((np.zeros(self.r_order), np.array([temperature])))
            solver = root(objective_function, x0=x0, method='lm', tol=1e-8)
            print('Optimization complete.')
            lms = solver.x[:-1]
            if minimize_T:
                temperature = solver.x[-1]
                print('Effective/renormlaized temperature: {}'.format(solver.x[-1]))
        return self.collect_properties(temperature=temperature, lms=lms, eps=eps, Veffs=Veffs, dV1s=dV1s, return_rho_1=return_rho_1)
    
    def run_npt(self, temperature=100., pressure=1e-4, eps=1.02, return_rho_1=False, minimize_T=False):
        """
        Runs the mean-field NPT minimization for a given input temperature and pressure.
        
        Parameters:
            temperature (float): Input temperature of the system.
            pressure (float): Target pressure of the system.
            eps (float): Initial strain of the system.
            return_rho_1 (boolean): Whether or not to return the bond density.
        
        Returns:
            Returns:
            T_vir (np.array): Array of virial temperatures calculated for each shell by the model. May not equal to the input temperature.
            P (np.array): Array of pressures calculated for each shell by the model.
            P_offset (float): Offset pressure from the energy-volume curve, corresponding to the strain on the system. 
            b_eps (np.array): Strained bond lengths calculated for each shell by the model.
            eps (float): Strain on the system.
            lms (np.array): Lagrange multipliers for each shell.
            rho_1 (np.ndarray): Bond density, if return_rho_1 is True.
        """
        # Here, the Veff is redefined inside the objective function for each new value of eps.
        # Further, as the total pressure of the system is the sum of the pressures due to each shell, all shells are considered simultaneously in the objective function.
        print('T: {}\nP: {}\n'.format(temperature, pressure))
        print('Optimizing all shells at once. This might take a while...')
        dV1s = np.array([self.mfm_instances[shell].V1_gradient() for shell in range(self.r_order)]) 

        if not minimize_T:
            def objective_function(args):
                P_virs = []
                b_eps_diff = []
                for shell in range(self.r_order):
                    Veff = self.mfm_instances[shell].Veff(eps=args[-1])
                    _, p_vir, b_eps, _ = self.vq_instances[shell].get_virial_quantities(Veff=Veff, dV1=dV1s[shell], temperature=temperature, eps=args[-1], lm=args[shell], shell=shell)
                    P_virs.append(p_vir)
                    b_eps_diff.append(self.b_0s[shell]*args[-1]-b_eps)
                P_offset, _ = self.get_epsilon_pressure(eps=args[-1])
                P_diff = np.abs(np.sum(P_virs)+P_offset-pressure)
                print(args)
                return np.concatenate((np.array(b_eps_diff), np.array([P_diff])))
            x0 = np.concatenate((np.zeros(self.r_order), np.array([eps])))
            solver = root(objective_function, x0=x0, tol=1e-8)
            print('Optimization complete.')
            lms = solver.x[:self.r_order+1]
            eps = solver.x[-1]
        else:
            def objective_function(args):
                T_virs = []
                P_virs = []
                b_eps_diff = []
                for shell in range(self.r_order):
                    Veff = self.mfm_instances[shell].Veff(eps=args[-1])
                    t_vir, p_vir, b_eps, _ = self.vq_instances[shell].get_virial_quantities(Veff=Veff, dV1=dV1s[shell], temperature=args[-2], eps=args[-1], lm=args[shell], shell=shell)
                    T_virs.append(t_vir)
                    P_virs.append(p_vir)
                    b_eps_diff.append(self.b_0s[shell]*args[-1]-b_eps)
                P_offset, _ = self.get_epsilon_pressure(eps=args[-1])
                P_diff = np.abs(np.sum(P_virs)+P_offset-pressure)
                T_diff = np.abs(np.sum(T_virs)-temperature)
                print(args)
                return np.concatenate((np.array(b_eps_diff), np.array([T_diff]), np.array([P_diff])))
            x0 = np.concatenate((np.zeros(self.r_order), np.array([temperature]), np.array([eps])))
            solver = root(objective_function, x0=x0, method='lm', tol=1e-8)
            print('Optimization complete.')
            lms = solver.x[:-2]
            temperature = solver.x[-2]
            eps = solver.x[-1]
        return self.collect_properties(temperature=temperature, lms=lms, eps=eps, dV1s=dV1s, return_rho_1=return_rho_1)
    
class GenerateAlphas():
    """
    Class that generates the correlation matrices \alpha_{plr} between different bond pairs, defined for each shell with respect to a reference bond in that shell.

    Args:
        project_path (str): Path to the project directory.
        b_0s (list/np.array): The equilibrium bond length of each shell. Must be of length shells.
        basis (list/np.ndarray): (shells x 3 x 3) array of unit vectors for each shell. Must be of length shells.
        rotations (list/np.ndarray): Rotation matrices which transform the reference bond vector (l=0) to other bond vectors (k) in the same shell, for each shell. Must be of length shells.
        shells (int): Number of shells up to which the input b_0s, potentials, rotations, alphas are specified.
        r_order (int): Number of shells up to which to compute bond densities for. Defaults to 1, the 1st nearest neighbor shell.
        s_order (int): Number of shifts to consider for each shell. Defaults to 0, shifts up to r_order. 
                       (1 = up to the shell, including all inner shells, 2 = all inner shells, 1 outer shell, 3 = all inner shells, 2 outer shells, etc.)
        crystal (str): Crystal structure. Defaults to 'fcc'.
        bloch_hessians (list/np.array): Bloch hessians for each k-point. Defaults to None.
        kpoint_vectors (np.array): k-point vectors. Defaults to None.
    """
    def __init__(self, project_path, b_0s, basis, rotations, shells=1, r_order=1, s_order=0, crystal='fcc', bloch_hessians=None, kpoint_vectors=None, rewrite_alphas=False,
                 alpha_threshold=1e-2):
        self.project_path = project_path
        self.b_0s = b_0s
        self.basis = basis
        self.rotations = rotations
        self.shells = shells
        self.r_order = r_order
        self.s_order = s_order
        self.bloch_hessians = bloch_hessians
        self.kpoint_vectors = kpoint_vectors
        self.crystal = crystal
        self.rewrite_alphas = rewrite_alphas
        self.alpha_threshold = alpha_threshold

        self.alphas = None
        self.alphas_rot_ids = None

        self.check_inputs()

    def load_or_run(self):
        exists = [os.path.exists(os.path.join(self.project_path, file)) for file in ['alphas.npy', 'alphas_rot_ids.npy']]
        self.alphas = np.load(os.path.join(self.project_path, 'alphas.npy'), allow_pickle=True).tolist() if exists[0] else None
        self.alphas_rot_ids = np.load(os.path.join(self.project_path, 'alphas_rot_ids.npy'), allow_pickle=True).tolist() if exists[1] else None
        if (not np.all(exists)) or (self.rewrite_alphas):
            self.make_model()
            self.generate_alphas(write=True)

    def check_inputs(self):
        """
        """
        # factors to get primitive cells from the basis
        if self.crystal == 'fcc':
            # set the number of bonds per shell.
            self._n_bonds_per_shell = np.array([12, 6, 24, 12, 24, 8])
            self._primitive_factors = np.array([np.sqrt(2)*0.5, 1., np.sqrt(3/2), np.sqrt(2), np.sqrt(5/2), np.sqrt(3)])
        elif self.crystal == 'bcc':
            self._n_bonds_per_shell = np.array([8, 6, 12, 24, 8, 6, 24, 24])
            self._primitive_factors = np.array([np.sqrt(3)*0.5, 1., np.sqrt(2), np.sqrt(11)/2, np.sqrt(3), 2., np.sqrt(19)/2, np.sqrt(5)])
        else:
            raise ValueError('crystal must be fcc or bcc')
        
    def make_model(self):
        # Iterate over all Bloch modes
        nus, eig_vecs = [], []
        for H in self.bloch_hessians:
            ev, evec = np.linalg.eigh(H)
            nus.append(ev)
            eig_vecs.append(evec.T)
        
        self._nu = np.array(nus).flatten()
        nu_inv = 1/self._nu
        
        self._V = np.array(eig_vecs).reshape((-1, 3))
        self._AC = np.einsum('ij,ik->ijk', self._V, self._V)
        self._AC = np.einsum('i,ijk->ijk', nu_inv, self._AC)
    
    def get_correlation(self, bond_1, bond_0, shift, threshold=1e-3):
        kps = self.kpoint_vectors.repeat(3, axis=0)
        # we match terms k,-k as this is the noise correlation
        r_p = np.exp(-1j*kps@shift)+np.exp(-1j*kps@(shift+bond_1-bond_0))-np.exp(-1j*kps@(shift-bond_0))-np.exp(-1j*kps@(shift+bond_1))
        sel = self._nu > threshold*self._nu.max()
        return np.einsum('i,ijk->jk', r_p[sel], self._AC[sel])/self.kpoint_vectors.shape[0]
        
    def generate_alphas(self, write=True):
        """
        Generate the correlation matrices \alpha_{plr} between different bond pairs, defined for a shell with respect to a reference bond in that shell.
        """
        print('Generating alphas...')

        self._nn_pos = []
        for s in range(self.shells):
            self._nn_pos.append(self.basis[s][0]@self.rotations[s]*self._primitive_factors[s])

        self.alphas = []
        self.alphas_rot_ids = []
        for shell in range(self.r_order):
            ref_bond = self._nn_pos[shell][0]
            bond_0_corr = self.get_correlation(ref_bond, ref_bond, shift=np.array([0.0, 0.0, 0.0]))
            inv_bond_0_corr = np.linalg.inv(bond_0_corr)

            # write exceptions for this!
            upto_shell = min(self.shells, shell+self.s_order+1)
            if (self.crystal == 'bcc') and (shell in [0]):
                upto_shell = min(self.shells, shell+self.s_order+2)
            shifts = np.vstack((np.zeros((1, 3)), *self._nn_pos[:upto_shell]))

            # iterate over all 1NN, 2NN etc. bonds
            alphas = [[] for _ in range(upto_shell)]
            alphas_rot_ids = [[] for _ in range(upto_shell)]
            for shift in shifts:
                for shl, nn in enumerate(self._nn_pos[:upto_shell]):
                    for bond_1 in nn:
                        bond_1_corr = self.get_correlation(bond_1=bond_1, bond_0=ref_bond, shift=shift)
                        alphas[shl].append(np.real(bond_1_corr@inv_bond_0_corr))
            
            # reshape the alphas to be (no. of shifts, neighbors in a shell, 3, 3)
            for shl in range(upto_shell):
                alphas[shl] = np.array(alphas[shl]).reshape(len(shifts), self._n_bonds_per_shell[shl], 3, 3)
                alphas_rot_ids[shl] = [np.argwhere(np.all(np.any(abs(alphas[shl][i])>=self.alpha_threshold, axis=-1), axis=-1)).flatten() for i in range(len(alphas[shl]))] 
                for i in range(len(alphas[shl])):
                    if np.any(alphas_rot_ids[shl][i]):
                        continue
                    else:
                        alphas_rot_ids[shl][i] = np.array([0])
                        alphas[shl][i] = np.zeros((3, 3))
                alphas[shl] = [alphas[shl][i][alphas_rot_ids[shl][i]] for i in range(len(alphas_rot_ids[shl]))]
            self.alphas.append(alphas)
            self.alphas_rot_ids.append(alphas_rot_ids)
        if write:
            np.save(os.path.join(self.project_path, 'alphas.npy'), np.array(self.alphas, dtype=object))
            np.save(os.path.join(self.project_path, 'alphas_rot_ids.npy'), np.array(self.alphas_rot_ids, dtype=object))
        print('Alphas generated.')
    
class MeanFieldJob():
    """
    User class for running the mean-field anharmonic bond (MAB) model for the NPT or NVT ensemble to generate anharmonic internal energies.
        
    Args:
        project_path (str): Path to the project directory.
        shells (int): Number of nearest neighbor shells to consider for the model. Defaults to 1 shell, the 1st nearest neighbor shell.
        r_order (int): Number of shells up to which to compute bond densities for. Defaults to 1, the 1st nearest neighbor shell.
        s_order (int): Number of shifts to consider for each shell. Defaults to 0, shifts up to r_order. 
                       (1 = up to the shell, including all inner shells, 2 = all inner shells, 1 outer shell, 3 = all inner shells, 2 outer shells, etc.)
        bond_grids (list/np.ndarray): (shells, len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates for each shell. Must be of length shells..
        b_0s (list/np.array): The equilibrium bond length of each shell. Must be of length shells.
        basis (list/np.ndarray): (shells x 3 x 3) array of unit vectors for each shell. Must be of length shells.
        rotations (list/np.ndarray): Rotation matrices which transform the reference bond vector (l=0) to other bond vectors (k) in the same shell, for each shell. Must be of length shells.
        kpoints (int): Number of kpoints to use for the Debye model. Defaults to 10.
        r_potentials (list of functions): Function for the ij (or 'radial' or 'central') potential (we take this to be the same as the longitudinal (lo) potential). Must be of length shells.
        t1_potentials (list of functions, optional): Function for the t1 potential, if any. Defaults to None. Must be of length shells.
        t2_potentials (list of functions, optional): Function for the t2 potential, if any. Defaults to None.  Must be of length shells.
        crystal (str): Crystal structure. Currently only supports 'fcc'.
        energy_list (list/np.array): Energies of the energy-volume curve. Defaults to None, where no correction terms are added.
        strain_list (list/np.array): Strains corresponding to the volumes of the energy-volume curve. Defaults to None, where no correction terms are added.
    """

    def __init__(self, project_path, bond_grids, b_0s, basis, rotations, r_potentials, t1_potentials=None, t2_potentials=None, shells=1, r_order=1, s_order=0,
                 crystal='fcc', energy_list=None, strain_list=None, bloch_hessians=None, kpoint_vectors=None, rewrite_alphas=False, alpha_threshold=1e-3):
        self.project_path = project_path
        self.b_0s = b_0s
        self.basis = basis
        self.rotations = rotations
        self.bond_grids = bond_grids
        self.r_potentials = r_potentials
        self.t1_potentials = t1_potentials
        self.t2_potentials = t2_potentials
        self.shells = shells
        self.r_order = r_order
        self.s_order = s_order
        self.crystal = crystal
        self.energy_list = energy_list
        self.strain_list = strain_list
        self.bloch_hessians = bloch_hessians
        self.kpoint_vectors = kpoint_vectors
        self.rewrite_alphas = rewrite_alphas
        self.alpha_threshold = alpha_threshold

        self.output = None
        self.rho_1s = None
        self._npt = False

        self.check_inputs()
        self.meshes = np.array([get_meshes(self.bond_grids[s], self.basis[s]) for s in range(self.r_order)])

    def check_inputs(self):
        """
        Check if the inputs are properly defined.
        """
        assert isinstance(self.shells, int), 'shells must be an integer.'
        assert self.shells > 0, 'shells must be greater than 0.'
        assert len(self.bond_grids) == self.shells, 'length of bond_grids must be equal to shells.'
        assert isinstance(self.r_order, int), 'r_order must be an integer.'
        assert self.r_order <= self.shells, 'r_order must be lesser than or equal to shells.'
        assert isinstance(self.s_order, int), 's_order must be an integer.'
        # assert self.s_order+self.r_order <= self.shells, 's_order+r_order must be lesser than or equal to shells.'

        if (self.energy_list is not None) and (self.strain_list is not None):
            assert len(self.energy_list) == len(self.strain_list), 'length of energy_list and strain_list must be equal.'
            self.energy_list = np.array(self.energy_list)
            self.strain_list = np.array(self.strain_list)

        self.generate_alphas()
        self.initialize_instances()

    def generate_alphas(self):
        """
        Generate the correlation matrices \alpha_{plr} between different bond pairs, defined for a shell with respect to a reference bond in that shell.
        """
        alp = GenerateAlphas(project_path=self.project_path, 
                             b_0s=self.b_0s, 
                             basis=self.basis, 
                             rotations=self.rotations, 
                             shells=self.shells, 
                             r_order=self.r_order, 
                             s_order=self.s_order, 
                             crystal=self.crystal, 
                             bloch_hessians=self.bloch_hessians, 
                             kpoint_vectors=self.kpoint_vectors,
                             rewrite_alphas=self.rewrite_alphas,
                             alpha_threshold=self.alpha_threshold)
        alp.load_or_run()
        self.alphas = alp.alphas
        self.alphas_rot_ids = alp.alphas_rot_ids
        self._n_bonds_per_shell = alp._n_bonds_per_shell
    
    def initialize_instances(self):
        """
        Initialize MeanFieldModel and VirialQuantities instances for each shell.
        """
        print('Initializing instances...')
        self._mfms = []
        self._vqs = []
        for r in range(self.r_order):
            l_order = len(self.alphas[r])
            self._mfms.append(MeanFieldModel(bond_grid=self.bond_grids[r], 
                                             b_0s=self.b_0s, 
                                             basis=self.basis, 
                                             rotations=self.rotations, 
                                             alphas=self.alphas[r], 
                                             alphas_rot_ids=self.alphas_rot_ids[r], 
                                             ref_shell=r, 
                                             shells=self.shells, 
                                             r_potentials=self.r_potentials, 
                                             t1_potentials=self.t1_potentials, 
                                             t2_potentials=self.t2_potentials, 
                                             l_order=l_order))
            self._vqs.append(VirialQuantities(bond_grid=self.bond_grids[r], 
                                              b_0=self.b_0s[r], 
                                              basis=self.basis[r], 
                                              crystal=self.crystal))
        print('Instances initialized.')

    def run_ensemble(self, temperature=100., pressure=None, eps=None, return_output=False, re_run=False, minimize_T=False):
        """
        Optimize the mean-field model parameters to simulate an NPT or NVT simulation. Either pressure or eps must be specified. If pressure is specified, run the NPT model. If eps is specified, run the NVT model.

        Parameters:
            temperature (float): Input temperature of the system. Defaults to 100. K.
            pressure (float): Target pressure of the system. Defaults to None.
            eps (float): Target strain of the system. Defaults to None.
            return_output (boolean): Whether or not to return the thermo output of the model. Defaults to False.
            re_run (boolean): Whether or not to re-run the job. Defaults to False.

        Returns:
            output (dict): Dictionary containing the output of the mean field job, if return_output is True.
        """
        assert isinstance(temperature, (int, float)), 'temperature must be an integer or float.'
        assert temperature > 0., 'temperature must be greater than 0.'

        if self.output is not None:
            if not re_run:
                raise ValueError('job has already been run. Set re_run=True to re-run the job.')
            else:
                self.output = None
                self.rho_1s = None
        
        self._temperature = temperature
        opt = Optimizer(mfm_instances=self._mfms, vq_instances=self._vqs, energy_list=self.energy_list, strain_list=self.strain_list, r_order=self.r_order)
        
        if minimize_T:
            print('Temperature will be optimized.')
            
        if pressure is None:
            assert eps is not None, 'eps must be specified if pressure is None.'
            assert isinstance(eps, (int, float)), 'eps must be an integer an integer or float.'
            print('Pressure not specified, running NVT model... ')
            out = opt.run_nvt(temperature=temperature, eps=eps, return_rho_1=True, minimize_T=minimize_T)
        elif pressure is not None:
            assert isinstance(pressure, (int, float)), 'pressure must be an integer or float.'
            self._npt = True
            if eps is None:
                eps = 1.02
            print('Pressure specified, setting intial eps = {}, running NPT model...'.format(eps))
            out = opt.run_npt(temperature=temperature, pressure=pressure, eps=eps, return_rho_1=True, minimize_T=minimize_T)
        else:
            raise ValueError('either pressure or eps must be specified.')
        
        self.output = {'T': temperature,
                       'T_vir': out[0].tolist(),
                       'P': out[1].tolist(),
                       'P_offset': out[2],
                       'E_offset': out[3],
                       'b_eps': out[4].tolist(),
                       'eps': out[5],
                       'lms': out[6].tolist()}
        self.rho_1s = out[7]
        ah_U, per_shell_ah_U = self.get_anharmonic_U(return_components=True)
        print('per_shell_ah_U: {} meV/atom\n'.format(per_shell_ah_U*1000))
        print('ah_U: {} meV/atom\n'.format(ah_U*1000))
        self.output['per_shell_ah_U'] = (per_shell_ah_U*1000).tolist()
        self.output['ah_U'] = ah_U*1000
        
        if return_output:
            return self.output
    
    def get_rho_1s(self):
        """
        Get the bond density of each shell.

        Returns:
            rho_1s (np.ndarray): Array containing the bond density of each shell.
        """
        assert self.rho_1s is not None, 'run_ensemble() must be run first.'
        return self.rho_1s
    
    def get_anharmonic_U(self, return_components=False):
        """
        Get the anharmonic internal energy in [eV/atom] estimated by the model.

        Returns:
            ah_U (float): Anharmonic internal energy.
        """
        assert self.output is not None, 'run_ensemble() must be run first.'
        
        per_bond_energy = np.array([(self._mfms[s].V1(bond_grid=self.bond_grids[s], shell=s)*self.rho_1s[s]).sum() for s in range(self.r_order)])
        per_atom_energy = (self._n_bonds_per_shell[:self.r_order]/2*per_bond_energy)
        per_shell_ah_U = per_atom_energy-1.5*KB*np.array(self.output['T_vir'])
        ah_U = per_shell_ah_U.sum()
        if self._npt:
            ah_U += self.output['E_offset']
        if return_components:
            return ah_U, per_shell_ah_U
        return ah_U
    