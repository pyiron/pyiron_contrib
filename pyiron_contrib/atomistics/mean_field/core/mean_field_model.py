import numpy as np

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
        bond_grid (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates. Must be defined with respect to the reference shell.
        shells (int): Number of nearest-neighbor shells to consider for the mean-field model.
        b_0s (list/np.array): The equilibrium bond length of each shell. Must be of length shells.
        basis (list/np.ndarray): (shells x 3 x 3) array of unit vectors for each shell. Must be of length shells.
        rotations (list/np.ndarray): Rotation matrices which transform the reference bond vector (l=0) to other bond vectors (k) in the same shell, for each shell. Must be of length shells.
        alphas (list/np.ndarray): Correlation matrices \alpha_{pl} between different bond pairs, defined for each shell with respect to a reference bond in that shell. Must be of length shells.
        
        r_potentials (list of functions): Function for the ij (or 'radial' or 'central') potential (we take this to be the same as the longitudinal (lo) potential). Must be of length shells.
        t1_potentials (list of functions, optional): Function for the t1 potential, if any. Defaults to None. Must be of length shells.
        t2_potentials (list of functions, optional): Function for the t2 potential, if any. Defaults to None.  Must be of length shells.
    """

    def __init__(self, bond_grid, b_0s, basis, rotations, alphas, r_potentials, t1_potentials=None, t2_potentials=None, shells=1, ref_shell=0):
        self.bond_grid = bond_grid
        self.b_0s = b_0s
        self.basis = basis
        self.rotations = rotations
        self.alphas = alphas
        self.r_potentials = r_potentials
        self.t1_potentials = t1_potentials
        self.t2_potentials = t2_potentials
        self.shells = shells
        self.ref_shell = ref_shell

        self.meshes = get_meshes(self.bond_grid, self.basis[ref_shell])

    def check_inputs(self):
        """
        Check if the inputs are properly defined.
        """
        assert isinstance(self.shells, int), 'shells must be an integer.'
        assert self.shells > 0, 'shells must be greater than 0.'
        assert isinstance(self.ref_shell, int), 'ref_shell must be an integer.'
        assert self.ref_shell < self.shells, 'ref_shell must be less than shells.'
              
        inputs = [self.b_0s, self.basis, self.rotations, self.alphas, self.r_potentials, self.t1_potentials, self.t2_potentials]
        inputs_str = ['b_0s', 'basis', 'rotations', 'alphas', 'r_potentials', 't1_potentials', 't2_potentials']
        for inp, inp_str in zip(inputs, inputs_str):
            if len(inp) != self.shells:
                raise TypeError(f'length of {inp_str} is not equal to shells.')

    def V1(self, bond_grid=None, shell=None, rotation=np.eye(3)):
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
        r = np.linalg.norm(bond_grid, axis=-1)
        v1 = self.r_potentials[shell](r)
        if self.t1_potentials is not None:
            t1 = np.dot(bond_grid, self.basis[shell][1]@rotation)
            v1 += self.t1_potentials[shell](t1)
        if self.t2_potentials is not None:
            t2 = np.dot(bond_grid, self.basis[shell][2]@rotation)
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

    def Vmf_component(self, shell=None, rotation=np.eye(3), alpha=np.eye(3), eps=1.):
        """
        Calculate the bonding potential for a shell along the 'k' direction using symmetry operations.

        Parameters:
            shell (int, optional): Shell for which the bonding potential is calculated. Defaults to 0, the reference shell.
            rotation (3 x 3, optional): Rotation matrix between the reference direction (l=0) and the desired direction (k) in the shell. Defaults to the identity matrix.
            alpha (3 x 3, optional): Correlation matrix. Defaults to the identity matrix.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

        Returns:
            v1^k (np.ndarray): The bonding potential along the 'k' direction.
        """
        shell = self.ref_shell if shell is None else shell
        b_l = self.bond_grid
        a_l = self.b_0s[self.ref_shell]*eps*self.basis[self.ref_shell][0]
        a_k = self.b_0s[shell]*eps*(self.basis[shell][0]@rotation)
        b_k = (b_l-a_l)@alpha+a_k  # here, alpha scales the displacement from equilibrium, i.e. db = (b_l-a_l)
        return self.V1(bond_grid=b_k, shell=shell, rotation=rotation)

    def Veff(self, eps=1.):
        """
        Calculate the correlated mean-field effective potential.

        Parameters:
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

        Returns:
            vmfc (np.ndarray): The correlated mean-field effective potential.
        """
        vmfc = 0
        for shell in range(self.shells):
            # each shell with a ref bond b_{0r} has its own alphas, which have the shape (no. of shifts, total neighbors, 3, 3)
            # total neighbors can include 1NN, 2NN, 3NN, etc.
            # for convenience, we sort the alphas by the shell 
            # i.e. alphas[shell_0].shape = (no. of shifts, 12, 3, 3), alphas[shell_1].shape = (no. of shifts, 6, 3, 3), etc.
            # this is done so we can choose the correct symmetry operation (rotation) for each shell 
            for shift in self.alphas[shell]:
                for rot, al in zip(self.rotations[shell], shift):
                    vmfc += 0.5*self.Vmf_component(shell=shell, rotation=rot, alpha=al, eps=eps)
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
        assert self.crystal in ['fcc'], 'crystal must be fcc.'   

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
        num_offset = np.min(rho_1[rho_1>0])  # add a numerical offset to avoid divide by zero errors
        return rho_1+num_offset

    def get_virial_quantities(self, Veff, dV1, temperature=100., eps=1., lm=0., shell=1, return_rho_1=False):
        """
        Calculate virial temperature, pressure, and equilibrium bond at the given strain.

        Parameters:
            Veff (float): The correlated mean-field effective potential.
            dV1 (3, np.ndarray): The gradient of the single bonding potential along the longitudinal and 2 transversal directions.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            lm (float, optional): Lagrange multiplier. Defaults to 0.
            shell (int, optional): Shell with respect to which the volume will be calculated. Defaults to 1, the 1st shell.
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
        a_dV1 = b_eps*dV1[0]
        # for each shell, we have a different number of nn bonds and determine volume differently based off of b_eps
        if shell == 0:
            N_by_V = 4/(b_eps*np.sqrt(2))**3
            n_bonds = 12
        elif shell == 1:
            N_by_V = 4/(b_eps)**3
            n_bonds = 6
        elif shell == 2:
            N_by_V = 4/(b_eps*np.sqrt(2/3))**3
            n_bonds = 24
        else:
            raise ValueError('shell must be 0, 1, or 2.')
        T_vir = n_bonds/6/KB*(db_dV1*rho_1).sum()
        P_vir = -n_bonds/6*N_by_V*(a_dV1*rho_1).sum()
        if not return_rho_1:
            rho_1 = np.array([None])  
        return T_vir, P_vir, b_eps, rho_1
    
class Optimizer():
    """
    Class for NVT and NPT for mean-field calculations.
    
    Args:
        shells (int): Number of nearest-neighbor shells to consider for the mean-field model.
        b_0s (list/np.array): The equilibrium bond length for each shell. Must be of length shells.
        mfm_instances (list): List of MeanFieldModel instances. Must be of length shells.
        vq_instances (list): List of VirialQuantities instances. Must be of length shells.
        energy_list (list/np.array): Energies of the energy-volume curve. Defaults to None, where no correction terms are added.
        strain_list (list/np.array): Strains corresponding to the volumes of the energy-volume curve. Defaults to None, where no correction terms are added.
        crystal (str): Crystal structure. Defaults to 'fcc'.
    """
    def __init__(self, b_0s, mfm_instances, vq_instances, energy_list=None, strain_list=None, crystal='fcc', shells=1):
        self.b_0s = b_0s
        self.shells = shells
        self.mfm_instances = mfm_instances
        self.vq_instances = vq_instances
        self.energy_list = energy_list
        self.strain_list = strain_list
        self.crystal = crystal

        self.check_inputs()

        if self.crystal == 'fcc':
            # set the number of bonds per shell.
            self._n_bonds = np.array([12, 6, 24])

    def check_inputs(self):
        """
        Check if the inputs are properly defined.
        """
        
        for inp, inp_str in zip([self.mfm_instances, self.vq_instances], ['mfm_instances', 'vq_instances']):
            if len(inp) != self.shells:
                raise TypeError(f"Length of {inp_str} is not equal to shells.")
        
        for vq, mfm in zip(self.vq_instances, self.mfm_instances):
            assert len(self.energy_list) == len(self.strain_list), 'length of energy_list and strain_list must be equal.'
            assert isinstance(vq, VirialQuantities), 'vq_instance must be a VirialQuantities instance.'
            assert isinstance(mfm, MeanFieldModel), 'mfm_instance must be a MeanFieldModel instance.'

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
            # from static calculations
            strains = 1+self.strain_list
            u_md_fit_eqn = np.poly1d(np.polyfit(strains, self.energy_list, deg=4))
            # from mean-field model
            u_mf = 0.
            for shell in range(self.shells):
                u_mf += self._n_bonds[shell]/2.*self.mfm_instances[shell].r_potentials[shell](strains*self.b_0s[shell])
            u_mf_fit_eqn = np.poly1d(np.polyfit(strains, u_mf, deg=4))
            E_offset = u_md_fit_eqn(eps)-u_mf_fit_eqn(eps)
            
            e_offset_fit = np.poly1d(np.polyfit(strains, self.energy_list-u_mf, deg=4))
            de_offset_fit = e_offset_fit.deriv(m=1)(eps)
            P_offset = -1./(3.*eps**2*(self.b_0s[0]*np.sqrt(2))**3)*de_offset_fit
            
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
        for shell in range(self.shells):
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

    def run_nvt(self, temperature=100., eps=1., return_rho_1=False):
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
        lms = []
        Veffs = []
        dV1s = []
        print('T: {}\neps: {}\n'.format(temperature, eps))
        for shell in range(self.shells):
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
        return self.collect_properties(temperature=temperature, lms=lms, eps=eps, Veffs=Veffs, dV1s=dV1s, return_rho_1=return_rho_1)
    
    def run_npt(self, temperature=100., pressure=1e-4, return_rho_1=False):
        """
        Runs the mean-field NPT minimization for a given input temperature and pressure.
        
        Parameters:
            temperature (float): Input temperature of the system.
            pressure (float): Target pressure of the system.
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
        dV1s = np.array([self.mfm_instances[shell].V1_gradient() for shell in range(self.shells)])
        def objective_function(args):
            P_virs = []
            b_eps_diff = []
            for shell in range(self.shells):
                Veff = self.mfm_instances[shell].Veff(eps=args[-1])
                _, p_vir, b_eps, _ = self.vq_instances[shell].get_virial_quantities(Veff=Veff, dV1=dV1s[shell], temperature=temperature, eps=args[-1], lm=args[shell], shell=shell)
                P_virs.append(p_vir)
                b_eps_diff.append(self.b_0s[shell]*args[-1]-b_eps)
            P_diff = np.abs(np.sum(P_virs)-pressure)
            print(args)
            return np.concatenate((np.array(b_eps_diff), np.array([P_diff])))
        x0 = np.concatenate((np.zeros(self.shells), np.array([1.])))
        solver = root(objective_function, x0=x0, method='lm', tol=1e-8)
        print('Optimization complete.')
        lms = solver.x[:-1]
        eps = solver.x[-1]
        return self.collect_properties(temperature=temperature, lms=lms, eps=eps, dV1s=dV1s, return_rho_1=return_rho_1)
    
class MeanFieldJob():
    """
    End-user class for running the mean-field anharmonic bond (MAB) model for the NPT or NVT ensemble to generate anharmonic internal energies.

    bond_grid (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates. Must be defined with respect to the reference shell.
        shells (int): Number of nearest-neighbor shells to consider for the mean-field model.
        b_0s (list/np.array): The equilibrium bond length of each shell. Must be of length shells.
        
    Args:
        shells (int): Number of nearest neighbor shells to consider for the model. Defaults to 1 shell, the 1st nearest neighbor shell.
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

    def __init__(self, bond_grids, b_0s, basis, rotations, r_potentials, t1_potentials=None, t2_potentials=None, shells=1, kpoints=10, crystal='fcc', energy_list=None, strain_list=None):
        self.b_0s = b_0s
        self.basis = basis
        self.rotations = rotations
        self.kpoints = kpoints
        self.bond_grids = bond_grids
        self.r_potentials = r_potentials
        self.t1_potentials = t1_potentials
        self.t2_potentials = t2_potentials
        self.shells = shells
        self.crystal = crystal
        self.energy_list = energy_list
        self.strain_list = strain_list
        self.alphas = None
        self.output = None
        self.rho_1s = None
        self._mfms = None
        self._vqs = None
        self._n_bonds = None

        self.check_inputs()
        self.meshes = np.array([get_meshes(self.bond_grids[s], self.basis[s]) for s in range(self.shells)])

    def check_inputs(self):
        """
        Check if the inputs are properly defined.
        """
        assert isinstance(self.shells, int), 'shells must be an integer.'
        assert self.shells > 0, 'shells must be greater than 0.'
        assert len(self.bond_grids) == self.shells, 'length of bond_grids must be equal to shells.'
        assert len(self.energy_list) == len(self.strain_list), 'length of energy_list and strain_list must be equal.'
        assert self.crystal in ['fcc'], 'crystal must be fcc.'

        if self.crystal == 'fcc':
            # set the number of bonds per shell.
            self._n_bonds = np.array([12, 6, 24])

        self.generate_alphas()
        self.initialize_instances()
    
    def initialize_instances(self):
        """
        Initialize MeanFieldModel and VirialQuantities instances for each shell.
        """
        print('Initializing instances...')
        self._mfms = []
        self._vqs = []
        for s in range(self.shells):
            self._mfms.append(MeanFieldModel(bond_grid=self.bond_grids[s], b_0s=self.b_0s, basis=self.basis, rotations=self.rotations, alphas=self.alphas[s], shells=self.shells,  ref_shell=s,
                                             r_potentials=self.r_potentials, t1_potentials=self.t1_potentials, t2_potentials=self.t2_potentials))
            self._vqs.append(VirialQuantities(bond_grid=self.bond_grids[s], b_0=self.b_0s[s], basis=self.basis[s], crystal=self.crystal))
        print('Instances initialized.')

    def generate_alphas(self):
        """
        Generate the correlation matrices \alpha_{plr} between different bond pairs, defined for a shell with respect to a reference bond in that shell.
        """
        print('Generating alphas...')
        nn_shifts = []
        for shell in range(self.shells):
            if shell == 0:  # convert to primitive cell vector
                factor = np.sqrt(2)*0.5
            elif shell == 1:
                factor = 1.
            elif shell == 2:
                factor = np.sqrt(3/2)
            nn_shifts.append(self.basis[shell][0]@self.rotations[shell]*factor)
        shifts = np.vstack((np.zeros((1, 3)), *nn_shifts))

        model = DebyeModel(lattice='fcc', kappa_t=0.0, kpoints=self.kpoints)

        alphas = []
        for shell in range(self.shells):
            ref_bond = nn_shifts[shell][0]
            bond_0_corr = model.correlation(ref_bond, ref_bond, shift=np.array([0.0, 0.0, 0.0]))
            inv_bond_0_corr = np.linalg.inv(bond_0_corr)

            alphas_nn = [[], [], []]  # each shell with a reference bond has correlations with 1NN, 2NN, 3NN bonds.
            for shift in shifts:
                for shl, nn in enumerate(nn_shifts):
                    for bond_1 in nn:
                        bond_1_corr = model.correlation(bond_1=bond_1, bond_0=ref_bond, shift=shift)
                        alphas_nn[shl].append(bond_1_corr@inv_bond_0_corr)
            
            for shell in range(self.shells):  # reshape the alphas to be (no. of shifts, neighbors in a shell, 3, 3)
                alphas_nn[shell] = np.array(alphas_nn[shell]).reshape(len(shifts), self._n_bonds[shell], 3, 3)
            alphas.append(alphas_nn)
        self.alphas = alphas
        print('Alphas generated.')

    def run_ensemble(self, temperature=100., pressure=None, eps=None, return_output=False, re_run=False):
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
        if self.output is not None:
            if not re_run:
                raise ValueError('job has already been run. Set re_run=True to re-run the job.')
            else:
                self.output = None
                self.rho_1s = None
        assert isinstance(temperature, (int, float)), 'temperature must be an integer or float.'
        assert temperature > 0., 'temperature must be greater than 0.'

        opt = Optimizer(b_0s=self.b_0s, shells=self.shells, mfm_instances=self._mfms, vq_instances=self._vqs, energy_list=self.energy_list, 
                        strain_list=self.strain_list, crystal=self.crystal)
        
        if pressure is None:
            assert eps is not None, 'eps must be specified if pressure is None.'
            assert isinstance(eps, (int, float)), 'eps must be an integer an integer or float.'
            out = opt.run_nvt(temperature=temperature, eps=eps, return_rho_1=True)
        elif pressure is not None:
            assert eps is None, 'eps must be None if pressure specified.'
            assert isinstance(pressure, (int, float)), 'pressure must be an integer or float.'
            out = opt.run_npt(temperature=temperature, pressure=pressure, return_rho_1=True)
        else:
            raise ValueError('either pressure or eps must be specified.')
        
        self.output = {'T': temperature,
                       'T_vir': out[0],
                       'P': out[1],
                       'P_offset': out[2],
                       'E_offset': out[3],
                       'b_eps': out[4],
                       'eps': out[5],
                       'lms': out[6]}
        self.rho_1s = out[7]
        
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
        
        per_bond_energy = np.array([(self._mfms[s].V1(bond_grid=self.bond_grids[s], shell=s)*self.rho_1s[s]).sum() for s in range(self.shells)])
        per_atom_energy = (self._n_bonds[:self.shells]/2*per_bond_energy)
        per_shell_ah_U = per_atom_energy-1.5*KB*self.output['T_vir']
        ah_U = per_shell_ah_U.sum()+self.output['E_offset']
        if return_components:
            return ah_U, per_shell_ah_U
        return ah_U
    

# class MeanFieldModelVectorized():
#     """
#     Class that computes the mean-field effective potential.

#     Args:
#         ref_shell (int): Reference shell with respect to which the mean-field model is defined.
#         bond_grid (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates. Must be defined with respect to the reference shell.
#         shells (int): Number of nearest-neighbor shells to consider for the mean-field model.
#         b_0s (list/np.array): The equilibrium bond length of each shell. Must be of length shells.
#         basis (list/np.ndarray): (shells x 3 x 3) array of unit vectors for each shell. Must be of length shells.
#         rotations (list/np.ndarray): Rotation matrices which transform the reference bond vector (l=0) to other bond vectors (k) in the same shell, for each shell. Must be of length shells.
#         alphas (list/np.ndarray): Correlation matrices \alpha_{pl} between different bond pairs, defined for each shell with respect to a reference bond in that shell. Must be of length shells.
        
#         r_potentials (list of functions): Function for the ij (or 'radial' or 'central') potential (we take this to be the same as the longitudinal (lo) potential). Must be of length shells.
#         t1_potentials (list of functions, optional): Function for the t1 potential, if any. Defaults to None. Must be of length shells.
#         t2_potentials (list of functions, optional): Function for the t2 potential, if any. Defaults to None.  Must be of length shells.
#     """

#     def __init__(self, bond_grid, b_0s, basis, rotations, alphas, r_potentials, t1_potentials=None, t2_potentials=None, shells=1, ref_shell=0):
#         self.bond_grid = bond_grid
#         self.b_0s = b_0s
#         self.basis = basis
#         self.rotations = rotations
#         self.alphas = alphas
#         self.r_potentials = r_potentials
#         self.t1_potentials = t1_potentials
#         self.t2_potentials = t2_potentials
#         self.shells = shells
#         self.ref_shell = ref_shell

#         self.meshes = get_meshes(self.bond_grid, self.basis[ref_shell])

#     def check_inputs(self):
#         """
#         Check if the inputs are properly defined.
#         """
#         assert isinstance(self.shells, int), 'shells must be an integer.'
#         assert self.shells > 0, 'shells must be greater than 0.'
#         assert isinstance(self.ref_shell, int), 'ref_shell must be an integer.'
#         assert self.ref_shell < self.shells, 'ref_shell must be less than shells.'
              
#         inputs = [self.b_0s, self.basis, self.rotations, self.alphas, self.r_potentials, self.t1_potentials, self.t2_potentials]
#         inputs_str = ['b_0s', 'basis', 'rotations', 'alphas', 'r_potentials', 't1_potentials', 't2_potentials']
#         for inp, inp_str in zip(inputs, inputs_str):
#             if len(inp) != self.shells:
#                 raise TypeError(f'length of {inp_str} is not equal to shells.')

#     def V1(self, bond_grid=None, shell=None, rotation=np.eye(3)):
#         """
#         Calculate the bonding potential.

#         Parameters:
#             bond_grid (np.ndarray, optional): Modified bond grid. Defaults to None, uses self.bond_grid.
#             shell (int, optional): Shell for which the bonding potential is calculated. Defaults to None, the reference shell.
#             rotation (3 x 3, optional): Rotation matrix. Defaults to the identity matrix.

#         Returns:
#             v1 (np.ndarray): The bonding potential.
#         """
#         bond_grid = self.bond_grid if bond_grid is None else bond_grid
#         shell = self.ref_shell if shell is None else shell
#         r = np.linalg.norm(bond_grid, axis=-1)
#         v1 = self.r_potentials[shell](r)
#         if self.t1_potentials is not None:
#             t1 = np.dot(bond_grid, self.basis[shell][1]@rotation)
#             v1 += self.t1_potentials[shell](t1)
#         if self.t2_potentials is not None:
#             t2 = np.dot(bond_grid, self.basis[shell][2]@rotation)
#             v1 += self.t2_potentials[shell](t2)
#         return v1

#     def V1_gradient(self):
#         """
#         Calculate the gradient of the bonding potential for the reference shell.

#         Returns:
#             dv1 (3, np.ndarray): Gradients of the bonding potential along the long, t1, and t2 directions.
#         """
#         V = self.V1()
#         lo_mesh, t1_mesh, t2_mesh = self.meshes
#         gradient = np.gradient(V, lo_mesh[:, 0, 0], t1_mesh[0, :, 0], t2_mesh[0, 0, :], edge_order=2)
#         return np.array(gradient)
    
#     def V1_vectorized(self, bond_grid=None, shell=None, rotations=None):
#         r = np.linalg.norm(bond_grid, axis=-1)
#         v1 = self.r_potentials[shell](r)
#         if self.t1_potentials is not None:
#             t1 = np.einsum('mnijkl,nl->mnijk', bond_grid, self.basis[shell][1]@rotations)
#             v1 += self.t1_potentials[shell](t1)
#         if self.t2_potentials is not None:
#             t2 = np.einsum('mnijkl,nl->mnijk', bond_grid, self.basis[shell][2]@rotations)
#             v1 += self.t2_potentials[shell](t2)
#         return v1.sum(axis=(0,1))

#     def Vmf_component(self, rotations, alphas, shell=None, eps=1.):
#         """
#         Calculate the bonding potential for a shell along the 'k' direction using symmetry operations.

#         Parameters:
#             shell (int, optional): Shell for which the bonding potential is calculated. Defaults to 0, the reference shell.
#             rotation (3 x 3, optional): Rotation matrix between the reference direction (l=0) and the desired direction (k) in the shell. Defaults to the identity matrix.
#             alpha (3 x 3, optional): Correlation matrix. Defaults to the identity matrix.
#             eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

#         Returns:
#             v1^k (np.ndarray): The bonding potential along the 'k' direction.
#         """
#         shell = self.ref_shell if shell is None else shell
#         b_l = self.bond_grid
#         a_l = self.b_0s[self.ref_shell]*eps*self.basis[self.ref_shell][0]
#         scaled_db = np.einsum('ijkl,mnlp->mnijkl', (b_l-a_l), alphas)  # here, alpha scales the displacement from equilibrium, i.e. db = (b_l-a_l)
#         a_k = self.b_0s[shell]*eps*(self.basis[shell][0]@rotations)
#         b_k = scaled_db+a_k[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, :]
#         return self.V1_vectorized(bond_grid=b_k, shell=shell, rotations=rotations)

#     def Veff(self, eps=1.):
#         """
#         Calculate the correlated mean-field effective potential.

#         Parameters:
#             eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

#         Returns:
#             vmfc (np.ndarray): The correlated mean-field effective potential.
#         """
#         vmfc = 0
#         for shell in range(self.shells):
#             # each shell with a ref bond b_{0r} has its own alphas, which have the shape (no. of shifts, total neighbors, 3, 3)
#             # total neighbors can include 1NN, 2NN, 3NN, etc.
#             # for convenience, we sort the alphas by the shell 
#             # i.e. alphas[shell_0].shape = (no. of shifts, 12, 3, 3), alphas[shell_1].shape = (no. of shifts, 6, 3, 3), etc.
#             # this is done so we can choose the correct symmetry operation (rotation) for each shell
#             vmfc += 0.5*self.Vmf_component(shell=shell, rotations=self.rotations[shell], alphas=self.alphas[shell], eps=eps)
#         return vmfc
    