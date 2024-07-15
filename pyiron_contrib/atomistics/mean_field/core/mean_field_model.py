import numpy as np
import os

from scipy.optimize import root, root_scalar
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.constants import physical_constants
KB = physical_constants['Boltzmann constant in eV/K'][0]

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

def get_anharmonic_F(anharmonic_U, temperatures, n_fine_samples=10000, offset=False):
    """
    Interpolate and integrate the anharmonic internal energies (U) over inverse temperature to obtain the anharmonic free energy (F).

    Parameters:
        anharmonic_U (list/np.ndarray): Anharmonic internal energies.
        temperatures (list/np.ndarray): Array of temperatures.
        n_fine (int, optional): Number of points for the fine temperature interpolation. Defaults to 10000.
        offset (boolean): Whether or not to perform the integration with y=0. for x=x[0] and add the y value post integration.

    Returns:
        tuple: Tuple containing the fine temperatures and the integrated anharmonic free energy.
    """
    # if not np.isclose(temperatures[0], 1.):
    #     ah_U = np.concatenate(([0.], anharmonic_U))
    #     temps = np.concatenate(([1.], temperatures))
    fine_temps = np.linspace(temperatures[0], temperatures[-1], n_fine_samples, endpoint=True)
    if offset:
        off = anharmonic_U[0]
        ah_U = anharmonic_U-anharmonic_U[0]
    else:
        off = 0.
        ah_U = anharmonic_U
    ah_U_eqn = CubicSpline(x=temperatures, y=ah_U)
    ah_F = -cumulative_trapezoid(ah_U_eqn(fine_temps), 1/fine_temps)*fine_temps[1:]
    return CubicSpline(fine_temps[1:], ah_F)(temperatures)+off

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
        bond_grid (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates. Must be defined with respect to the reference shell.
        b_0s (list/np.array): The equilibrium bond length of each shell. Must be of length shells.
        basis (list/np.ndarray): (shells x 3 x 3) array of unit vectors for each shell. Must be of length shells.
        rotations (list/np.ndarray): Rotation matrices which transform the reference bond vector (l=0) to other bond vectors (k) in the same shell, for each shell. Must be of length shells.
        alphas (list/np.ndarray): Correlation matrices \alpha_{pl} between different bond pairs, defined for each shell with respect to a reference bond in that shell. Must be of length shells.
        r_potentials (list of functions): Function for the ij (or 'radial' or 'central') potential (we take this to be the same as the longitudinal (lo) potential). Must be of length shells.
        t1_potentials (list of functions, optional): Function for the t1 potential, if any. Defaults to None. Must be of length shells.
        t2_potentials (list of functions, optional): Function for the t2 potential, if any. Defaults to None.  Must be of length shells.
        ref_shell (int): Reference shell with respect to which the mean-field model is defined. Defaults to 0, the 1st shell.
        shells (int): Number of shells up to which the input b_0s, potentials, rotations, alphas are specified. Defaults to 1, the 1st shell.
    """

    def __init__(self, bond_grid, b_0s, basis, rotations, alphas, alphas_rot_ids, r_potentials, t1_potentials=None, t2_potentials=None, eta=None,
                 r_harm_potentials=None, t1_harm_potentials=None, t2_harm_potentials=None, ref_shell=0, shells=1):
        self.bond_grid = bond_grid
        self.b_0s = b_0s
        self.basis = basis
        self.rotations = rotations
        self.alphas = alphas
        self.alphas_rot_ids = alphas_rot_ids
        self.r_potentials = r_potentials
        self.t1_potentials = t1_potentials
        self.t2_potentials = t2_potentials
        self.eta = eta
        self.r_harm_potentials = r_harm_potentials
        self.t1_harm_potentials = t1_harm_potentials
        self.t2_harm_potentials = t2_harm_potentials
        self.shells = shells
        self.ref_shell = ref_shell

        self.meshes = get_meshes(self.bond_grid, self.basis[ref_shell])
        self.check_inputs()

    def check_inputs(self):
        """
        Check if the inputs are properly defined.
        """
        assert isinstance(self.shells, int), 'shells must be an integer.'
        assert self.shells > 0, 'shells must be greater than 0.'
        assert isinstance(self.ref_shell, int), 'ref_shell must be an integer.'
        assert self.ref_shell < self.shells, 'ref_shell must be less than shells.'
        assert (self.eta is None) or (isinstance(self.eta, (int, float))), 'eta must be None or of type int or float'
              
        # enusre that these inputs have the same lengths
        inputs = [self.b_0s, self.basis, self.rotations, self.r_potentials, self.t1_potentials, self.t2_potentials]
        inputs_str = ['b_0s, basis', 'rotations', 'r_potentials', 't1_potentials', 't2_potentials']

        if self.eta is not None:
            inputs.extend([self.r_harm_potentials, self.t1_harm_potentials, self.t2_harm_potentials])
            inputs_str.extend(['r_harm_potentials', 't1_harm_potentials', 't2_harm_potentials'])
        for inp, inp_str in zip(inputs, inputs_str):
            if len(inp) != self.shells:
                raise TypeError(f'length of {inp_str} is not equal to shells.')

    def V1_template(self, bond_grid=None, shell=None, harmonic=False):
        """
        Calculate the bonding potential for a shell.

        Parameters:
            bond_grid (np.ndarray, optional): Modified bond grid. Defaults to None, uses self.bond_grid.
            shell (int, optional): Shell for which the bonding potential is calculated. Defaults to None, the reference shell.

        Returns:
            v1 (np.ndarray): The bonding potential.
        """
        bond_grid = self.bond_grid if bond_grid is None else bond_grid
        shell = self.ref_shell if shell is None else shell
        
        r_potential  = self.r_harm_potentials[shell]  if harmonic else self.r_potentials[shell]
        t1_potential = self.t1_harm_potentials[shell] if harmonic else self.t1_potentials[shell]
        t2_potential = self.t2_harm_potentials[shell] if harmonic else self.t2_potentials[shell]

        r = np.dot(bond_grid, self.basis[shell][0]) if harmonic else np.linalg.norm(bond_grid, axis=-1)
        # r = np.linalg.norm(bond_grid, axis=-1)
        
        v1 =  r_potential(r)
        v1 += t1_potential(np.dot(bond_grid, self.basis[shell][1])) if t1_potential is not None else 0
        v1 += t2_potential(np.dot(bond_grid, self.basis[shell][2])) if t2_potential is not None else 0

        return v1
    
    def V1(self, bond_grid=None, shell=None):
        v1 = self.V1_template(bond_grid=bond_grid, shell=shell)
        if self.eta is not None:
            v1_harm = self.V1_template(bond_grid=bond_grid, shell=shell, harmonic=True)
            v1_mix = self.eta*v1+(1-self.eta)*v1_harm
            return v1_mix
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
        return gradient
    
    def Veff(self, eps=1.):
        """
        Calculate the mean-field effective potential.

        Parameters:
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

        Returns:
            vmfc (np.ndarray): The correlated mean-field effective potential.
        """
        vmf = 0.
        for shell in range(len(self.alphas)):
            b_l = self.bond_grid
            a_l = self.b_0s[self.ref_shell]*self.basis[self.ref_shell][0]*eps
            a_k = self.b_0s[shell]*self.basis[shell][0]*eps
            for alphas, alphas_rot_ids in zip(self.alphas[shell], self.alphas_rot_ids[shell]):
                # we scale first, rotate next, and then add a_k
                # Note that the rotation is done on db and not a_k! This lets us circumvent the rotation that needs to be made for the transveral components in V1
                # for future optimization.
                for alpha, rot in zip(alphas, self.rotations[shell][alphas_rot_ids]):
                    if len(alpha) != 0:
                        b_k = (b_l-a_l)@alpha@rot.T+a_k
                        vmf += 0.5*self.V1(bond_grid=b_k, shell=shell)
        return vmf - vmf.min()
    
class VirialQuantities():
    """
    Class that calculates the virial quantities.

    Args:
        bond_grid (np.ndarray): (len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates defined with respect to a shell.
        b_0 (list/np.array): The equilibrium bond length of the shell.
        basis (list/np.ndarray): (3 x 3) array of unit vectors of the shell.
        crystal (str): Crystal structure. Defaults to 'fcc'.
    """

    def __init__(self, bond_grid, b_0, basis, crystal='fcc', scale=1.):
        self.bond_grid = bond_grid
        self.b_0 = b_0
        self.basis = basis
        self.crystal = crystal
        self.scale = scale

        self.meshes = get_meshes(self.bond_grid, self.basis)
        self._rho_1_temp = None

        if self.crystal == 'fcc':
            # set the number of bonds per shell.
            self.n_bonds_per_shell = np.array([12, 6, 24, 12, 24, 8])
            # factors to obtain the correct volume from the bond length of each shell
            self.factors = np.array([np.sqrt(2), 1., np.sqrt(2/3), np.sqrt(1/2), np.sqrt(2/5), np.sqrt(1/3)])
        elif self.crystal == 'bcc':
            self.n_bonds_per_shell = np.array([8, 6, 12, 24, 8, 6, 24, 24])
            self.factors = np.array([2/np.sqrt(3), 1., 1/np.sqrt(2), 2/np.sqrt(11), 1/np.sqrt(3), 1/2, 2/np.sqrt(19), 1/np.sqrt(5)])
        else:
            raise TypeError('crystal must be fcc or bcc.')


    def get_rho(self, Veff, temperature=100., eps=1., lm=0.):
        """
        Calculate the bond density from the mean-field effective bonding potential. 
        
        In order to maintain volume invariance, we include a Lagrange multiplier term. This is fitted automatically in the 'nvt' and 'npt' routines 
        (as get_rho is part of their objective functions).

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
            rho_1 = self._rho_1_temp
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
        Veff = np.array(Veff, dtype=float)
        dV1 = np.array(dV1, dtype=float)
        rho_1 = self.get_rho(Veff=Veff, temperature=temperature, eps=eps, lm=lm)
        b_eps = (lo_mesh*rho_1).sum()
        db_dV1 = -((lo_mesh-b_eps)*dV1[0]+t1_mesh*dV1[1]+t2_mesh*dV1[2])
        T_vir = -(db_dV1*rho_1).sum()*self.n_bonds_per_shell[shell]/6/KB*self.scale
        
        if self.crystal == 'fcc':
            N_by_V = 4/(b_eps*self.factors[shell])**3
        elif self.crystal == 'bcc':
            N_by_V = 2/(b_eps*self.factors[shell])**3
        else:
            raise TypeError('crystal must be fcc or bcc.')
        b_dV1 = -(lo_mesh*dV1[0]+t1_mesh*dV1[1]+t2_mesh*dV1[2])
        P_vir = N_by_V*(KB*T_vir+(b_dV1*rho_1).sum()*self.n_bonds_per_shell[shell]/6)

        if not return_rho_1:
            rho_1 = np.array([None])  
        return T_vir, P_vir, b_eps, rho_1
    
class Optimizer():
    """
    Class for NVT and NPT optimizers..
    
    Args:
        project_path (str): Path to the project directory.
        mfm_instances (list): List of MeanFieldModel instances. Must be of length shells.
        vq_instances (list): List of VirialQuantities instances. Must be of length shells.
        energy_list (list/np.array): Energies of the energy-volume curve. Defaults to None, where no correction terms are added.
        strain_list (list/np.array): Strains corresponding to the volumes of the energy-volume curve. Defaults to None, where no correction terms are added.
        rewrite_veff (boolean): Whether or not to rewrite the mean-field effective potential for a particular value of eps. Defaults to False. Ignored for an NPT optimization.
    """
    def __init__(self, project_path, mfm_instances, vq_instances, energy_list=None, strain_list=None, r_order=1, rewrite_veff=False):
        self.project_path = project_path
        self.mfm_instances = mfm_instances
        self.vq_instances = vq_instances
        self.energy_list = energy_list
        self.strain_list = strain_list
        self.r_order = r_order
        
        self.rewrite_veff = rewrite_veff

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
        self.crystal = self.vq_instances[0].crystal
        self.n_bonds_per_shell = self.vq_instances[0].n_bonds_per_shell
        self.factors = self.vq_instances[0].factors
        
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
                    u_mf += 0.5*self.n_bonds_per_shell[shell]*self.mfm_instances[shell].V1(s_b_xyz)
                self._u_mf_fit_eqn = np.poly1d(np.polyfit(strains, u_mf, deg=4))
                self._du_mf_fit_eqn = np.poly1d(np.polyfit(strains, np.gradient(u_mf, strains, edge_order=2), deg=4))

                if self.crystal == 'fcc':
                    self._N_by_V_0 = 4/(self.b_0s[0]*self.factors[0])**3
                elif self.crystal == 'bcc':
                    self._N_by_V_0 = 2/(self.b_0s[0]*self.factors[0])**3
                else:
                    raise TypeError('crystal must be fcc or bcc.')
                
                self._ev_fit_initialized = True

            E_offset = self._u_md_fit_eqn(eps)-self._u_mf_fit_eqn(eps)
            P_offset = -self._N_by_V_0/(3*eps**2)*(self._du_md_fit_eqn(eps)-self._du_mf_fit_eqn(eps))
            return P_offset, E_offset

    def collect_properties(self, temperature, lms, eps, Veffs=None, dV1s=None, fix_T=False, return_rho_1=False):
        """
        Collects the properties of the system.
        
        Parameters:
            temperature (float): Temperature of the system.
            lms (list/np.array): Lagrange multipliers for each shell.
            eps (float): Strain on the system.
            Veffs (list/np.array, optional): Effective potentials for each shell. Defaults to None.
            dV1s (list/np.array, optional): Gradients of the bonding potential for each shell. Defaults to None.
            fix_T (boolean): Whether or not to fix temperature during the optimization routine. Defaults to False.
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
        if fix_T:
            original_temperature = np.sum(T_vir)
        else:
            original_temperature = temperature
        # print('T_eff: {}\nT_vir: {}\nP_vir: {}\nP_offset: {}\nE_offset: {}\neps: {}\nlm: {}\n'.format(temperature, np.array(T_vir), np.array(P_vir), P_offset, E_offset, eps, lms))
        return [original_temperature, temperature, T_vir, P_vir, P_offset, E_offset, b_eps, eps, lms, rho_1s]
    
    def load_Veff(self, shell=0, eps=1., eta=None):
        if eta is not None:
            filename = f'Veff_eta_{eta}_{shell+1}NN_eps_{eps}'.replace('.','_')+'.npy'
        else:
            filename = f'Veff_{shell+1}NN_eps_{eps}'.replace('.','_')+'.npy'
        Veff = np.load(os.path.join(self.project_path, 'resources', filename), allow_pickle=True)
        return Veff

    def run_nvt(self, temperature=100., eps=1., eta=None, return_rho_1=False, fix_T=False):
        """
        Runs the mean-field NVT optimization.
        
        Parameters:
            temperature (float): Input temperature of the system.
            eps (float): Target strain of the system.
            return_rho_1 (boolean): Whether or not to return the bond density.
            fix_T (boolean): Whether or not to fix temperature during the optimization routine. Defaults to False.
        
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
        Veffs = [self.load_Veff(shell=shell, eps=eps, eta=eta) for shell in range(self.r_order)]
        dV1s = [self.mfm_instances[shell].V1_gradient() for shell in range(self.r_order)]
        lms = []
        # print('T: {}\neps: {}\n'.format(temperature, eps))
        if not fix_T:
            for shell in range(self.r_order):
                # print('Optimizing shell {}...'.format(shell))
                def objective_function(args):
                    _, _, b_eps, _ = self.vq_instances[shell].get_virial_quantities(Veff=Veffs[shell], dV1=dV1s[shell], temperature=temperature, eps=eps, lm=args, shell=shell)
                    return np.abs(self.b_0s[shell]*eps-b_eps)
                solver = root_scalar(objective_function, x0=0., x1=0.001, rtol=1e-20)
                # print('Optimization complete.')
                lms.append(solver.root)
            lms = np.array(lms)
        else:
            def objective_function(args):
                T_virs = []
                b_eps_diff = []
                for shell in range(self.r_order):
                    t_vir, _, b_eps, _ = self.vq_instances[shell].get_virial_quantities(Veff=Veffs[shell], dV1=dV1s[shell], temperature=args[-1], eps=eps, lm=args[shell], shell=shell)
                    T_virs.append(t_vir)
                    b_eps_diff.append(np.abs(self.b_0s[shell]-b_eps))
                T_diff = np.abs(np.sum(T_virs)-temperature)
                return np.concatenate((np.array(b_eps_diff), np.array([T_diff])))
            # print('Optimizing all shells at once...')
            x0 = np.concatenate((np.zeros(self.r_order), np.array([temperature])))
            solver = root(objective_function, x0=x0, method='lm', tol=1e-20)
            # print('Optimization complete.')
            lms = solver.x[:-1]
            temperature = solver.x[-1]
            # print('Effective/renormlaized temperature: {}'.format(temperature))
        return self.collect_properties(temperature=temperature, lms=lms, eps=eps, Veffs=Veffs, dV1s=dV1s, fix_T=fix_T, return_rho_1=return_rho_1)
    
    def run_npt(self, temperature=100., pressure=1e-4, eps=1., return_rho_1=False, fix_T=False):
        """
        Runs the mean-field NPT optimization.
        
        Parameters:
            temperature (float): Input temperature of the system.
            pressure (float): Target pressure of the system.
            eps (float): Initial strain of the system.
            return_rho_1 (boolean): Whether or not to return the bond density.
            fix_T (boolean): Whether or not to fix temperature during the optimization routine. Defaults to False.
        
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
        # print('T: {}\nP: {}\n'.format(temperature, pressure))
        # print('Optimizing all shells at once. This might take a while...')
        dV1s = np.array([self.mfm_instances[shell].V1_gradient() for shell in range(self.r_order)]) 
        if not fix_T:
            def objective_function(args):
                P_virs = []
                b_eps_diff = []
                for shell in range(self.r_order):
                    Veff = self.mfm_instances[shell].Veff(eps=args[-1])
                    _, p_vir, b_eps, _ = self.vq_instances[shell].get_virial_quantities(Veff=Veff, dV1=dV1s[shell], temperature=temperature, eps=args[-1], lm=args[shell], shell=shell)
                    P_virs.append(p_vir)
                    b_eps_diff.append(np.abs(self.b_0s[shell]*args[-1]-b_eps))
                P_offset, _ = self.get_epsilon_pressure(eps=args[-1])
                P_diff = np.abs(np.sum(P_virs)+P_offset-pressure)
                print(args)
                return np.concatenate((np.array(b_eps_diff), np.array([P_diff])))
            x0 = np.concatenate((np.zeros(self.r_order), np.array([eps])))
            solver = root(objective_function, x0=x0, method='lm', tol=1e-20)
            # print('Optimization complete.')
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
                    b_eps_diff.append(np.abs(self.b_0s[shell]*args[-1]-b_eps))
                P_offset, _ = self.get_epsilon_pressure(eps=args[-1])
                P_diff = np.abs(np.sum(P_virs)+P_offset-pressure)
                T_diff = np.abs(np.sum(T_virs)-temperature)
                print(args)
                return np.concatenate((np.array(b_eps_diff), np.array([T_diff]), np.array([P_diff])))
            x0 = np.concatenate((np.zeros(self.r_order), np.array([temperature]), np.array([eps])))
            solver = root(objective_function, x0=x0, method='lm', tol=1e-20)
            # print('Optimization complete.')
            lms = solver.x[:-2]
            temperature = solver.x[-2]
            eps = solver.x[-1]
        return self.collect_properties(temperature=temperature, lms=lms, eps=eps, dV1s=dV1s, fix_T=fix_T, return_rho_1=return_rho_1)
    
class GenerateAlphas():
    """
    Class that generates the covariance matrices \alpha_{rpl} between a reference bond r and all the other bonds.

    Args:
        project_path (str): Path to the project directory.
        b_0s (list/np.array): The equilibrium bond length of each shell. Must be of length shells.
        basis (list/np.ndarray): (shells x 3 x 3) array of unit vectors for each shell. Must be of length shells.
        rotations (list/np.ndarray): Rotation matrices which transform the reference bond vector (l=0) to other bond vectors (k) in the same shell, for each shell. Must be of length shells.
        bloch_hessians (list/np.ndarray): Hessians in reciprocal space coordinates computed from the Hessian class.
        kpoint_vectors (list/np.ndarray): Kpoint vectors computed from the Hessian class.
        kpoint_weights (list/np.ndarray): Weights of the kpoint vectors from the Hessian class.
        shells (int): Number of shells up to which the input b_0s, potentials, rotations, alphas are specified.
        r_order (int): Number of shells up to which to compute bond densities for. Defaults to 1, the 1st nearest neighbor shell.
        s_order (int): Number of shifts to consider for each shell. Defaults to 0, shifts up to r_order. 
                       (0 = up to the shell, including all inner shells, 1 = all inner shells, 1 outer shell, 2 = all inner shells, 2 outer shells, etc.)
        crystal (str): Crystal structure. Defaults to 'fcc'.
        cutoff_radius (float): Radius up to which an atom is considered a neighbor. Defaults to 25 Angstroms.
        rewrite_alphas (bool): Whether or not to rewite the alphas (covariance matrices). Defaults to False.
        alpha_threshold (float): Value above which each element of the alpha matrix should be for it to be considered. Defaults to 1e-3.
    """
    def __init__(self, project_path, b_0s, basis, rotations, bloch_hessians, kpoint_vectors, kpoint_weights, shells=1, r_order=1, s_order=0, crystal='fcc',  
                 cutoff_radius=25., rewrite_alphas=False, alpha_threshold=1e-3):
        self.project_path = project_path
        self.b_0s = b_0s
        self.basis = basis
        self.rotations = rotations
        self.shells = shells
        self.r_order = r_order
        self.s_order = s_order
        self.bloch_hessians = bloch_hessians
        self.kpoint_vectors = kpoint_vectors
        self.kpoint_weights = kpoint_weights
        self.crystal = crystal
        self.cutoff_radius = cutoff_radius
        self.rewrite_alphas = rewrite_alphas
        self.alpha_threshold = alpha_threshold

        self.alphas = None
        self.alphas_rot_ids = None

        self.set_n_bonds()

    @staticmethod
    def count_decimals(number):
        # Convert the number to a float to handle scientific notation
        float_repr = float(number)
        # Convert the float to a string in standard notation
        str_repr = f"{float_repr:.20f}"
        # Split the string into whole number and decimal parts
        parts = str_repr.split('.')
        if len(parts) == 2:
            decimal_part = parts[1].rstrip('0')  # Remove trailing zeros
            return len(decimal_part)
        else:
            return 0

    def load_or_run(self):
        """
        Whether or not to load existing alphas or calculate new ones.
        """
        if not os.path.exists(os.path.join(self.project_path, 'resources')):
            os.mkdir(os.path.join(self.project_path, 'resources'))
        exists = [os.path.exists(os.path.join(self.project_path, 'resources', file)) for file in ['alphas.npy', 'alphas_rotation_ids.npy']]
        self.alphas = np.load(os.path.join(self.project_path, 'resources', 'alphas.npy'), allow_pickle=True).tolist() if exists[0] else None
        self.alphas_rotation_ids = np.load(os.path.join(self.project_path, 'resources', 'alphas_rotation_ids.npy'), allow_pickle=True).tolist() if exists[1] else None
        if (not np.all(exists)) or (self.rewrite_alphas):
            self.make_model()
            self.generate_alphas(write=True)
        else:
            print('Loading existing alphas')

    def set_n_bonds(self):
        """
        Set the number of bonds based on the crystal structure.
        """
        # factors to get primitive cells from the basis
        if self.crystal == 'fcc':
            # set the number of bonds per shell.
            self.n_bonds_per_shell = np.array([12, 6, 24, 12, 24, 8])
        elif self.crystal == 'bcc':
            self.n_bonds_per_shell = np.array([8, 6, 12, 24, 8, 6, 24, 24])
        else:
            raise ValueError('crystal must be fcc or bcc')
        
    def make_model(self):
        """
        Helper method to calulate the required terms for the covariance matrices.
        """
        # Iterate over all Bloch modes
        nu_sq, eig_vecs = [], []
        for H in self.bloch_hessians:
            eval, evec = np.linalg.eig(H)
            nu_sq.append(eval)
            eig_vecs.append(evec.T)
        eig_vecs = np.array(eig_vecs).reshape((-1, 3))
        
        self.nu_sq = np.array(nu_sq).flatten()
        nu_sq_inv = 1/self.nu_sq
        
        eig_vecs_outer = np.einsum('ij,ik->ijk', eig_vecs, eig_vecs)
        self.nu_factor = np.einsum('i,ijk->ijk', nu_sq_inv, eig_vecs_outer)
    
    def get_covariance(self, bond, ref_bond, position):
        """
        Calculate covariance matrix \alpha_{rpl}.

        Parameters:
            bond (np.array): The bond vector k of shape (1, 3).
            ref_bond (np.array): The reference bond vector l of shape (1, 3).
            position (np.array): Coordinates of shape (1, 3) of the atom from where the bonds are evaluated.

        Returns:
            covariance matrix \alpha_{rpl} (np.ndarray): The covariance matrix of shape (3, 3).
        """
        kps = self.kpoint_vectors.repeat(3, axis=0)
        kpsw = self.kpoint_weights.repeat(3, axis=0)
        # we match terms k,-k as this is the noise correlation
        r_p = np.exp(1j*kps@position)+np.exp(1j*kps@(position+bond-ref_bond))-np.exp(1j*kps@(position-ref_bond))-np.exp(1j*kps@(position+bond))
        sel = self.nu_sq > self.nu_sq.max()*1e-3
        return np.einsum('i,i,ijk->jk', kpsw[sel]**2, r_p[sel], self.nu_factor[sel])/self.kpoint_weights.sum()
        
    def generate_alphas(self, write=True):
        """
        Generate the covariance matrices \alpha_{plr} between different bond pairs, defined for a shell with respect to a reference bond in that shell.

        Parameters:
            write (boolean): Whether or not to write the alphas. Defaults to True.
        """
        print('Generating alphas...')

        self._nn_positions = []
        for s in range(self.shells):
            self._nn_positions.append(self.basis[s][0]@self.rotations[s]*self.b_0s[s])

        self.alphas = []
        self.alphas_rotation_ids = []
        for shell in range(self.r_order):
            ref_bond = self._nn_positions[shell][0]
            ref_bond_cov = self.get_covariance(bond=ref_bond, ref_bond=ref_bond, position=np.array([0.0, 0.0, 0.0]))

            upto_shell = min(self.shells, shell+self.s_order+1)
            if (self.crystal == 'bcc') and (shell in [0]) and (self.s_order==0):
                upto_shell = min(self.shells, shell+self.s_order+2)
            positions = np.vstack((np.zeros((1, 3)), *self._nn_positions[:upto_shell]))

            alphas = [[] for _ in range(upto_shell)]
            alphas_rotation_ids = [[] for _ in range(upto_shell)]
            for shl, nn in enumerate(self._nn_positions[:upto_shell]):
                al_p = []
                al_p_ids = []
                for pos in positions:
                    al = []
                    al_ids = []
                    within_radius = np.argwhere(np.round(np.linalg.norm(pos+nn, axis=-1), decimals=5) <= 
                                                np.round(self.cutoff_radius, decimals=5)).flatten()
                    for b, bond in zip(within_radius, nn[within_radius]):
                        bond_cov = self.get_covariance(bond=bond, ref_bond=ref_bond, position=pos)
                        norm_bond_cov = np.real(bond_cov@np.linalg.inv(ref_bond_cov))
                        if np.round(abs(norm_bond_cov), decimals=self.count_decimals(self.alpha_threshold)).max() >= self.alpha_threshold:
                            al.append(norm_bond_cov)
                            al_ids.append(b)
                    al_p.append(al)
                    al_p_ids.append(al_ids)
                alphas[shl] = al_p
                alphas_rotation_ids[shl] = al_p_ids
            self.alphas.append(alphas)
            self.alphas_rotation_ids.append(alphas_rotation_ids)
            print('{}NN alphas generated'.format(shell+1))
        if write:
            np.save(os.path.join(self.project_path, 'resources', 'alphas.npy'), np.array(self.alphas, dtype=object))
            np.save(os.path.join(self.project_path, 'resources', 'alphas_rotation_ids.npy'), np.array(self.alphas_rotation_ids, dtype=object))
            print('Alphas saved')
    
class MeanFieldJob():
    """
    User class for running the mean-field anharmonic bond (MAB) model for the NPT or NVT ensemble.
        
    Args:
        project_path (str): Path to the project directory.
        bond_grids (list/np.ndarray): (shells, len(lo_disps), len(t1_disps), len(t2_disps), 3) array of bond vectors in Cartesian xyz coordinates for each shell. Must be of length shells.
        b_0s (list/np.array): The equilibrium bond length of each shell. Must be of length shells.
        basis (list/np.ndarray): (shells x 3 x 3) array of unit vectors for each shell. Must be of length shells.
        rotations (list/np.ndarray): Rotation matrices which transform the reference bond vector (l=0) to other bond vectors (k) in the same shell, for each shell. Must be of length shells.
        bloch_hessians (list/np.ndarray): Hessians in reciprocal space coordinates computed from the Hessian class.
        kpoint_vectors (list/np.ndarray): Kpoint vectors computed from the Hessian class.
        kpoint_weights (list/np.ndarray): Weights of the kpoint vectors from the Hessian class.
        r_potentials (list of functions): Function for the ij (or 'radial' or 'central') potential (we take this to be the same as the longitudinal (lo) potential). Must be of length shells.
        t1_potentials (list of functions, optional): Function for the t1 potential, if any. Defaults to None. Must be of length shells.
        t2_potentials (list of functions, optional): Function for the t2 potential, if any. Defaults to None.  Must be of length shells.
        shells (int): Number of nearest neighbor shells to consider for the model. Defaults to 1 shell, the 1st nearest neighbor shell.
        r_order (int): Number of shells up to which to compute bond densities for. Defaults to 1, the 1st nearest neighbor shell.
        s_order (int): Number of shifts to consider for each shell. Defaults to 0, shifts up to r_order. 
                       (0 = up to the shell, including all inner shells, 1 = all inner shells, 1 outer shell, 2 = all inner shells, 2 outer shells, etc.).
                       If r_order+s_order > shells, only consider up to the last available shell.
        crystal (str): Crystal structure. Currently only supports 'fcc' and 'bcc'.
        energy_list (list/np.array): Energies of the energy-volume curve. Defaults to None, where no correction terms are added.
        strain_list (list/np.array): Strains corresponding to the volumes of the energy-volume curve. Defaults to None, where no correction terms are added.
        cutoff_radius (float): Radius up to which an atom is considered a neighbor. Defaults to 25 Angstroms.
        rewrite_alphas (bool): Whether or not to rewite the alphas (covariance matrices). Defaults to False.
        alpha_threshold (float): Value above which each element of the alpha matrix should be for it to be considered. Defaults to 1e-3.
    """

    def __init__(self, project_path, bond_grids, b_0s, basis, rotations, bloch_hessians, kpoint_vectors, kpoint_weights, r_potentials, t1_potentials=None, t2_potentials=None, 
                 r_harm_potentials=None, t1_harm_potentials=None, t2_harm_potentials=None, shells=1, r_order=1, s_order=0, crystal='fcc', energy_list=None, 
                 strain_list=None, cutoff_radius=25., rewrite_alphas=False, alpha_threshold=1e-3):
        self.project_path = project_path
        self.bond_grids = bond_grids
        self.b_0s = b_0s
        self.basis = basis
        self.rotations = rotations
        self.bloch_hessians = bloch_hessians
        self.kpoint_vectors = kpoint_vectors
        self.kpoint_weights = kpoint_weights
        self.r_potentials = r_potentials
        self.t1_potentials = t1_potentials
        self.t2_potentials = t2_potentials
        self.r_harm_potentials = r_harm_potentials
        self.t1_harm_potentials = t1_harm_potentials
        self.t2_harm_potentials = t2_harm_potentials
        self.shells = shells
        self.r_order = r_order
        self.s_order = s_order
        self.crystal = crystal
        self.energy_list = energy_list
        self.strain_list = strain_list
        self.cutoff_radius = cutoff_radius
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

        if (self.energy_list is not None) and (self.strain_list is not None):
            assert len(self.energy_list) == len(self.strain_list), 'length of energy_list and strain_list must be equal.'
            self.energy_list = np.array(self.energy_list)
            self.strain_list = np.array(self.strain_list)

    def generate_alphas(self):
        """
        Generate the correlation matrices \alpha_{rpl} between different bond pairs, defined for a shell with respect to a reference bond in that shell.
        """
        alp = GenerateAlphas(project_path=self.project_path, 
                             b_0s=self.b_0s, 
                             basis=self.basis, 
                             rotations=self.rotations,
                             bloch_hessians=self.bloch_hessians, 
                             kpoint_vectors=self.kpoint_vectors,
                             kpoint_weights=self.kpoint_weights, 
                             shells=self.shells, 
                             r_order=self.r_order, 
                             s_order=self.s_order, 
                             crystal=self.crystal, 
                             cutoff_radius=self.cutoff_radius,
                             rewrite_alphas=self.rewrite_alphas,
                             alpha_threshold=self.alpha_threshold)
        alp.load_or_run()
        self.alphas = alp.alphas
        self.alphas_rotation_ids = alp.alphas_rotation_ids
        self.n_bonds_per_shell = alp.n_bonds_per_shell
    
    def initialize_instances(self, eta=None):
        """
        Initialize MeanFieldModel and VirialQuantities instances for each shell.
        """
        # print('Initializing instances...')
        self._mfms = []
        self._vqs = []
        for r in range(self.r_order):
            self._mfms.append(MeanFieldModel(bond_grid=self.bond_grids[r], 
                                             b_0s=self.b_0s, 
                                             basis=self.basis, 
                                             rotations=self.rotations, 
                                             alphas=self.alphas[r], 
                                             alphas_rot_ids=self.alphas_rotation_ids[r], 
                                             ref_shell=r, 
                                             shells=self.shells, 
                                             r_potentials=self.r_potentials, 
                                             t1_potentials=self.t1_potentials, 
                                             t2_potentials=self.t2_potentials,
                                             eta=eta,
                                             r_harm_potentials=self.r_harm_potentials,
                                             t1_harm_potentials=self.t1_harm_potentials,
                                             t2_harm_potentials=self.t2_harm_potentials)
                                             )
            self._vqs.append(VirialQuantities(bond_grid=self.bond_grids[r], 
                                              b_0=self.b_0s[r], 
                                              basis=self.basis[r], 
                                              crystal=self.crystal,
                                              scale=1.)
                                              )
        # print('Instances initialized.')

    def create_Veffs(self, eps=1., eta=None, rewrite_veff=False):
        """
        Create and save mean-field effective potentials in advance for constant volume calculations.

        Parameters:
            eps (float): Target strain of the system. Defaults to None.
            rewrite_veff (boolean): Whether or not to rewrite the mean-field effective potential for a particular value of eps. Defaults to False.
        """
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)
        for shell in range(self.r_order):
            if eta is not None:
                filename = f'Veff_eta_{eta}_{shell+1}NN_eps_{eps}'.replace('.','_')+'.npy'
            else:
                filename = f'Veff_{shell+1}NN_eps_{eps}'.replace('.','_')+'.npy'
            exists = os.path.exists(self.project_path, filename)
            if not exists or rewrite_veff:
                print('Building Veff for {}NNs at epsilon {}...'.format(shell+1, eps))
                Veff = self._mfms[shell].Veff(eps=eps)
                np.save(os.path.join(self.project_path, 'resources', filename), np.array(Veff, dtype=object))
                print('Veff for {}NNs at epsilon {} saved'.format(shell+1, eps))

    @staticmethod
    def check_temperatures(temperatures):
        if isinstance(temperatures, (float, int)):
            temperatures = np.array([temperatures])
        elif isinstance(temperatures, (list, np.ndarray)):
            temperatures = np.array(temperatures)
        else:
            raise AssertionError('temperatures should either be of type int, float, list or a numpy array')
        assert np.all(temperatures > 0.), 'temperature must be greater than 0.'
        return temperatures
    
    @staticmethod
    def check_pressure_eps(pressure, eps):
        if pressure is None:
            assert eps is not None, 'eps must be specified if pressure is None.'
            assert isinstance(eps, (int, float)), 'eps must be an integer an integer or float.'
            # print('Pressure not specified, running NVT model... ')
        elif pressure is not None:
            assert isinstance(pressure, (int, float)), 'pressure must be an integer or float.'
            if eps is None:
                eps = 1.0
            # print('Pressure specified, setting intial eps = {}, running NPT model...'.format(eps))
        else:
            raise ValueError('either pressure or eps must be specified.')
        return pressure, eps

    def run_ensemble(self, temperatures=100., pressure=None, eps=None, eta=None, return_output=False, fix_T=False, return_rho_1s=False, rewrite_veff=False):
        """
        Optimize the mean-field model parameters to simulate an NPT or NVT simulation. Either pressure or eps must be specified. If pressure is specified, run the NPT model. 
        If eps is specified, run the NVT model.

        Parameters:
            temperatures (int, float, list): Input temperatures for which the mean-field model is evaluated. Defaults to 100. K.
            pressure (float): Target pressure of the system. Defaults to None.
            eps (float): Target strain of the system. Defaults to None.
            return_output (boolean): Whether or not to return the thermo output of the model. Defaults to False.
            fix_T (boolean): Whether or not to fix temperature during the optimization routine. Defaults to False.
            return_rho_1s (boolean): Whether or not to return the bond densities of the shell. Defaults to False.
            rewrite_veff (boolean): Whether or not to rewrite the mean-field effective potential for a particular value of eps. Defaults to False. Ignored for an NPT optimization.
        Returns:
            output (dict): Dictionary containing the output of the mean field job, if return_output is True.
        """
        temperatures = self.check_temperatures(temperatures=temperatures)
        pressure, eps = self.check_pressure_eps(pressure=pressure, eps=eps)

        self.initialize_instances(eta=eta)
        opt = Optimizer(project_path=self.project_path, mfm_instances=self._mfms, vq_instances=self._vqs, energy_list=self.energy_list, strain_list=self.strain_list, r_order=self.r_order, 
                        rewrite_veff=rewrite_veff)
        if pressure is None:
            self.create_Veffs(eps=eps, eta=eta, rewrite_veff=rewrite_veff)
        
        # if fix_T:
        #     print('Temperature will be optimized.')
            
        outs = []
        for temp in temperatures:
            if pressure is None:
                out = opt.run_nvt(temperature=temp, eps=eps, eta=eta, return_rho_1=True, fix_T=fix_T)
            elif pressure is not None:
                out = opt.run_npt(temperature=temp, pressure=pressure, eps=eps, return_rho_1=True, fix_T=fix_T)
            else:
                raise ValueError('either pressure or eps must be specified.')
            
            if eta is not None:
                per_atom_dU_deta = self.get_anharmonic_U(eta=eta, out=out, return_components=True)
                # print(f'eta: {eta}\n')
                # print(f'per_atom_dU_deta: {per_atom_dU_deta*1000} meV/atom\n')
                out.append(per_atom_dU_deta*1000)
            else:
                ah_U, per_shell_ah_U = self.get_anharmonic_U(out=out, return_components=True)
                # print(f'per_shell_ah_U: {per_shell_ah_U*1000} meV/atom\n')
                # print(f'ah_U: {ah_U*1000} meV/atom\n')
                out.append((per_shell_ah_U*1000).tolist())
                out.append(ah_U*1000)
            
            if not return_rho_1s:
                out[9] = None

            outs.append(out)
        
        self.output = {
            key: [o[i] for o in outs] 
            for i, key in enumerate(['T', 'T_eff', 'T_vir', 'P', 'P_offset', 'E_offset', 'b_eps', 'eps', 'lms'])
            }
        self.rho_1s = [o[9] for o in outs]

        if eta is not None:
            self.output['per_atom_dU_deta'] = [o[10] for o in outs]
        else:
            self.output['per_shell_ah_U'] = [o[10] for o in outs]
            self.output['ah_U'] = [o[11] for o in outs]
        
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
    
    def get_anharmonic_U(self, out, eta=None, return_components=False):
        """
        Get the anharmonic internal energy in [eV/atom] estimated by the model.

        Returns:
            ah_U (float): Anharmonic internal energy.
            return_components (boolean): Whether or not to return the anharmonic internal energies per shell. Defaults to False.
        """
        # assert self.output is not None, 'run_ensemble() must be run first.'
        T_vir = out[2]
        P_offset = out[4]
        E_offset = out[5]
        rho_1s = out[9]
        
        per_bond_energy = np.array([(self._mfms[s].V1_template()*rho_1s[s]).sum() for s in range(self.r_order)])
        if eta is not None:
            per_bond_energy -= np.array([(self._mfms[s].V1_template(harmonic=True)*rho_1s[s]).sum() for s in range(self.r_order)])
            per_atom_dU_deta = self.n_bonds_per_shell[:self.r_order]/2*per_bond_energy
            return per_atom_dU_deta.sum()
        else:
            per_atom_energy = self.n_bonds_per_shell[:self.r_order]/2*per_bond_energy
            per_shell_ah_U = per_atom_energy-1.5*KB*np.array(T_vir)
            ah_U = per_shell_ah_U.sum()
            if not np.isclose(P_offset, 0.):
                ah_U += E_offset
            if return_components:
                return ah_U, per_shell_ah_U
            return ah_U
    