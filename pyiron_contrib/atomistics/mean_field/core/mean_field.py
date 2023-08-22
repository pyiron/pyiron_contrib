# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar, root
from scipy.integrate import cumtrapz
from scipy.signal import argrelextrema
from scipy.constants import physical_constants
KB = physical_constants['Boltzmann constant in eV/K'][0]
   
class MeanField():
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
    - rotations (list): The rotation matrices which transform the reference bond vector (l=0) to the bond vectors in the same shell.
    - r_potential (callable): Function for the ij (or 'radial' or 'central') potential (we take this to be the same as the longitudinal potential).
    - t1_potential (callable): Function for the t1 potential, if any. Defaults to None.
    - t2_potential (callable): Function for the t2 potential, if any. Defaults to None.
    - energy_offset (float): A constant offset to the bonding potential. Defaults to 0.
    - pressure_offset (float): A constant offset to the pressure. Defaults to 0.
    """
    
    def __init__(self, b_0, basis, rotations, r_potential, t1_potential=None, t2_potential=None, energy_offset=0, pressure_offset=0):
        self.b_0 = b_0
        self.basis = np.array(basis)
        self.rotations = rotations
        self.r_potential = r_potential
        self.t1_potential = t1_potential
        self.t2_potential = t2_potential
        self.energy_offset = energy_offset
        self.pressure_offset = pressure_offset
        self._rho_1s = None
        self._dV1 = None
        
    def get_meshes(self, bonds):
        """
        Generate meshes based on the given bond vectors.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.

        Returns:
            long_mesh (ndarray): (n_long x n_t1 x n_t2) mesh generated using the longitudinal (long) basis vector.
            t1_mesh (ndarray): (n_long x n_t1 x n_t2) mesh generated using the 1st transversal (t1) basis vector.
            t2_mesh (ndarray): (n_long x n_t1 x n_t2) mesh generated using the 2nd transversal (t2) basis vector.
        """
        long_mesh = np.dot(bonds, self.basis[0])
        t1_mesh = np.dot(bonds, self.basis[1])
        t2_mesh = np.dot(bonds, self.basis[2])
        return long_mesh, t1_mesh, t2_mesh
        
    def V1(self, bonds, rotation=np.eye(3)):
        """
        Calculate the bonding potential.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            rotation (ndarray, optional): Rotation matrix. Defaults to the identity matrix.

        Returns:
            v1 (ndarray): The bonding potential.
        """
        r = np.linalg.norm(bonds, axis=-1)
        v1 = self.r_potential(r)-self.r_potential(self.b_0)
        if self.t1_potential is not None:
            t1 = np.dot(bonds, self.basis[1]@rotation)
            v1 += self.t1_potential(t1)
        if self.t2_potential is not None:
            t2 = np.dot(bonds, self.basis[2]@rotation)
            v1 += self.t2_potential(t2)
        return v1
        
    def dV1(self, bonds, meshes):
        """
        Calculate the gradient of the bonding potential.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.

        Returns:
            gradients (ndarray): Gradients of the bonding potential along the long, t1 and t2 directions.
        """
        V = self.V1(bonds=bonds)
        long_mesh, t1_mesh, t2_mesh = meshes
        gradients = np.gradient(V, long_mesh[:, 0, 0], t1_mesh[0, :, 0], t2_mesh[0, 0, :], edge_order=2)
        return gradients

    def Vmf_component(self, bonds, rotation, eps=1.):
        """
        Calculate the component 'l' of the mean-field effective potential: The bonds will be rotated according to the rotation provided and passed to self.V1 to generate the compoment.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            rotation (ndarray): Rotation matrix between the reference bond (l=0) and the desired bond (l) in the shell.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

        Returns:
            vmf_component (ndarray): The 'l'th component of the mean-field effective potential.
        """
        b_xyz = self.b_0*self.basis[0]
        rotated_bonds = bonds-b_xyz*eps+(self.basis[0]@rotation)*self.b_0*eps
        vmf_component = self.V1(rotated_bonds, rotation=rotation)
        return vmf_component

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
        for rot in self.rotations:
            vmf += 0.5*self.Vmf_component(bonds=bonds, rotation=rot, eps=eps)
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
        for i, rot in enumerate(self.rotations):
            factor = 1. if (i == 0) else 0.5 if (i != 1) else 0.
            vmfc += factor*self.Vmf_component(bonds, rotation=rot, eps=eps)
        return vmfc
    
    def find_linear_correction(self, Veff, long_mesh, temperature=100., eps=1.):
        """
        Find the linear correction factor (Lagrange multiplier) for the mean-field effective potential. The Lagrange multiplier constrains the single bond density to give the correct bond, thus preserving volume.

        Parameters:
            V_eff (float): Mean-field effective potential.
            long_mesh (ndarray): The longitudinal mesh.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.

        Returns:
            lm (float): Linear correction factor (Lagrange multiplier).
        """
        aa = self.b_0*eps
        def f(m):
            rho = brho*np.exp(-m*(long_mesh-aa)/temperature/KB)
            res = np.abs(np.log((rho*long_mesh).sum()/rho.sum()/aa)).sum()
            return res
        brho = np.exp(-(Veff-Veff.min())/temperature/KB)
        solver = root_scalar(f, x0=0., x1=0.001, rtol=1e-10)
        lm = solver.root
        return lm
    
    def get_rho(self, bonds, long_mesh, temperature=100., eps=1., lm=0., Veff=None):
        """
        Calculate the bond density from the effective bonding potential. 
        
        In order to maintain volume invariance, we include a Lagrange multiplier scalar term. This is fitted automatically in the 'nvt' and 'npt' routines (as get_rho is part of their objective functions).

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            lm (float, optional): Lagrange multiplier. Defaults to 0.
            Veff (float, optional): The mean-field effective potential. If not provided, it is calculated internally.

        Returns:
            rho (ndarray): Normalized single bond density.
        """
        if Veff is None:
            Veff = self.Vmfc(bonds=bonds, eps=eps)
        Veff -= Veff.min()
        rho_1 = np.exp(-(Veff+lm*(long_mesh-self.b_0*eps))/KB/temperature)
        return rho_1/rho_1.sum()
    
    def fix_Veff(self, meshes, Veff, temperature=100., eps=1., fix=[1, 1, 1]):
        """
        Upon analysis of the MD bond density we observe 2 things:
        
        1. The t1 and t2 potentials are theoretically not equivalent to each other, as t1 is along the 1NN direction and t2 along the 2NN direction. This inequivalence is included in the mean-field effective potential. However, we find from the effective potential generated from the MD bond density that for the 'effective' potential, t1 and t2 are equivalent.
        
        2. The 'attractive' components of the effective potential in the transversal and longitudinal directions closely resemble each other. The 'repulsive' components in the transversal directions mirror their attractive components (since the effective potential is harmonic in the transversal directions).
        
        3. It is also the case that the 'attractive' longitudinal component of the mean-field effective potetnial develops an inflection point as it scales with temperature.
        
        To address these observations, we apply fixes to the mean-field effective potential.
        
        Fix 1, 2 and 3 address points 1., 2. and 3. respectively.
        
        The fixes are applied to the effective potential with the volume constraint (Lagrange multiplier term) pre-imposed. 
        
        For fix 3, we fit a CubicSpline function up to the inflection point of the gradient of the effective potential. Beyond this point, the gradient is kept constant at the value of the inflection point (of the gradient).
        
        For fix 2, we fit a 2nd order polynomial to the transversal component upto the inflection point determined in fix 3 (even if fix 3 is False). Not doing this severly overestimates the transversal component as the longitudinal component has that inflection point. 

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            rho_1 (ndarray): The normalized bond density.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            fix (list of booleans): Default [1, 1, 1] correspond to fixes 1, 2 and 3 respectively. Fixes 2 and 3 assume fix 1 to be True.

        Returns:
            new_rho_1 (ndarray): 'Fixed' bond density.
        """
        if not np.any(fix):
            new_Veff = Veff
        else:
            long_mesh, t1_mesh, t2_mesh = meshes
            d_l = int(long_mesh.shape[0]/2)
            d_t = int(t2_mesh.shape[2]/2) if (t2_mesh[0, 0, :][0] < 0.) else 0

            # determine the inflection point of the gradient of the effective potential
            long_grad = np.gradient(Veff[:, d_t, d_t], long_mesh[:, d_t, d_t], edge_order=2)
            long_grad_fit = CubicSpline(long_mesh[:, d_t, d_t], long_grad)
            fine_long = np.linspace(long_mesh[:, d_t, d_t][0], long_mesh[:, d_t, d_t][-1], 10000)
            cutoff = fine_long[argrelextrema(long_grad_fit(fine_long), np.greater, order=5)[0][0]]-self.b_0*eps
            db = long_mesh[:, d_t, d_t]-self.b_0*eps

            if fix[2]:
                sel_1 = (db <= cutoff)
                trunc_grad = np.gradient(Veff[:, d_t, d_t][sel_1], long_mesh[:, d_t, d_t][sel_1], edge_order=2)
                long_fit = CubicSpline(long_mesh[:, d_t, d_t][sel_1], Veff[:, d_t, d_t][sel_1], 
                                       bc_type=((1, trunc_grad[0]), (1, trunc_grad[-1])))
            else:
                long_fit = CubicSpline(long_mesh[:, d_t, d_t], Veff[:, d_t, d_t])

            if fix[1]:
                sel_2 = ((db >= 0.) & (db <= cutoff))
                t_db = np.concatenate((-np.flip(db[sel_2]), db[sel_2]))
                t_pot = np.concatenate((np.flip(Veff[:, d_t, d_t][sel_2]), Veff[:, d_t, d_t][sel_2]))
                t_fit = np.poly1d(np.polyfit(t_db, t_pot-t_pot.min(), deg=2))
            elif fix[0]:
                t_fit = CubicSpline(t2_mesh[d_l, d_t, :][np.isfinite(Veff[d_l, d_t, :])], 
                                    Veff[d_l, d_t, :][np.isfinite(Veff[d_l, d_t, :])])
            new_Veff = long_fit(long_mesh)+t_fit(t1_mesh)+t_fit(t2_mesh)
        
        return new_Veff

    def find_virial_quant(self, bonds, meshes, temperature=100., eps=1., lm=0., fix=[1, 1, 1], Veff=None, return_rho_1=False):
        """
        Calculate virial temperature, pressure and also the equilibrium bond at the given strain.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            lm (float, optional): Lagrange multiplier. Defaults to 0.
            fix (list of booleans): Default [1, 1, 1] correspond to fixes 1, 2 and 3 respectively. Fixes 2 and 3 assume fix 1 to be True.
            return_rho_1 (boolean): Whether or not to return the bond density.

        Returns:
            T_vir (float): Virial temperature of the mean-field model.
            P_vir (float): Virial pressure of the mean-field model.
            b_eps (float): Equilibrium bond at the given strain.
        """
        long_mesh, t1_mesh, t2_mesh = meshes
        rho_1 = self.get_rho(bonds=bonds, long_mesh=long_mesh, temperature=temperature, eps=eps, lm=lm, Veff=Veff)
        if fix:
            Veff_w_lm = -KB*temperature*np.log(rho_1)
            fixed_Veff = self.fix_Veff(meshes=meshes, temperature=temperature, eps=eps, fix=fix, Veff=Veff_w_lm)
            rho_1 = self.get_rho(bonds=bonds, long_mesh=long_mesh, temperature=temperature, Veff=fixed_Veff, eps=eps, lm=0.)
        b_eps = (long_mesh*rho_1).sum()
        dV = self._dV1
        db_dV = (long_mesh-b_eps)*dV[0]+t1_mesh*dV[1]+t2_mesh*dV[2]
        a_dV = b_eps*dV[0]+(t1_mesh*rho_1).sum()*dV[1]+(t2_mesh*rho_1)*dV[2]
        N_by_V = 4./(b_eps*np.sqrt(2))**3 
        T_vir = 2./KB*(db_dV*rho_1).sum()
        P_vir = -2.*N_by_V*(a_dV*rho_1).sum()+self.pressure_offset
        if return_rho_1:
            return T_vir, P_vir, b_eps, rho_1
        return T_vir, P_vir, b_eps
    
    def run_nvt(self, bonds, temperature=100., eps=1., fix=[1, 1, 1], fix_T=False):
        """
        Consider the NVT case. If strain (eps) != 1., then the structure is strained, and the calculation is done at this 'strained' bond.
        The optimization converges the virial temperature to the target temperature and bond length (b_0*eps) to the target bond length.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            temperature (float, optional): Temperature value. Defaults to 100.
            eps (float, optional): Strain on bond b_0. Defaults to 1, no strain.
            fix (list of booleans): Default [1, 1, 1] correspond to fixes 1, 2 and 3 respectively. Fixes 2 and 3 assume fix 1 to be True.

        Returns:
            rho_1 (ndarray): Normalized bond density.
            eps (float, optional): Strain on bond b_0.
            lm (float): Lagrange multiplier.root_scalar(f, x0=0., x1=0.001, rtol=1e-10)
        """
        Veff = self.Vmfc(bonds=bonds, eps=eps)
        meshes = self.get_meshes(bonds)
        self._dV1 = self.dV1(bonds=bonds, meshes=meshes)
        if fix_T:
            def virial(args):
                T_vir, _, b_eps = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=args[0], eps=eps, lm=args[1], 
                                                         fix=fix, Veff=Veff)
                return [abs(temperature-T_vir), np.abs(self.b_0*eps-b_eps)]
            solver = root(virial, x0=(temperature, 0.), method='lm', tol=1e-10)
            eff_temp, lm = solver.x
        else:
            def virial(args):
                _, _, b_eps = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=temperature, eps=eps, lm=args, 
                                                     fix=fix, Veff=Veff)
                return np.abs(self.b_0*eps-b_eps)
            solver = root_scalar(virial, x0=0., x1=0.001, rtol=1e-10)
            eff_temp, lm = temperature, solver.root
        virial_q = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=eff_temp, eps=eps, lm=lm, fix=fix, 
                                          Veff=Veff, return_rho_1=True)
        print('T: {}\nT_eff: {}\nT_vir: {}\nP_vir: {}\neps: {}\nlm: {}\n'.format(temperature, eff_temp, *virial_q[:2], eps, lm))
        return *virial_q, eff_temp, eps, lm
    
    def run_npt(self, bonds, temperature=100., pressure=1e-4, fix=[1, 1, 1], fix_T=False):
        """
        Consider the NPT case. The optimization converges the virial temperature to the target temperature, the virial pressure to the target pressure and the bond length (b_0*eps) to the target bond length. The converged value of 'eps' is then the strain at the target temperture and pressure. 

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            temperature (float, optional): Temperature value. Defaults to 100.
            pressure (float, optional): Pressure value. Defaults to 1e-4 (1 atm).
            fix (list of booleans): Default [1, 1, 1] correspond to fixes 1, 2 and 3 respectively. Fixes 2 and 3 assume fix 1 to be True.

        Returns:
            rho_1 (ndarray): Normalized bond density.
            eps (float, optional): Strain on bond b_0 at the target temperature and pressure.
            lm (float): Lagrange multiplier.
        """
        meshes = self.get_meshes(bonds)
        self._dV1 = self.dV1(bonds=bonds, meshes=meshes)
        if fix_T:
            def virial(args):
                T_vir, P_vir, b_eps = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=args[0], eps=args[1], 
                                                             lm=args[2], fix=fix)
                return [abs(temperature-T_vir), abs(pressure-P_vir), np.abs(self.b_0*args[1]-b_eps)]
            solver = root(virial, x0=(temperature, 1., 0.), method='lm', tol=1e-10)
            eff_temp, eps, lm = solver.x
        else:
            def virial(args):
                _, P_vir, b_eps = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=temperature, eps=args[0], 
                                                         lm=args[1], fix=fix)
                return [abs(pressure-P_vir), np.abs(self.b_0*args[0]-b_eps)]
            solver = root(virial, x0=(1., 0.), method='lm', tol=1e-10)
            eff_temp, eps, lm = temperature, solver.x[0], solver.x[1]
        virial_q = self.find_virial_quant(bonds=bonds, meshes=meshes, temperature=eff_temp, eps=eps, lm=lm, fix=fix, return_rho_1=True)
        print('T: {}\nT_eff: {}\nT_vir: {}\nP_vir: {}\neps: {}\nlm: {}\n'.format(temperature, eff_temp, *virial_q[:2], eps, lm))
        return *virial_q, eff_temp, eps, lm
    
    @staticmethod
    def _validate(bonds, temperatures, pressures, epsilons):
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

    def get_ah_U(self, bonds, temperatures, pressures=None, epsilons=None, fix=[1, 1, 1], fix_T=False):
        """
        Collect the mean-filed approximated properties for a given set of temperatures. If pressures is not None, then
        the NPT case is considered. Ohterwise, considers the NVT case (at the strain (epsilons) value or at no 
        strain if epsilons=None). 

        Parameters:
            bonds (ndarray): Array of bond vectors.
            temperatures (list or float): List of temperatures or a single temperature value.
            pressures (list or float): List of pressures corresponding to the temperatures or a single pressure value. Defaults to None.
            epsilons (list or float): List of strains corresponding to the temperatures or a single strain value. Defaults to None.
            fix (list of booleans): Default [1, 1, 1] correspond to fixes 1, 2 and 3 respectively. Fixes 2 and 3 assume fix 1 to be True.

        Returns:
            dict: Dictionary containing the properties: 
                  'ah_U': Anharmonic internal energy,
                  'T_vir': Virial temperature,
                  'P_vir': Virial pressure,
                  'epsilon': Strain,
                  'lm': Lagrange multiplier.
        """
        ensemble = 'npt' if pressures is not None else 'nvt'
        epsilons = 1. if epsilons is None else epsilons
        
        # validate
        bonds, temperatures, pressures, epsilons = self._validate(bonds=bonds, temperatures=temperatures, pressures=pressures,
                                                                  epsilons=epsilons)
        
        # run npt or nvt
        T_virs, P_virs, _, rho_1s, eff_temps, eps, lms = zip(*[self.run_ensemble(bonds=bonds[i], temperature=temp, pressure=pressures[i], 
                                                               eps=epsilons[i], ensemble=ensemble, fix=fix, fix_T=fix_T) 
                                                               for i, temp in enumerate(temperatures)])
        # save for visualization
        self._rho_1s = rho_1s

        # collect anharmonic internal energy in meV/atom
        ah_U = []
        b_xyz = self.b_0*self.basis[0]
        for i, temp in enumerate(T_virs):
            per_bond_energy = (self.V1(bonds=bonds[i])*rho_1s[i]).sum()-self.V1(bonds=b_xyz)
            ah_U.append((6*per_bond_energy+self.energy_offset-1.5*KB*temp)*1000)

        return {
            'ah_U': np.array(ah_U),
            'T_vir': np.array(T_virs),
            'P_vir': np.array(P_virs),
            'epsilon': np.array(eps),
            'lattice': np.array(eps)*self.b_0*np.sqrt(2),
            'lm': np.array(lms)
        }

    def run_ensemble(self, bonds, temperature, pressure, eps, ensemble, fix, fix_T):
        """
        Run the appropriate ensemble simulation based on the ensemble type.

        Parameters:
            bonds (ndarray): (n_long x n_t1 x n_t2 x 3) bond vectors. Note the 'x 3' at the end, these are cartesian 'xyz' bond vectors.
            temperature (float): Target temperature.
            pressure (float): Target pressure.
            eps (float): Strain on bond b_0.
            ensemble (str): Ensemble type ('nvt' or 'npt').
            fix (bool): Whether to fix the single bond density.

        Returns:
            tuple: Tuple containing the properties: rho_1, epsilon, lm.
        """
        if ensemble == 'nvt':
            return self.run_nvt(bonds=bonds, temperature=temperature, eps=eps, fix=fix, fix_T=fix_T)
        elif ensemble == 'npt':
            return self.run_npt(bonds=bonds, temperature=temperature, pressure=pressure, fix=fix, fix_T=fix_T)

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