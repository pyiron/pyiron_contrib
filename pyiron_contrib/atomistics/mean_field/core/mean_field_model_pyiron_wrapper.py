import os
import dill
import codecs
from pyiron_base import PythonTemplateJob, DataContainer
from pyiron_base.interfaces.has_hdf import HasHDF
from pyiron_contrib.atomistics.mean_field.core.mean_field_model import MeanFieldJob

def _to_pickle(value):
    return codecs.encode(dill.dumps(value), "base64").decode()


def _from_pickle(value):
    return dill.loads(codecs.decode(value.encode(), "base64"))


class FunctionContainer(HasHDF):
    __slots__ = ("func",)

    def __init__(self, func=None):
        self.func = func
        self.__doc__ = func.__doc__

    def _to_hdf(self, hdf):
        hdf["dill"] = _to_pickle(self.func)

    def _from_hdf(self, hdf, version=None):
        self.func = _from_pickle(hdf["dill"])

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class MeanFieldPyironJob(PythonTemplateJob):
    """
    """

    def __init__(self, project, job_name):
        super(MeanFieldPyironJob, self).__init__(project, job_name)
        self.input['bond_grids'] = None
        self.input['b_0s'] = None
        self.input['basis'] = None
        self.input['rotations'] = None
        self.input['r_potentials'] = None
        self.input['t1_potentials'] = None
        self.input['t2_potentials'] = None
        self.input['r_harm_potentials'] = None
        self.input['t1_harm_potentials'] = None
        self.input['t2_harm_potentials'] = None
        self.input['shells'] = 1
        self.input['la_shells'] = 1
        self.input['energy_list'] = None
        self.input['strain_list'] = None
        self.input['r_order'] = 1
        self.input['s_order'] = 0
        self.input['bloch_hessians'] = None
        self.input['kpoint_vectors'] = None
        self.input['kpoint_weights'] = None
        self.input['crystal'] = 'fcc'
        self.input['alpha_threshold'] = 0.01
        self.input['cutoff_radius'] = 25.
        self.input['rewrite_alphas'] = False
        self.input['temperatures'] = 100.
        self.input['pressure'] = None
        self.input['eps'] = 1.0 
        self.input['eta'] = None
        self.input['return_output'] = False 
        self.input['fix_T'] = True 
        self.input['return_rho_1s'] = False 
        self.input['rewrite_veff'] = False
        self.input['scaled'] = False
        self.mf_job = None

    def initialize_job(self):
        self.mf_job = MeanFieldJob(project_path=self.working_directory,
                                   bond_grids=self.input['bond_grids'],
                                   b_0s=self.input['b_0s'],
                                   basis=self.input['basis'],
                                   rotations=self.input['rotations'],
                                   r_potentials=self.input['r_potentials'],
                                   t1_potentials=self.input['t1_potentials'],
                                   t2_potentials=self.input['t2_potentials'],
                                   r_harm_potentials=self.input['r_harm_potentials'],
                                   t1_harm_potentials=self.input['t1_harm_potentials'],
                                   t2_harm_potentials=self.input['t2_harm_potentials'],
                                   shells=self.input['shells'],
                                   la_shells=self.input['la_shells'],
                                   energy_list=self.input['energy_list'],
                                   strain_list=self.input['strain_list'],
                                   r_order=self.input['r_order'],
                                   s_order=self.input['s_order'],
                                   bloch_hessians=self.input['bloch_hessians'],
                                   kpoint_vectors=self.input['kpoint_vectors'],
                                   kpoint_weights=self.input['kpoint_weights'],
                                   crystal=self.input['crystal'],
                                   alpha_threshold=self.input['alpha_threshold'],
                                   cutoff_radius=self.input['cutoff_radius'],
                                   rewrite_alphas=self.input['rewrite_alphas'],
                                   scaled=self.input['scaled']
                                   )

    def validate_ready_to_run(self):
        self.initialize_job()
        if not os.path.exists(self.working_directory):
            self._create_working_directory()
        
    def initialize_alphas(self):
        self.mf_job.generate_alphas()

    def run_mf_job(self, temperatures, pressure=None, eps=1., eta=None, return_output=False, fix_T=True, return_rho_1s=False, rewrite_veff=False):
        self.mf_job.run_ensemble(temperatures=temperatures, 
                                 pressure=pressure, 
                                 eps=eps, 
                                 eta=eta,
                                 return_output=return_output, 
                                 fix_T=fix_T, 
                                 return_rho_1s=return_rho_1s, 
                                 rewrite_veff=rewrite_veff)
        self.output.update(self.mf_job.output)

    def run_static(self):
        self.status.initialized = True
        if self.mf_job is None:
            self.initialize_job()
        self.initialize_alphas()
        self.status.running = True
        self.run_mf_job(temperatures=self.input['temperatures'], 
                        pressure=self.input['pressure'], 
                        eps=self.input['eps'], 
                        eta=self.input['eta'],
                        return_output=self.input['return_output'], 
                        fix_T=self.input['fix_T'], 
                        return_rho_1s=self.input['return_rho_1s'], 
                        rewrite_veff=self.input['rewrite_veff'])
        self.status.collect = True
        self.to_hdf()
        self.status.finished = True
        
    def convert_to_data_container(self, func_list):
        if func_list is None:
            return None
        else:
            new_func_list = [None if func is None else FunctionContainer(func) for func in func_list]
            if all(func is None for func in new_func_list):
                return new_func_list
            else:
                return DataContainer(new_func_list)

    def convert_from_data_container(self, data_container_list):
        if data_container_list is None:
            return None
        else:
            if isinstance(data_container_list, DataContainer):
                return data_container_list.to_builtin()
            else:
                return data_container_list
    
    def to_hdf(self, hdf=None, group_name=None):
        self.input['r_potentials'] = self.convert_to_data_container(self.input['r_potentials'])
        self.input['t1_potentials'] = self.convert_to_data_container(self.input['t1_potentials'])
        self.input['t2_potentials'] = self.convert_to_data_container(self.input['t2_potentials'])
        self.input['r_harm_potentials'] = self.convert_to_data_container(self.input['r_harm_potentials'])
        self.input['t1_harm_potentials'] = self.convert_to_data_container(self.input['t1_harm_potentials'])
        self.input['t2_harm_potentials'] = self.convert_to_data_container(self.input['t2_harm_potentials'])
        self.input['rotations'] = DataContainer(self.input['rotations'])
        super(MeanFieldPyironJob, self).to_hdf(hdf=hdf, group_name=group_name)

    def from_hdf(self, hdf=None, group_name=None):
        super(MeanFieldPyironJob, self).from_hdf(hdf=hdf, group_name=group_name)
        self.input['r_potentials'] = self.convert_from_data_container(self.input['r_potentials'])
        self.input['t1_potentials'] = self.convert_from_data_container(self.input['t1_potentials'])
        self.input['t2_potentials'] = self.convert_from_data_container(self.input['t2_potentials'])
        self.input['r_harm_potentials'] = self.convert_from_data_container(self.input['r_harm_potentials'])
        self.input['t1_harm_potentials'] = self.convert_from_data_container(self.input['t1_harm_potentials'])
        self.input['t2_harm_potentials'] = self.convert_from_data_container(self.input['t2_harm_potentials'])
        self.input['rotations'] = self.input['rotations'].to_builtin()
