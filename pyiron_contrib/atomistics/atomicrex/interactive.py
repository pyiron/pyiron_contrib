import posixpath

from pyiron_base.job import InteractiveBase
from pyiron_contrib.atomistics.atomicrex.atomicrex_job import Atomicrex

try:
    import atomicrex
except ImportError:
    pass


class AtomicrexInteractive(InteractiveBase, Atomicrex):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self._interactive_library = atomicrex.Job()

    @property
    def atomicrex_job_object(self):
        return self._interactive_library

    def prepare_job(self):
        """
        Writes input files and calls necessary functions of the underlying atomicrex.Job class.
        """        
        self.write_input()
        input_file = posixpath.join(self.working_directory, "main.xml")
        self._interactive_library.parse_input_file(input_file)
        self._interactive_library.prepare_fitting()
        self._interactive_library.set_verbosity(2)
    
    def interactive_add_structure(identifier, structure, forces=None, params=None):
        """
        This should be done when the FlattenedARProperty is reworked to use the new FlattenedStorage,
        which allows to resize the necessary arrays on the fly.

        Wrapper around the atomicrex.Job add_ase_structure and add_library_structure methods

        Args:
            identifier ([type]): [description]
            structure ([type]): [description]
            params ([type], optional): [description]. Defaults to None.
        """        
        raise NotImplementedError("Changes needed in the atomicrex class before this can be implemented")
    
    def interactive_calculate_residual(self):
        """
        Calculate the residual. prepare_job needs to be called first
        """        
        self._interactive_library.calculate_residual()

    def interactive_calculate_hessian(self, parameters=None, eps=0.0001):
        """
        Calculate the hessian. prepare_job needs to be called first
        """  
        self._interactive_library.calculate_hessian(parameters=parameters, eps=eps)
    
    def interactive_calculate_gradient(self, parameters=None, eps=0.0001):
        """
        Calculate the gradient. prepare_job needs to be called first
        """  
        self._interactive_library.calculate_gradient(parameters=parameters, eps=eps)
    
    def run_if_interactive(self):
        self.prepare_job()
        self._interactive_library.perform_fitting()
