import posixpath, os

from scipy import optimize

from pyiron_contrib.atomistics.atomicrex.general_input import ScipyAlgorithm

from pyiron_contrib.atomistics.atomicrex import output

from pyiron_base.job.interactive import InteractiveBase
from pyiron_contrib.atomistics.atomicrex.base import AtomicrexBase

try:
    import atomicrex
except ImportError:
    pass


## Class defined for future addition of other codes
## Not sure which functionality (if any) can be extracted yet, but a similar pattern is followed in other pyiron modules



class AtomicrexInteractive(AtomicrexBase, InteractiveBase):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self._interactive_library = atomicrex.Job()
        self._read_input_files = False

    @property
    def atomicrex_job_object(self):
        return self._interactive_library

    def interactive_prepare_job(self):
        """
        Writes input files and calls necessary functions of the underlying atomicrex.Job class.
        """
        # Reading the input file again causes several issues
        if not self._read_input_files:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
            os.chdir(self.path)
            self.write_input(directory=self.path)
            input_file = ("main.xml")
            self._interactive_library.parse_input_file(input_file)
            self._read_input_files = True
        self._interactive_library.prepare_fitting()
        #self._interactive_library.set_verbosity(2)
        

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
        return self._interactive_library.calculate_residual()

    def interactive_calculate_hessian(self, parameters=None, eps=0.0001):
        """
        Calculate the hessian. prepare_job needs to be called first
        """  
        return self._interactive_library.calculate_hessian(parameters=parameters, eps=eps)
    
    def interactive_calculate_gradient(self, parameters=None, eps=0.0001):
        """
        Calculate the gradient. prepare_job needs to be called first
        """  
        return self._interactive_library.calculate_gradient(parameters=parameters, eps=eps)
    
    def run_if_interactive(self):
        self.interactive_prepare_job()
        if isinstance(self.fit_algorithm, ScipyAlgorithm):
            if self.fit_algorithm.global_minimizer is None:
                res = optimize.minimize(
                    fun = self._interactive_library.calculate_residual,
                    x0 = self._interactive_library.get_potential_parameters(),
                    **self.fit_algorithm.local_minimizer_kwargs)
            else:
                minimizer_func = optimize.__getattribute__(self.global_minimizer)
                res = minimizer_func(
                    func=self._interactive_library.calculate_residual,
                    **self.fit_algorithm.global_minimizer_kwargs,
                )
            
            self._interactive_library.set_potential_parameters(res.x)
            self.output.residual = self._interactive_library.calculate_residual()
            self.output.iterations = res.nit
            self._interactive_library.output_results()
            self._interactive_library.print_properties()

            ## Delete the atomicrex object at the end to flush outputs to file
            del(self._interactive_library)
             
        else:
            self._interactive_library.perform_fitting()
            ## Delete the atomicrex object at the end to flush outputs to file
            del(self._interactive_library)
            self.collect_output(cwd=self.path)
        
    # Use the library functions to collect output, since no output is produced
    # when fitting using scipy
    def _scipy_collect(self):
        pass

    def _interactive_parse_parameters(self):
        self._interactive_library.print_potential_parameters()
    
    def _interactive_parse_structures(self):
        for name, structure in self._interactive_library.structures.items():
            structure.compute_properties()
            structure.print_properties()

