import posixpath, os, time

from scipy import optimize
from pyiron_base import InteractiveBase

from pyiron_contrib.atomistics.atomicrex.general_input import ScipyAlgorithm
from pyiron_contrib.atomistics.atomicrex.base import AtomicrexBase

import_success = False
try:
    import atomicrex

    import_success = True
except ImportError:
    pass


class AtomicrexInteractive(AtomicrexBase, InteractiveBase):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        if import_success:
            self._interactive_library = atomicrex.Job()
        else:
            self._interactive_library = None
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
            input_file = "main.xml"
            self._interactive_library.parse_input_file(input_file)
            self._read_input_files = True
        self._interactive_library.prepare_fitting()
        # self._interactive_library.set_verbosity(2)

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
        raise NotImplementedError(
            "Changes needed in the atomicrex class before this can be implemented"
        )

    def interactive_calculate_residual(self):
        """
        Calculate the residual. prepare_job needs to be called first
        """
        return self._interactive_library.calculate_residual()

    def interactive_calculate_hessian(self, parameters=None, eps=0.0001):
        """
        Calculate the hessian. prepare_job needs to be called first
        """
        return self._interactive_library.calculate_hessian(
            parameters=parameters, eps=eps
        )

    def interactive_calculate_gradient(self, parameters=None, eps=0.0001):
        """
        Calculate the gradient. prepare_job needs to be called first
        """
        return self._interactive_library.calculate_gradient(
            parameters=parameters, eps=eps
        )

    def run_if_interactive(self):
        self.interactive_prepare_job()
        if isinstance(self.input.fit_algorithm, ScipyAlgorithm):
            self._scipy_run()
            # sleep between running and collecting so atomicrex output is flushed to file
            ## close to flush outputs to file
            self.interactive_close()
            self._scipy_collect(cwd=self.path)
        else:
            self._interactive_library.perform_fitting()
            ## close to flush outputs to file
            self.interactive_close()
            self.collect_output(cwd=self.path)

    def _scipy_run(self):
        if self.input.fit_algorithm.global_minimizer is None:
            res = optimize.minimize(
                fun=self._interactive_library.calculate_residual,
                x0=self._interactive_library.get_potential_parameters(),
                **self.input.fit_algorithm.local_minimizer_kwargs,
            )
        else:
            minimizer_func = optimize.__getattribute__(
                self.input.fit_algorithm.global_minimizer
            )
            res = minimizer_func(
                func=self._interactive_library.calculate_residual,
                **self.input.fit_algorithm.global_minimizer_kwargs,
            )

        # self._interactive_library.set_potential_parameters(res.x)
        self.output.residual = self._interactive_library.calculate_residual()
        self.output.iterations = res.nit
        self._interactive_library.print_potential_parameters()
        self._interactive_library.print_properties()
        self._interactive_library.output_results()
        return res

    def _scipy_collect(self, cwd=None):
        """
        Internal function that parses the output of an atomicrex job
        fitted using scipy.
        """
        if cwd is None:
            cwd = self.working_directory
        if self.input.__version__ >= "0.1.0":
            filepath = f"{cwd}/atomicrex.out"

        params_triggered = False
        structures_triggered = False

        with open(filepath, "r") as f:
            final_parameter_lines = []
            final_property_lines = []

            for l in f:
                if l.startswith("ERROR"):
                    self.status.aborted = True
                    self.output.error = l

                else:
                    if params_triggered:
                        if not l.startswith("---"):
                            final_parameter_lines.append(l)
                        else:
                            # Collecting lines with final parameters finished, hand over to the potential class
                            self.potential._parse_final_parameters(
                                final_parameter_lines
                            )
                            params_triggered = False

                    elif l.startswith("Potential parameters"):
                        # Get the number of dofs
                        n_fit_dofs = int(l.split("=")[1][:-3])
                        params_triggered = True

                    elif structures_triggered:
                        if not l.startswith("---"):
                            final_property_lines.append(l)
                        else:
                            # Collecting structure information finished, hand over structures class
                            self.structures._parse_final_properties(
                                final_property_lines
                            )
                            structures_triggered = False

                    elif l.startswith("Computing"):
                        structures_triggered = True
        self.status.finished = True
        self.to_hdf()
