from pyiron_contrib.atomistics.atomicrex.interactive import AtomicrexInteractive


class Atomicrex(AtomicrexInteractive):
    """Class to set up and run atomicrex jobs"""

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.__name__ = "Atomicrex"
        self._executable_activate(enforce=True)
