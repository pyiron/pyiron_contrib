from pyiron_base import InputList

class Output(InputList):
    def __init__(self):
        super().__init__(table_name="output")
        self.error = None
        self.residual = None
        self.steps = None

    @staticmethod
    def _get_structure_lines(l_index_complete, lines):
        end_index = None
        start_index = l_index_complete+3
        if lines[start_index-1] != "Computing structure properties\n":
            raise RuntimeError("Probably unknown formatting of output file")

        for i, l in enumerate(lines[start_index:]):
            if l.startswith("--"):
                end_index = start_index + i
                break
        return lines[start_index:end_index]

    @staticmethod
    def _get_parameter_lines(l_index_params, lines):
        # Get number of fitted parameters from a line similar to:
        # Potential parameters being optimized (#dof=18):
        n_params = int(lines[l_index_params].split("=")[-1].split(")")[0])
        lines = lines[l_index_params+1:l_index_params+n_params+1]
        return lines