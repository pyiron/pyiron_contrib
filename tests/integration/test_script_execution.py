import os
import unittest

from pyiron_contrib.workflow import Workflow


class TestPythonScriptCLI(unittest.TestCase):
    def test_python_script(self):
        MyPlot = Workflow.create.meta.python_script(
            os.path.abspath("../static/plot_script.py"),
            {"x_max": float, "n_points": int},
            kwargs={"--linestyle": None, "--linecolor": "k"},
            output_files={"plot": "plot.png"}
        )
        my_plot = MyPlot()
        my_plot(x_max=5.5, n_points=10, linestyle="'--'", linecolor="'green'")

        self.assertTrue(os.path.isfile(my_plot.outputs.plot.value))

        my_plot.working_directory.delete()

