from pyiron_base._tests import TestWithCleanProject
import pyiron_contrib


class LammpsInteractiveWithoutOutput(TestWithCleanProject):
    def test_clean_database_failure(self):
        og_status = self.project.state.settings.configuration["disable_database"]
        self.project.state.update({"disable_database": False})

        with self.assertRaises(RuntimeError):
            self.project.create.job.LammpsInteractiveWithoutOutput("foo")

        self.project.state.update({"disable_database": True})
        self.project.create.job.LammpsInteractiveWithoutOutput("foo")

        self.project.state.update({"disable_database": og_status})
