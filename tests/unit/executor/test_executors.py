from functools import partialmethod
from pickle import PickleError
from time import sleep
import unittest

from pyiron_contrib.executors.executors import CloudpickleProcessPoolExecutor


class Foo:
    """
    A base class to be dynamically modified for testing CloudpickleProcessPoolExecutor.
    """
    def __init__(self, fnc: callable):
        self.fnc = fnc
        self.result = None

    @property
    def run(self):
        return self.fnc

    def process_result(self, future):
        self.result = future.result()


def dynamic_foo():
    """
    A decorator for dynamically modifying the Foo class to test
    CloudpickleProcessPoolExecutor.

    Overrides the `fnc` input of `Foo` with the decorated function.
    """
    def as_dynamic_foo(fnc: callable):
        return type(
            "DynamicFoo",
            (Foo,),  # Define parentage
            {
                "__init__": partialmethod(
                    Foo.__init__,
                    fnc
                )
            },
        )

    return as_dynamic_foo


class TestCloudpickleProcessPoolExecutor(unittest.TestCase):
    def test_unpickleable_callable(self):
        """
        We should be able to use an unpickleable callable -- in this case, a method of
        a dynamically defined class.
        """
        fortytwo = 42  # No magic numbers; we use it in a couple places so give it a var

        @dynamic_foo()
        def slowly_returns_42():
            sleep(0.1)
            return fortytwo

        dynamic_42 = slowly_returns_42()  # Instantiate the dynamically defined class
        self.assertIsInstance(
            dynamic_42,
            Foo,
            msg="Just a sanity check that the test is set up right"
        )
        self.assertIsNone(
            dynamic_42.result,
            msg="Just a sanity check that the test is set up right"
        )
        executor = CloudpickleProcessPoolExecutor()
        fs = executor.submit(dynamic_42.run)
        fs.add_done_callback(dynamic_42.process_result)
        self.assertFalse(fs.done(), msg="Should be running on the executor")
        self.assertEqual(fortytwo, fs.result(), msg="Future must complete")
        self.assertEqual(fortytwo, dynamic_42.result, msg="Callback must get called")

    def test_unpickleable_return(self):
        """
        We should _not_ be able to use an unpickleable return value -- in this case, a
        method of a dynamically defined class.
        """

        @dynamic_foo()
        def does_nothing():
            return

        @dynamic_foo()
        def slowly_returns_unpickleable():
            """
            Returns a complex, dynamically defined variable
            """
            sleep(0.1)
            inside_variable = does_nothing()
            inside_variable.result = "it was an inside job!"
            return inside_variable

        dynamic_dynamic = slowly_returns_unpickleable()
        executor = CloudpickleProcessPoolExecutor()
        fs = executor.submit(dynamic_dynamic.run)
        with self.assertRaises(PickleError):
            print(fs.result())  # Can't (un)pickle the result


if __name__ == '__main__':
    unittest.main()
