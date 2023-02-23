from functools import wraps

def wrap_without_files(cls, name, *methods):
    """
    Returns a subclass that defines the _interactive_disable_log_file flag and
    overrides to_hdf, child_project and refresh_job_status to neither actually
    write anything nor register the job with the database.

    Additional methods (!, not method names) can be given to also wrap them so
    that they are only called when the flag is not set.

    Args:
        cls (type): base class to derive from
        name (str): name of the new class
        *methods (functions): additional methods to be wrapped.

    Returns:
        type: subclass of cls that creates jobs that don't touch HDF or the
        database
    """

    def init(self, project, job_name):
        self._interactive_disable_log_file = False
        super(cls, self).__init__(project=project, job_name=job_name)

    @property
    @wraps(cls.child_project)
    def child_project(self):
        if not self._interactive_disable_log_file:
            return super(cls, self).child_project
        else:
            return self.project

    @wraps(cls.to_hdf)
    def to_hdf(self, *args, **kwargs):
        if not self._interactive_disable_log_file:
            super(cls, self).to_hdf(*args, **kwargs)

    @wraps(cls.refresh_job_status)
    def refresh_job_status(self):
        if not self._interactive_disable_log_file:
            return super(cls, self).refresh_job_status()

    body = {
        '__init__': init,
        'to_hdf': to_hdf,
        'child_project': child_project,
        'refresh_job_status': refresh_job_status
    }

    def wrap_meth(meth):
        """Return a method that calls its super() only if the flag is not set."""
        @wraps(meth)
        def wrapper(self, *args, **kwargs):
            if self._interactive_disable_log_file:
                return getattr(
                        super(cls, self),
                        meth.__name__
                )(self, *args, **kwargs)
        return wrapper

    for meth in methods:
        body[meth.__name__] = wrap_meth(meth)

    return type(name, (cls,), body)
