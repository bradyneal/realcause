
class NotFittedError(ValueError):
    """Exception class to raise if estimator is used before fitting."""

    def __init__(self, msg=None, *args, **kwargs):
        if msg is None:
            msg = 'Call "fit" with appropriate arguments before using this estimator.'
        super().__init__(msg, *args, **kwargs)
