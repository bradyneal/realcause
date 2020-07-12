import numpy as np


class Preprocess(object):
    def __init__(self, data=None):
        pass

    def transform(self, x):
        raise NotImplementedError

    def untransform(self, x):
        raise NotImplementedError


class PlaceHolderTransform(Preprocess):
    def transform(self, x):
        return x

    def untransform(self, x):
        return x


class Shifting(Preprocess):
    def __init__(self, b):
        self.b = np.cast['float32'](b)

    def transform(self, x):
        return x + self.b

    def untransform(self, x):
        return x - self.b


class Centering(Shifting):
    def __init__(self, data=None, mean=None):
        assert data is not None or mean is not None, 'at least one of data or mean must be provided'
        if data is not None:
            m = data.mean(0).astype('float32')
            if mean is not None:
                assert np.isclose(m, mean), 'mean of data is not close to the provided value'
        else:
            m = np.cast['float32'](mean)
        super(Centering, self).__init__(-m)


class Scaling(Preprocess):
    def __init__(self, s):
        self.s = np.cast['float32'](s)

    def transform(self, x):
        return x * self.s

    def untransform(self, x):
        return x / self.s


class VarianceRescaling(Scaling):
    def __init__(self, data=None, stdv=None, gain=1.0):
        assert data is not None or stdv is not None, 'at least one of data or stdv must be provided'
        if data is not None:
            s = data.std(0)
            if stdv is not None:
                assert np.isclose(s, stdv), 'stdv of data is not close to the provided value'
        else:
            s = np.cast['float32'](stdv)
        super(VarianceRescaling, self).__init__(gain / s)


class SequentialTransforms(Preprocess):
    def __init__(self, *transforms):
        self.transforms = transforms

    def transform(self, x):
        for t in self.transforms:
            x = t.transform(x)
        return x

    def untransform(self, x):
        for t in reversed(self.transforms):
            x = t.untransform(x)
        return x


class Standardize(SequentialTransforms):
    """
    standardize data so that data.mean() = 0 and data.std() = 1
    """

    def __init__(self, data):
        super(Standardize, self).__init__(Centering(data), VarianceRescaling(data))


class Normalize(SequentialTransforms):
    """
    normalize data so that data.max() = 1 and data.min() = 0
    """

    def __init__(self, data):
        a = data.min()
        b = data.max()
        super(Normalize, self).__init__(Shifting(-a), Scaling(1 / (b - a)))