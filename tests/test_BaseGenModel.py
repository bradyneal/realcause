import pytest

from models.BaseGenModel import BaseGenModel


def test_subclass_working():

    class GenModelPassing(BaseGenModel):

        def __init__(self, w, t, y):
            self.w = w
            self.t = t
            self.y = y

        def sample_t(self, w):
            pass

        def sample_y(self, t, w):
            pass

    GenModelPassing(0, 0, 0)


def test_subclass_missing_attr():

    class GenModelMissingAttr(BaseGenModel):

        def __init__(self, w, t, y):
            self.w = w
            self.t = t

        def sample_t(self, w):
            pass

        def sample_y(self, t, w):
            pass

    with pytest.raises(TypeError):
        GenModelMissingAttr(0, 0, 0)


def test_subclass_missing_method():

    class GenModelMissingMethod(BaseGenModel):

        def __init__(self, w, t, y):
            self.w = w
            self.t = t
            self.y = y

        def sample_t(self, w):
            pass

    with pytest.raises(TypeError):
        GenModelMissingMethod(0, 0, 0)
