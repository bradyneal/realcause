from utils import get_num_positional_args


def test_get_num_positional_args():
    def f0():
        return

    def f1p_0kw(x):
        return

    def f1p_1kw(x, a=None):
        return

    def f1p_2kw(x, a=None, b=None):
        return

    def f2p_1kw(x, y, a=None):
        return

    def f2p_2kw(x, y, a=None, b=None):
        return

    assert get_num_positional_args(f0) == 0
    assert get_num_positional_args(f1p_0kw) == 1
    assert get_num_positional_args(f1p_1kw) == 1
    assert get_num_positional_args(f1p_2kw) == 1
    assert get_num_positional_args(f2p_1kw) == 2
    assert get_num_positional_args(f2p_2kw) == 2
