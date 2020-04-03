from utils import to_pandas_df, to_tensors

Z = 'z'
T = 't'
Y = 'y'

PANDAS = 'pandas'
TORCH = 'torch'


def to_data_format(format, z, t, y):
    if format.lower() == PANDAS:
        return to_pandas_df(z, t, y)
    elif format.lower() == TORCH:
        return to_tensors(z, t, y)
