import torch
import torch.nn as nn
import numpy as np

DELTA = 1e-7


def softplus(x, delta=DELTA):
    return nn.functional.softplus(x) + delta


def sigmoid(x, delta=DELTA):
    return torch.sigmoid(x) * (1 - delta) + 0.5 * delta


def logsigmoid(x):
    return -softplus(-x)


def log(x, delta=DELTA):
    return torch.log(x * 1e2 + DELTA) - np.log(1e2 + delta)


def logit(x):
    return log(x) - log(1 - x)


def act_a(x):
    return nn.functional.softplus(x)


def act_b(x):
    return x


def act_w(x):
    return nn.functional.softmax(x, dim=2)


def oper(array, operetor, axis=-1, keepdims=False):
    a_oper = operetor(array)
    if keepdims:
        shape = []
        for j, s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def log_sum_exp(a, axis=-1, sum_op=torch.sum):
    def maximum(x):
        return x.max(axis)[0]

    a_max = oper(a, maximum, axis, True)

    def summation(x):
        return sum_op(torch.exp(x - a_max), axis)

    b = torch.log(oper(a, summation, axis, True)) + a_max
    return b
