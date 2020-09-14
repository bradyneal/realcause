import gc
from torch.nn import functional as F
from models.distributions.functional import *


def sigmoid_flow(x, logdet=0, ndim=4, params=None, delta=DELTA, logit_end=True):
    """
    element-wise sigmoidal flow described in `Neural Autoregressive Flows` (https://arxiv.org/pdf/1804.00779.pdf)
    :param x: input
    :param logdet: accumulation of log-determinant of jacobian
    :param ndim: number of dimensions of the transform
    :param params: parameters of the transform (batch_size x dimensionality of features x ndim*3 parameters)
    :param delta: small value to deal with numerical stability
    :param logit_end: whether to logit-transform it back to the real space
    :return:
    """
    assert params is not None, 'parameters not provided'
    assert params.size(2) == ndim*3, 'params shape[2] does not match ndim * 3'

    a = act_a(params[:, :, 0 * ndim: 1 * ndim])
    b = act_b(params[:, :, 1 * ndim: 2 * ndim])
    w = act_w(params[:, :, 2 * ndim: 3 * ndim])

    pre_sigm = a * x[:, :, None] + b
    sigm = torch.sigmoid(pre_sigm)
    x_pre = torch.sum(w * sigm, dim=2)

    logj = F.log_softmax(
      params[:, :, 2 * ndim: 3 * ndim], dim=2) + logsigmoid(pre_sigm) + logsigmoid(-pre_sigm) + log(a)
    logj = log_sum_exp(logj, 2).sum(2)
    if not logit_end:
        return x_pre, logj.sum(1) + logdet

    x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
    x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
    xnew = x_

    logdet_ = logj + np.log(1 - delta) - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
    logdet = logdet_.sum(1) + logdet

    return xnew, logdet


def sigmoid_flow_integral(x, ndim=4, params=None):
    assert params is not None, 'parameters not provided'
    assert params.size(2) == ndim * 3, 'params shape[2] does not match ndim * 3'

    a = act_a(params[:, :, 0 * ndim: 1 * ndim])
    b = act_b(params[:, :, 1 * ndim: 2 * ndim])
    w = act_w(params[:, :, 2 * ndim: 3 * ndim])

    pre_softplus = a * x[:, :, None] + b
    sfp = torch.nn.functional.softplus(pre_softplus)
    x_pre = torch.sum(w * sfp / a, dim=2)
    return x_pre


def sigmoid_flow_inverse(y, ndim=4, params=None, logit_end=True, x=None, tol=1e-2, max_iter=100, lr=0.1, verbose=False):
    if logit_end:
        y = torch.sigmoid(y)
    if x is None:
        x = y.clone().detach().requires_grad_(True)
    error_old = (sigmoid_flow(x, 0, ndim=ndim, params=params, logit_end=False)[0] - y).abs().max().item()

    def closure():
        """ Solves x such that f(x) - y = 0 <=> Solves x such that argmin_x F(x) - <x,y> """
        loss = sigmoid_flow_integral(x, ndim=ndim, params=params).sum() - torch.sum(x * y)
        x.grad = torch.autograd.grad(loss, x)[0].detach()
        return loss

    optimizer = torch.optim.LBFGS([x], lr=lr, max_iter=max_iter, tolerance_grad=tol, line_search_fn="strong_wolfe")
    optimizer.step(closure)

    error_new = (sigmoid_flow(x, 0, ndim=ndim, params=params, logit_end=False)[0] - y).abs().max().item()
    if verbose:
        print('inversion error', error_new)
    torch.cuda.empty_cache()
    gc.collect()

    if error_new > error_old:
        if verbose:
            print('learning rate too large for inversion')
        return sigmoid_flow_inverse(y, ndim=ndim, params=params, logit_end=False, x=x)
    else:
        return x


def quick_test():
    import matplotlib.pyplot as plt
    ndim = 4
    x = torch.linspace(-5, 5, 1000).unsqueeze(1)
    params = torch.randn(1, 1, ndim*3)
    y = sigmoid_flow(x, 0, ndim, params)[0]
    plt.plot(x.numpy(), y.data.numpy())


def quick_test_integral():
    import matplotlib.pyplot as plt
    ndim = 4
    x = torch.linspace(-5, 5, 1000).unsqueeze(1)
    params = torch.randn(1, 1, ndim * 3)

    y = sigmoid_flow(x, 0, ndim, params, logit_end=False)[0]
    plt.plot(x.numpy(), y.data.numpy())

    x_diff = x.clone().requires_grad_(True)
    y_ = torch.autograd.grad(
      sigmoid_flow_integral(x_diff, ndim, params),
      x_diff, torch.ones_like(x_diff))[0]
    plt.plot(x.numpy(), y_.data.numpy())

    assert torch.allclose(y, y_), 'failed'


def quick_test_inverse():
    import matplotlib.pyplot as plt
    ndim = 40
    logit_end = True
    x = torch.linspace(-5, 5, 1000).unsqueeze(1)
    params = torch.randn(1, 1, ndim * 3)

    y = sigmoid_flow(x, 0, ndim, params, logit_end=logit_end)[0]
    x_ = sigmoid_flow_inverse(y, ndim=ndim, params=params, logit_end=logit_end, x=None, tol=1e-3, max_iter=100, lr=0.1)
    print((x-x_).abs().max())

    plt.plot(x.numpy(), y.data.numpy())
    plt.plot(x.numpy(), x_.data.numpy())
