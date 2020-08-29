from models.distributions import *
import matplotlib.pyplot as plt

# hyper params
ndim = 100
bs = 256
lr = 0.1
sf = SigmoidFlow(ndim=ndim, base_distribution='uniform')


# toy data
def data_sampler(n=64):
    m1 = bernoulli_sampler(torch.ones(n, 1))
    x = exponential_sampler(torch.ones(n, 1) * 0.5) * m1
    x += gaussian_sampler(torch.ones(n, 1) - 1.5, torch.zeros(n, 1) - 1.0) * (1-m1)
    x -= 0.5
    x += torch.sigmoid(x * 5 + 1) * 2 - 1.0
    return x


# training
params = torch.randn(sf.num_params, requires_grad=True)
params.data.div_(1)
optim = torch.optim.Adam([params], lr=lr)
for i in range(2000):
    data = data_sampler(n=bs)
    logp = sf.likelihood(data, params[None, None])
    loss = - logp.mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    if (i+1) % 100 == 0:
        print(i+1, loss.item())

# plotting
data = data_sampler(n=1000000).squeeze()
plt.hist(data.data.numpy(), 100, density=True)

data = torch.linspace(-5, 5, 1000).unsqueeze(1)
logp = sf.likelihood(data, params[None, None])
plt.plot(data.data.numpy(), torch.exp(logp).data.numpy())
