from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_zty_linear_scalar_data, generate_zty_linear_multi_z_data
from data.whynot_simulators import generate_lalonde_random_outcome
from models import linear_gaussian_full_model, linear_gaussian_outcome_model, linear_multi_z_outcome_model, VAE

df = generate_zty_linear_scalar_data(500)

gen_model = DataGenModel(df, linear_gaussian_full_model, AutoNormal, n_iters=2000)
gen_model.plot_ty_dists(n_samples_per_z=10, name='SVI Test')

gen_model2 = DataGenModel(df, linear_gaussian_full_model, svi=False, mcmc='nuts')
gen_model2.plot_ty_dists(n_samples_per_z=10, name='NUTS Test')

df2 = generate_zty_linear_scalar_data(500, alpha=2, beta=10, delta=-5)
gen_model3 = DataGenModel(df2, linear_gaussian_full_model, AutoNormal, lr=.03, n_iters=1500)
print('Expected ATE: 5\t Actual: ', gen_model3.get_interventional_mean(1) - gen_model3.get_interventional_mean(0))
print('Expected ATE: 5\t Actual: ', gen_model3.get_ate())

gen_model4 = DataGenModel(df2, linear_gaussian_outcome_model, AutoNormal, lr=.03, n_iters=1500)
print('Expected ATE: 5\t Actual: ', gen_model4.get_interventional_mean(1) - gen_model4.get_interventional_mean(0))
print('Expected ATE: 5\t Actual: ', gen_model4.get_ate())

zdim = 10
(z, t, y) = generate_zty_linear_multi_z_data(100, zdim=zdim, delta=10)
gen_model = DataGenModel((z, t, y), linear_multi_z_outcome_model, AutoNormal, lr=.03, n_iters=2500)
print('Expected ATE: 10\t Actual: ', gen_model.get_ate())

zdim = 5
(z, t, y) = generate_zty_linear_multi_z_data(100, zdim=zdim, delta=10)
vae = VAE(w_dim=zdim)
gen_model = DataGenModel((z, t, y), vae.model, vae.guide, lr=.001, n_iters=20000)
