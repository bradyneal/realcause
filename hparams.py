from collections import OrderedDict


HP = OrderedDict(
    # dataset
    data=['lalonde'],
    dataroot=['PATH_TO_YOUR_DATASETS'],  # TODO: MODIFY THIS PATH LOCALLY
    # saveroot=['save'],
    # train=[True],
    # eval=[True],
    # overwrite_reload=[''],

    # distribution of outcome (y)
    dist=['SigmoidFlow'],
    dist_args=[['ndim=10', 'base_distribution=normal'], ['ndim=5', 'base_distribution=uniform']],
    atoms=[[0.0], []],  # list of floats, or empty list

    # architecture
    n_hidden_layers=[1],
    dim_h=[64],
    activation=['ReLU'],

    # training params
    lr=[0.001],
    batch_size=[64],
    num_epochs=[10],
    early_stop=[True],
    ignore_w=[False],
    grad_norm=['inf'],

    w_transform=['Standardize'],
    y_transform=['Normalize'],
    train_prop=[0.5],
    val_prop=[0.1],
    test_prop=[0.4],
    seed=[123],

    # evaluation
    num_univariate_tests=[100]
)
