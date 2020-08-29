from collections import OrderedDict


HP = OrderedDict(
    # dataset
    data=['lalonde'],
    # dataroot=['data'],
    # saveroot=['save'],
    # train=[True],
    # eval=[True],
    # overwrite_reload=[''],

    # distribution of outcome (y)
    dist=['FactorialGaussian'],
    atoms=[[0]], # list of floats, or empty list

    # architecture
    n_hidden_layers=[1],
    dim_h=[64, 128],
    activation=['ReLU'],

    # training params
    lr=[0.001],
    batch_size=[64, 128],
    num_epochs=[10],
    early_stop=[True],
    ignore_w=[False],

    outcome_min=[0.0],
    outcome_max=[1.0],
    train_prop=[0.5],
    val_prop=[0.1],
    test_prop=[0.4],
    seed=[123],

    # evaluation
    num_univariate_tests=[100]
)
