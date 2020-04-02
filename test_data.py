from data.synthetic import generate_zty_linear_scalar_data
from data.whynot_simulators import generate_lalonde_random_outcome


def test_linear_scalar_data():
    df = generate_zty_linear_scalar_data(10)


def test_lalonde_random_outcome_data():
    df, causal_effects = generate_lalonde_random_outcome()
