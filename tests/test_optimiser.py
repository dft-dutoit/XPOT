import os
import shutil

import skopt
from xpot.optimiser import NamedOptimiser


def test_ask_tell():
    os.mkdir("tests/mock_runs/optimisation")
    my_params = {
        "a": skopt.space.Integer(1, 10),
        "b": skopt.space.Categorical(["test", "test2"]),
    }
    optimiser = NamedOptimiser(
        my_params,
        "tests/mock_runs/optimisation",
        {"n_initial_points": 1},
    )
    assert optimiser._optimisable_params == my_params
    assert optimiser.iter == 1
    assert optimiser.sweep_path == "tests/mock_runs/optimisation"

    new_params = optimiser.ask()
    assert new_params != my_params
    assert len(new_params) == 2
    assert new_params["a"] in range(1, 11)
    assert new_params["b"] in ["test", "test2"]

    shutil.rmtree("tests/mock_runs/optimisation")


def test_optimisation():
    mypath = "tests/mock_runs/optimisation"
    os.mkdir(mypath)
    my_params = {
        "a": skopt.space.Integer(1, 10),
        "b": skopt.space.Categorical(["test", "test2"]),
    }
    optimiser = NamedOptimiser(
        my_params,
        mypath,
        {"n_initial_points": 1},
    )

    def dummy_func(x, **args):
        return 0.5

    out = optimiser.run_optimisation(dummy_func, f"{optimiser.sweep_path}")
    assert out == 0.5
    assert optimiser.iter == 2

    out2 = optimiser.run_optimisation(dummy_func, f"{optimiser.sweep_path}")
    assert out2 == 0.5

    # Test plotting
    optimiser.plot_results(mypath)
    assert os.path.exists(f"{mypath}/objective.pdf")
    assert os.path.exists(f"{mypath}/evaluations.pdf")

    # Test error for missing parameters.csv
    params = optimiser.ask()
    out3 = 0.6
    optimiser.tell(params, out3)
    os.remove(f"{mypath}/parameters.csv")
    try:
        optimiser.write_param_result(mypath, 3)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "FileNotFoundError not raised despite misssing parameters.csv"
        )


def test_optimiser():
    # Test the correct working of the optimiser under normal circumstances
    mypath = "tests/mock_runs/optimisation"
    optimiser = NamedOptimiser(
        {"a": skopt.space.Integer(1, 10)}, mypath, {"n_initial_points": 1}
    )
    optimiser.tabulate_final_results(mypath)
    assert os.path.exists(f"{mypath}/parameters_final")
    assert os.path.exists(f"{mypath}/atomistic_errors_final")
    assert os.path.exists(f"{mypath}/loss_function_errors_final")

    # Test the dumping of an optimiser object
    optimiser.dump_optimiser(mypath)
    assert os.path.exists(f"{mypath}/xpot-optimiser.pkl")

    # Test loading of matching optimiser object (same numer of params)
    optimiser._optimiser = None
    optimiser.load_optimiser(f"{mypath}/xpot-optimiser.pkl")
    assert optimiser._optimiser is not None
    assert len(optimiser._optimiser.space.dimensions) == 1

    # Test loading of non-matching optimiser object (different number of params)
    optimiser._optimiser = None
    optimiser._optimisable_params = {
        "a": skopt.space.Integer(1, 10),
        "b": skopt.space.Categorical(["test", "test2"]),
    }
    try:
        optimiser.load_optimiser(f"{mypath}/xpot-optimiser.pkl")
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError not raised")

    shutil.rmtree("tests/mock_runs/optimisation")
