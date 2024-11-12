import os
import shutil

from xpot.models.ace import PACE
from xpot.models.gap import GAP

# from xpot.models.mace import MACE
# from xpot.models.nequip import NequIP
# from xpot.models.snap import SNAP
from xpot.optimiser import NamedOptimiser

_path = os.getcwd()


def test_pace():
    mlip_class = PACE("tests/mock_runs/test_models/ace-input.hjson")
    assert mlip_class.optimisation_space is not None
    assert mlip_class.sweep_path is not None
    optimiser = NamedOptimiser(
        mlip_class.optimisation_space,
        mlip_class.sweep_path,
        {"n_initial_points": 1},
    )
    n_calls = 2
    while optimiser.iter <= n_calls:
        optimiser.run_optimisation(mlip_class.fit, path=mlip_class.sweep_path)

    assert optimiser.iter == 3
    os.chdir(_path)
    shutil.rmtree(mlip_class.sweep_path)


# def test_mace():
#     mlip_class = MACE("tests/mock_runs/test_models/mace-input.hjson")
#     assert mlip_class.optimisation_space is not None
#     assert mlip_class.sweep_path is not None
#     optimiser = NamedOptimiser(
#         mlip_class.optimisation_space,
#         mlip_class.sweep_path,
#         {"n_initial_points": 1},
#     )
#     n_calls = 2
#     while optimiser.iter <= n_calls:
#         optimiser.run_optimisation(mlip_class.fit, path=mlip_class.sweep_path)

#     assert optimiser.iter == 3
#     os.chdir(_path)
#     shutil.rmtree(mlip_class.sweep_path)


def test_gap():
    mlip_class = GAP("tests/mock_runs/test_models/gap-input.hjson")
    assert mlip_class.optimisation_space is not None
    assert mlip_class.sweep_path is not None
    optimiser = NamedOptimiser(
        mlip_class.optimisation_space,
        mlip_class.sweep_path,
        {"n_initial_points": 1},
    )
    n_calls = 2
    while optimiser.iter <= n_calls:
        optimiser.run_optimisation(mlip_class.fit, path=mlip_class.sweep_path)

    assert optimiser.iter == 3
    os.chdir(_path)
    shutil.rmtree(mlip_class.sweep_path)


# def test_snap():
#     # TODO: Fix SNAP 'number_of_atoms' keyerror in fitSNAP3
#     mlip_class = SNAP("tests/mock_runs/test_models/snap-input.hjson")
#     assert mlip_class.optimisation_space is not None
#     assert mlip_class.sweep_path is not None
#     optimiser = NamedOptimiser(
#         mlip_class.optimisation_space,
#         mlip_class.sweep_path,
#         {"n_initial_points": 1},
#     )
#     n_calls = 2
#     while optimiser.iter <= n_calls:
#         optimiser.run_optimisation(mlip_class.fit, path=mlip_class.sweep_path)

#     assert optimiser.iter == 3
#     os.chdir(_path)


# def test_nequip():
#     mlip_class = NequIP("tests/mock_runs/test_models/nequip-input.hjson")
#     assert mlip_class.optimisation_space is not None
#     assert mlip_class.sweep_path is not None
#     optimiser = NamedOptimiser(
#         mlip_class.optimisation_space,
#         mlip_class.sweep_path,
#         {"n_initial_points": 1},
#     )
#     n_calls = 2
#     while optimiser.iter <= n_calls:
#         optimiser.run_optimisation(mlip_class.fit, path=mlip_class.sweep_path)

#     assert optimiser.iter == 3
#     os.chdir(_path)
#     shutil.rmtree(mlip_class.sweep_path)
