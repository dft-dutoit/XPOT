from xpot.models import SNAP
from xpot.optimiser import NamedOptimiser

mlip = SNAP("./4q-si-gap18.hjson")

kwargs = {
    "n_initial_points": 3,
    "initial_point_generator": "hammersly",
    "verbose": True,
    "n_jobs": 1,
}

opt = NamedOptimiser(mlip.optimisation_space, mlip.sweep_path, kwargs)
n_calls = 10

while opt.iter <= n_calls:
    opt.run_optimisation(mlip.fit, path=mlip.sweep_path)

opt.tabulate_final_results(mlip.sweep_path)

# Alternatively can use the optimise method directly:
# from xpot.workflows import optimise
# # mlip_obj is the class object of the MLIP model
# optimise(SNAP, input_file, kwargs, max_iter=n_calls, min_loss=0.1, time_limit = 360)
