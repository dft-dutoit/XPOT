from __future__ import annotations

from xpot.models.ace import PACE
from xpot.optimiser import NamedOptimiser

mlip = PACE("ace_input.hjson")

kwargs = {
    "n_initial_points": 5,
}

opt = NamedOptimiser(mlip.optimisation_space, mlip.sweep_path, kwargs)
n_calls = 10

while opt.iter <= n_calls:
    opt.run_optimisation(mlip.fit, path=mlip.sweep_path)

opt.tabulate_final_results(mlip.sweep_path)
