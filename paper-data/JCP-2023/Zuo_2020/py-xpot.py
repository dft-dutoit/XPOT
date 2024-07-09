from xpot.transfer_funcs.optimizers import *

kwargs = {
    "n_calls" : 150,
    "n_initial_points" : 32,
    "initial_point_generator" : "hammersly",
    "verbose" : True,
    "n_jobs" : 1
}

sweep_init_gp("SNAP", **kwargs)