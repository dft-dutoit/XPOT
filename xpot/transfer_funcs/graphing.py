#%%
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from skopt import plots
from tabulate import tabulate

#%%

def plot_data(data, mlip_obj):
    directory = Path(".") / mlip_obj.project_name / mlip_obj.sweep_name
    
    # test plotting is possible
    try:
        plots.plot_convergence(data)
        plt.savefig(directory / "convergence.pdf")
    except:
        print("Convergence plot failed")

    a = plots.plot_objective(data, levels=20, size=3)
    plt.tight_layout()
    a.flatten()[0].figure.savefig(directory / "objective.pdf")

    b = plots.plot_evaluations(data)
    b.flatten()[0].figure.savefig(directory / "evaluations.pdf")
    
#%%
from matplotlib.ticker import ScalarFormatter

def plot_convergence(data, init_points=32):
    df = pd.read_csv(data)
    column = df.loss
    mins = [np.min(column[:i]) for i in range(1, len(column) + 1)]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(column)+1), column, '-o', color="darkgrey")
    ax.plot(range(1, len(column)+1), mins, '-o', label="Minimum", color="black")
    ax.axvspan(0, init_points, alpha=0.2, color='orange', label="Initial Points")
    ax.axvspan(init_points, len(column)+1, alpha=0.2, color='green', label='Bayesian Optimization')
    ax.set_xlim(0, len(column)+1)
    ax.set_title('Convergence of Loss')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_major_formatter(ScalarFormatter())
    

plot_convergence('./xpot_graphing/xpot/transfer_funcs/snap_17_lin_2j_16_fixed_data.csv')
# %%
