#%%
import csv
import sys
import math
from collections import defaultdict
from pathlib import Path
from multiprocessing import Process, Manager

import matplotlib.pyplot as plt
import skopt
from joblib import Parallel, delayed
from skopt import gp_minimize, plots, Optimizer
from skopt.callbacks import CheckpointSaver
from tabulate import tabulate
from xpot.transfer_funcs.general import _get_args, _parse_method


def tabulate_pretty(file):
        with open(f"{file}.csv") as csv_file:
            reader = csv.reader(csv_file)
            rows = [row for row in reader]
            table = tabulate(rows, headers="firstrow", tablefmt="github")
        with open(f"{file}", "w+") as f:
            f.write(table)

def sweep_init_gp(method, input_script=None, **kwargs):
    args = _get_args(input_script)
    if 'x0' in kwargs:
        n = len(kwargs.get('x0'))
    else:
        n = 0
    mlip_obj = _parse_method(method, args)
    project = mlip_obj.project_name
    sweep = mlip_obj.sweep_name
    method = mlip_obj.error_method
    opt_space = mlip_obj.test_opt_space
    data_dict = defaultdict(list)
    print(project, sweep, opt_space)

    @skopt.utils.use_named_args(opt_space)
    def init_csv(**data):
        nonlocal data_dict, sweep, project
        data_dict["iteration"] = []
        data_dict["loss"] = []
        for k, v in data.items():
            data_dict[k] = []
        with open(f"./{project}/{sweep}/params.csv", "a") as f:
            for i in data_dict.keys():
                f.write(f"{i},")
        with open(f"./{project}/{sweep}/errors.csv", "a") as f:
            f.writelines(
                f"Iteration, {method} Energy (training), {method} Energy (testing), {method} Forces (training), {method} Forces (testing), {method} Stress (training), {method} Stress (testing)\n"
            )
        with open(f"./{project}/{sweep}/atom_errors.csv", "a") as f:
            f.writelines(
                f"Iteration, {method} Energy/atom (training), {method} Energy/atom (testing), {method} Force Components (training), {method} Force Components (testing),\n"
            )

    init_csv(opt_space)

    @skopt.utils.use_named_args(opt_space)
    def run_init(**params):
        nonlocal mlip_obj, n, sweep, method, data_dict
        n += 1
        print(params)
        #try:
        return mlip_obj.train(params, sweep, n, data_dict)
        #except BaseException as error:
        #    print(f"Error: {method} MLIP training failed:" 
        #            " please check the input file")
        #    return 1500

    #copy("./opt_GAP/base.in", f"./opt_GAP/{sweep}/")
    print(mlip_obj)
    #chck_saver = CheckpointSaver(f"./{project}/{sweep}/checkpoint.pkl", compress=9)
    #subprocess.run(f"cp ./opt_GAP/base.in ./opt_GAP/{sweep_name}", shell=True)
    results = gp_minimize(run_init, opt_space, **kwargs)
    #params = results.x
    plot_data(results, mlip_obj)
    tabulate_pretty(f"./{project}/{sweep}/params")
    tabulate_pretty(f"./{project}/{sweep}/errors")
    tabulate_pretty(f"./{project}/{sweep}/atom_errors")

def sweep_init_gp_xval(method, input_script=None, **kwargs):
    args = _get_args(input_script)
    if 'x0' in kwargs:
        n = len(kwargs.get('x0'))
    else:
        n = 0
    mlip_obj = _parse_method(method, args)
    project = mlip_obj.project_name
    sweep = mlip_obj.sweep_name
    method = mlip_obj.error_method
    opt_space = mlip_obj.test_opt_space
    data_dict = defaultdict(list)
    print(project, sweep, opt_space)

    @skopt.utils.use_named_args(opt_space)
    def init_csv(**data):
        nonlocal data_dict, sweep, project
        data_dict["iteration"] = []
        data_dict["loss"] = []
        for k, v in data.items():
            data_dict[k] = []
        with open(f"./{project}/{sweep}/params.csv", "a+") as f:
            for i in data_dict.keys():
                f.write(f"{i},")
        with open(f"./{project}/{sweep}/errors.csv", "a+") as f:
            f.writelines(
                f"Iteration, {method} Energy (training), {method} Energy (testing), {method} Forces (training), {method} Forces (testing), {method} Stress (training), {method} Stress (testing)\n"
            )
        with open(f"./{project}/{sweep}/atom_errors.csv", "a+") as f:
            f.writelines(
                f"Iteration, {method} Energy/atom (training), {method} Energy/atom (testing), {method} Force Components (training), {method} Force Components (testing), {method} Stress (training), {method} Stress (testing)\n"
            )

    init_csv(opt_space)

    @skopt.utils.use_named_args(opt_space)
    def run_init(**params):
        nonlocal mlip_obj, n, sweep, method, data_dict
        n += 1
        print(params)
        #try:
        return mlip_obj.train_xval(params, sweep, n, data_dict)
        #except BaseException as error:
        #    print(f"Error: {method} MLIP training failed:" 
        #            " please check the input file")
        #    return 1500

    #copy("./opt_GAP/base.in", f"./opt_GAP/{sweep}/")
    print(mlip_obj)
    #chck_saver = CheckpointSaver(f"./{project}/{sweep}/checkpoint.pkl", compress=9)
    #subprocess.run(f"cp ./opt_GAP/base.in ./opt_GAP/{sweep_name}", shell=True)
    results = gp_minimize(run_init, opt_space, **kwargs)
    #params = results.x
    plot_data(results, mlip_obj)
    tabulate_pretty(f"./{project}/{sweep}/params")
    tabulate_pretty(f"./{project}/{sweep}/errors")

def __sweep_init_gp_parallel(method, input_script=None, threads=2, device="cpu", **kwargs):
    manager = Manager()
    gpu_assignments = manager.dict()
    args = _get_args(input_script)
    if 'x0' in kwargs:
        n = len(kwargs.get('x0'))
    else:
        n = 0
    mlip_obj = _parse_method(method, args)
    project = mlip_obj.project_name
    sweep = mlip_obj.sweep_name
    method = mlip_obj.error_method
    opt_space = mlip_obj.test_opt_space
    data_dict = defaultdict(list)
    print(project, sweep)

    @skopt.utils.use_named_args(opt_space)
    def init_csv(**data):
        nonlocal data_dict, sweep, project
        data_dict["iteration"] = []
        data_dict["loss"] = []
        for k, v in data.items():
            data_dict[k] = []
        with open(f"./{project}/{sweep}/params.csv", "a+") as f:
            for i in data_dict.keys():
                f.write(f"{i},")
        with open(f"./{project}/{sweep}/errors.csv", "a+") as f:
            f.writelines(
                f"Iteration, {method} Energy (training), {method} Energy (testing), {method} Forces (training), {method} Forces (testing), {method} Stress (training), {method} Stress (testing)\n"
            )
        with open(f"./{project}/{sweep}/atom_errors.csv", "a+") as f:
            f.writelines(
                f"Iteration, {method} Energy/atom (training), {method} Energy/atom (testing), {method} Force Components (training), {method} Force Components (testing),\n"
            )

    init_csv(opt_space)

    def assign_device(device=device):
        if device=="cpu":
            pass
        else:
            gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            process_id = str(Process.current_process().ident)
            gpu_assignments[process_id] = gpu_ids.pop(0)

    def worker():
        while True:
            try:
                x = optimizer.ask()
                print(x)
                y = run_init(*x)
                optimizer.tell(x, y)
            except StopIteration:
                print("Finish")
                break

    @skopt.utils.use_named_args(opt_space)
    def run_init(**params):
        nonlocal mlip_obj, n, sweep, method, data_dict
        n += 1
        print(params)
        process_id = str(Process.current_process().ident)
        if process_id not in gpu_assignments:
            assign_gpu()
        gpu_id = gpu_assignments[process_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return mlip_obj.train(params, sweep, n, data_dict)

    opt_kwargs = kwargs.copy()
    opt_kwargs.pop("n_calls")
    optimizer = Optimizer(opt_space, base_estimator="GP", **opt_kwargs)

    workers = []
    if device=="gpu":
        for i in len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")):
            p = Process(target=worker)
            p.start()
            workers.append(p)
        
        for i in range(kwargs.get("n_calls", 20)):
            optimizer.ask()
        
        for p in workers:
            p.terminate()
        
    else:
        print("CPU parallel mode not implemented yet")

    print(mlip_obj)
    results = optimizer.get_result()
    plot_data(results, mlip_obj)
    tabulate_pretty(f"./{project}/{sweep}/params")
    tabulate_pretty(f"./{project}/{sweep}/errors")
    tabulate_pretty(f"./{project}/{sweep}/atom_errors")

def single_fit(method, input_script=None, **kwargs):
    args = _get_args(input_script)
    mlip_obj = _parse_method(method, args)
    project = mlip_obj.project_name
    sweep = mlip_obj.sweep_name
    method = mlip_obj.error_method
    data_dict = defaultdict(list)
    print(project, sweep)

    result = mlip_obj.single_train(sweep, data_dict)
    tabulate_pretty(f"./{project}/{sweep}/params")
    tabulate_pretty(f"./{project}/{sweep}/errors")
    tabulate_pretty(f"./{project}/{sweep}/atom_errors")
    print("Finished single fit")
    return result

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