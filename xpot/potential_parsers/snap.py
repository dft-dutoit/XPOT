import copy
import fileinput
import json
import os
import subprocess
from collections import defaultdict, namedtuple
from pathlib import Path
from os import listdir
from os.path import isfile, join
import shutil

import hjson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skopt
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS
from ase.io import iread, read, write
from ase.data import chemical_symbols, atomic_masses

path = os.getcwd()
print(path)

this_file = Path(__file__).resolve()

def _load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def _load_general_params(self, inputs):
    self.fit_executable = inputs[0]["fitting_executable"]
    self.lammps_executable = inputs[0]["lammps_executable"]
    self.elements = inputs[0]["atomic_numbers"]
    self.project_name = inputs[0]["project_name"]
    self.sweep_name = inputs[0]["sweep_name"]
    self.nodes = inputs[0]["mpi_nodes"]
    self.mpi_cores_per_node = inputs[0]["mpi_cores_per_node"]
    self.loss_ratio = float(inputs[0]["error_energy_ratio"])
    self.error_method = inputs[0]["error_method"]
    self.base_directory = inputs[0]["base_directory"]
    try:
        self.xval_sets = inputs[0]["xval_sets"]
    except:
        self.xval_sets = 1
    if type(self.elements) is list:
        self.elements = [eval(i) for i in self.elements]
    elif type(self.elements) is str:
        print(self.elements)
        self.elements = eval(self.elements) 


class SNAP:
    def __init__(self, infile):
        #self.infile = sys.argv[1]
        #self.infile = args[0]
        self.potential_path = ""
        self.optimisation_space = {}
        self.opt_space = {}
        self.test_opt_space = []
        self.snap_total = _load_json(this_file.parent / "snap_defaults.json")
        self.file_input(infile)
        os.chdir(self.base_directory)
        self.define_opt_scikit()
        

    def file_input(self, infile):
        with open(infile, "r") as f:
            inputs = hjson.load(f)
        self.infile = join(os.getcwd(), infile)
        _load_general_params(self, inputs)
 
        # override default parameters with user inputs
        for config_section, overrides in zip(self.snap_total.keys(), inputs[1:]):
            self.snap_total[config_section].update(overrides)
        if self.snap_total["[SOLVER]"]["solver"] == "NETWORK":
            self.snap_total["[NETWORK]"] = self.snap_total.pop("[PYTORCH]")
            self.snap_total.pop("[BISPECTRUM]")
            customs = {"[CUSTOM]":inputs[1]}
            self.snap_total = {**customs, **self.snap_total}
            


        (Path(".") / self.project_name / self.sweep_name).mkdir(
            parents=True, exist_ok=True
        )

    def define_opt_scikit(self):
        # Define the optimization space in scikit-optimize format.
        opt_space = []
        for i in self.snap_total:
            if i == "[GROUPS]":
                self.unoptimizables = [
                    "group_sections",
                    "group_types",
                    "random_sampling",
                    "BOLTZT",
                    "smartweights",
                ]
                group = defaultdict(list)
                group_headers = self.snap_total["[GROUPS]"]["group_sections"]
                #print(group_headers)
                for header in group_headers:
                    group[header] = []
                    #print(group_headers)
                for k, v in self.snap_total[i].items():
                    if k not in self.unoptimizables:
                        #print(k  + "+" +  v)
                        group["name"].append(k)
                        v_new = v.replace("  ", " ")
                        v = v_new.split(" ")
                        for index in range(len(v)):
                            #print(index)
                            group[group_headers[index + 1]].append(v[index])
                    else:
                        continue
                for k, v in group.items():
                    #print(k)
                    num = 0
                    for item in v:
                        if "skopt" in str(item):
                            name = group["name"][num] + "-" + k
                            opt_space.append(
                                eval(item[:-1] + f", name='{name}')")
                            )
                        elif str(item).startswith("["):
                            opt_space.append(
                                skopt.space.Categorical(
                                    item,
                                    name=f"{group['name'][num]} + '-' + {k}",
                                )
                            )
                        else:
                            num += 1
                            continue
                        num += 1
            else:
                for k, v in self.snap_total[i].items():
                    if "skopt" in str(v):
                        opt_space.append(eval(v[:-1] + ", name=k)"))
                    elif str(v).startswith("["):
                        opt_space.append(skopt.space.Categorical(v, name=k))
                    else:
                        continue
            print(f"{i} completed")
        self.test_opt_space = opt_space
        self.groups = group
        self.snap_total["[GROUPS]"]["group_sections"] = " ".join(self.snap_total["[GROUPS]"]["group_sections"])

    def mkdir_and_move(self, sweep_name, iteration, xval=False, xval_set=0):
        # Crate and move to directory of the current iteration
        if xval == False:
            Path(f"./{self.project_name}/{sweep_name}/{iteration}").mkdir(
                parents=True, exist_ok=True
            )
            subprocess.run(
                [
                    "cp",
                    "snap_input.in",
                    f"./{self.project_name}/{sweep_name}/{iteration}/",
                ]
            )
            os.chdir(f"./{self.project_name}/{sweep_name}/{iteration}")
        else:
            Path(f"./{self.project_name}/{sweep_name}/{iteration}/xval_{xval_set}"
            ).mkdir(parents=True, exist_ok=True)

            subprocess.run(
                [
                    "cp",
                    "snap_input.in",
                    f"./{self.project_name}/{sweep_name}/{iteration}/xval_{xval_set}",
                ]
            )
            os.chdir(f"./{self.project_name}/{sweep_name}/{iteration}/xval_{xval_set}")

    def file_parse_snap(self, config):
        # Write snap input file for individual iteration.
        os.chdir(path)
        print(config)
        f = open("snap_input.in", "w+")
        for i in self.snap_total:
            if i != "[PYTORCH]":
                f.writelines(i + "\n")
            elif i == "[PYTORCH]":
                continue
                
            if i != "[GROUPS]" and i != "[PYTORCH]":
                for k, v in self.snap_total[i].items():
                    if k == "dataPath":
                        self.input_loc = v
                    if k in config and k != "dataPath":
                        f.writelines(f"{k} = {config[k]}\n")
                    elif k not in config and v != "":
                        f.writelines(f"{k} = {v}\n")
                    else:
                        pass
                print(f"{i} hyperparameters written")

            elif i == "[GROUPS]":
                options = copy.deepcopy(self.groups)
                for k, v in self.groups.items():
                    #print(self.groups)
                    #print("hi")
                    numgroup = 0
                    for p in v:
                        if "skopt" in str(p):
                            options[k][numgroup] = config[
                                f"{self.groups['name'][numgroup]}-{k}"
                            ]
                            #print(options[k][numgroup])
                        else:
                            pass
                        numgroup += 1
                        #print(numgroup)
                for k, v in self.snap_total["[GROUPS]"].items():
                    if k in self.unoptimizables:
                        if k in config:
                            f.writelines(f"{k} = {v}\n")
                        elif (
                            k not in config
                            and v != ""
                            and k not in options["name"]
                        ):
                            f.writelines(f"{k} = {v}\n")
                    else:
                        num = options["name"].index(k)
                        #print(num)
                        #print(options)
                        if num < len(options["name"]):
                            f.writelines(
                                f"""{options['name'][num]} = {' '.join([str(values[num]) 
                                for values in np.array(list(options.values()))[1:,:]])}\n"""
                            )
        f.close()

    def file_parse_snap_xval(self, config, xval_set):
        # Write snap input file for individual iteration.
        os.chdir(path)
        print(config)
        f = open("snap_input.in", "w+")
        for i in self.snap_total:
            if i != "[PYTORCH]":
                f.writelines(i + "\n")
            elif i == "[PYTORCH]":
                continue
                
            if i != "[GROUPS]" and i != "[PYTORCH]":
                for k, v in self.snap_total[i].items():
                    if k == "dataPath":
                        f.writelines(f"{k} = {v}/xval_{xval_set}\n")
                        self.input_loc = f"{v}/xval_{xval_set}"
                    elif k in config and k != "dataPath":
                        f.writelines(f"{k} = {config[k]}\n")
                    elif k not in config and v != "":
                        f.writelines(f"{k} = {v}\n")
                    else:
                        pass
                print(f"{i} hyperparameters written")

            elif i == "[GROUPS]":
                options = copy.deepcopy(self.groups)
                for k, v in self.groups.items():
                    #print(self.groups)
                    #print("hi")
                    numgroup = 0
                    for p in v:
                        if "skopt" in str(p):
                            options[k][numgroup] = config[
                                f"{self.groups['name'][numgroup]}-{k}"
                            ]
                            #print(options[k][numgroup])
                        else:
                            pass
                        numgroup += 1
                        #print(numgroup)
                for k, v in self.snap_total["[GROUPS]"].items():
                    if k in self.unoptimizables:
                        if k in config:
                            f.writelines(f"{k} = {v}\n")
                        elif (
                            k not in config
                            and v != ""
                            and k not in options["name"]
                        ):
                            f.writelines(f"{k} = {v}\n")
                    else:
                        num = options["name"].index(k)
                        #print(num)
                        #print(options)
                        if num < len(options["name"]):
                            f.writelines(
                                f"""{options['name'][num]} = {' '.join([str(values[num]) 
                                for values in np.array(list(options.values()))[1:,:]])}\n"""
                            )
            
        f.close()


    def train(self, config, sweep_name, iteration, data_dict):
        # Main run function - one iteration per call.
        os.chdir(path)
        self.file_parse_snap(config)
        self.mkdir_and_move(sweep_name, iteration)
        for k, v in config.items():
            data_dict[k].append(v)
        data_dict["loss"].append(np.NaN)
        print(os.getcwd())
        subprocess.run(
            f"mpiexec -np {self.mpi_cores_per_node} python -m"
            f" {self.fit_executable} --printlammps -v snap_input.in",
            shell=True)
        print("bye")
        if self.snap_total["[SOLVER]"]["solver"] == "PYTORCH":
            self.val_snap_pytorch(sweep_name, iteration, method=self.error_method)  
        else:
            #self.val_snap_standard(
            #    sweep_name,
            #    iteration,
            #    f"./newsnap_metrics.md",
            #    method="rmse")
            self.final_val_external(path, self.input_loc)
            self.val_snap_standard(
                                sweep_name,
                                iteration,
                                str(self.snap_total["[OUTFILE]"]["metrics"]),
                                method=self.error_method
                                )
        data_dict["loss"].pop()
        data_dict["loss"].append(self.combined_error)
        data_dict["iteration"].append(iteration)
        self.write_snap_results_panda(sweep_name, data_dict)
        return self.combined_error

    def train_xval(self, config, sweep_name, iteration, data_dict):
        # x-validation run function - one iteration per call.
        os.chdir(path)
        data_dict["loss"].append(np.NaN)
        xval_errors = []
        for xval_set in range(self.xval_sets):
            self.file_parse_snap_xval(config, xval_set)
            self.mkdir_and_move(sweep_name, iteration, xval=True, xval_set=xval_set)
            print(os.getcwd())
            subprocess.run(
                f"mpiexec -np {self.mpi_cores_per_node} python -m"
                f" {self.fit_executable} --printlammps -v snap_input.in",
                shell=True)
            print("bye")
            if self.snap_total["[SOLVER]"]["solver"] == "PYTORCH":
                self.val_snap_pytorch(sweep_name, iteration, method=self.error_method)  
            else:
                self.final_val_external(path, self.input_loc)
                self.val_snap_standard(
                                    sweep_name,
                                    iteration,
                                    str(self.snap_total["[OUTFILE]"]["metrics"]),
                                    method=self.error_method
                                    )
            xval_errors.append(self.combined_error)
        for k, v in config.items():
                data_dict[k].append(v)
        print(xval_errors, np.mean(xval_errors))
        data_dict["loss"].pop()
        data_dict["loss"].append(np.mean(xval_errors))
        data_dict["iteration"].append(iteration)
        self.clean_errors_xval(sweep_name, iteration)
        self.write_snap_results_panda(sweep_name, data_dict)
        return self.combined_error

    def single_train(self, sweep_name, data_dict):
        iteration = "single_fit"
        os.chdir(path)
        self.file_parse_snap([])
        self.mkdir_and_move(sweep_name, "single_fit")
        data_dict["loss"].append(np.NaN)
        subprocess.run(
            f"mpiexec -np {self.mpi_cores_per_node} python -m"
            f" {self.fit_executable} --printlammps -v snap_input.in",
            shell=True)
        print("bye")
        if self.snap_total["[SOLVER]"]["solver"] == "PYTORCH":
            self.val_snap_pytorch(sweep_name, "single_fit", method=self.error_method)
        else:
            self.final_val_external(path, self.input_loc)
            self.val_snap_standard(
                                sweep_name,
                                "single_fit",
                                str(self.snap_total["[OUTFILE]"]["metrics"]),
                                method=self.error_method
                                )
        data_dict["loss"].pop()
        data_dict["loss"].append(self.combined_error)
        data_dict["iteration"].append(iteration)
        self.write_snap_results_panda(sweep_name, data_dict)
        return self.combined_error

    def val_snap_standard(self, sweep_name, iteration, input_table, method="mae"):
        # validation function for SNAP with new metrics output.
        df = (
            pd.read_table(
                input_table,
                sep="|",
                header=0,
                index_col=0,
                skipinitialspace=True,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
        )
        df.columns = df.columns.str.strip()
        test_rows = []
        dfs = [
            "test_energy",
            "test_force",
            "test_stress",
            "train_energy",
            "train_force",
            "train_stress",
        ]
        dfs = dict.fromkeys(dfs, pd.DataFrame())

        vrbls = sorted(
            [
                i
                for i in self.snap_total["[CALCULATOR]"].keys()
                if i in ["energy", "force", "stress"]
            ]
        )

        for i in [i for i in df["Unnamed: 1"]]:
            for v in vrbls:
                if all(x in i for x in ["test", "nweighted", f"{v[1:]}"]):
                    dfs[f"test_{v}"] = pd.concat(
                        [dfs[f"test_{v}"], df.loc[df["Unnamed: 1"] == i]],
                        axis=0,
                    )
                elif all(x in i for x in ["train", "nweighted", f"{v[1:]}"]):
                    dfs[f"train_{v}"] = pd.concat(
                        [dfs[f"train_{v}"], df.loc[df["Unnamed: 1"] == i]],
                        axis=0,
                    )
                else:
                    continue

        if method == "mae":
            mae_test, mae_train = dict.fromkeys(vrbls, "NaN"), dict.fromkeys(
                vrbls, "NaN"
            )
            for v in vrbls:
                try:
                    mae_train_tmp = [
                        a * b
                        for a, b in zip(
                            [
                                float(i)
                                for i in dfs[f"train_{v}"]["mae"].tolist()
                            ],
                            [
                                float(i)
                                for i in dfs[f"train_{v}"]["ncount"].tolist()
                            ],
                        )
                    ]
                    mae_train[v] = sum(mae_train_tmp) / sum(
                        [float(i) for i in dfs[f"train_{v}"]["ncount"].tolist()]
                    )

                    mae_test_tmp = [
                        a * b
                        for a, b in zip(
                            [
                                float(i)
                                for i in dfs[f"test_{v}"]["mae"].tolist()
                            ],
                            [
                                float(i)
                                for i in dfs[f"test_{v}"]["ncount"].tolist()
                            ],
                        )
                    ]
                    mae_test[v] = sum(mae_test_tmp) / sum(
                        [float(i) for i in dfs[f"test_{v}"]["ncount"].tolist()]
                    )

                except:
                    continue

            os.chdir(path)
            with open(
                f"./{self.project_name}/{sweep_name}/atom_errors.csv",
                "a+",
            ) as f:
                f.writelines(
                    f"{iteration},{mae_train['energy']},{mae_test['energy']},{mae_train['force']},"
                    f"{mae_test['force']}\n"
                )

            self.atom_error = self.loss_ratio * mae_test["energy"] + (
                                    1 - self.loss_ratio) * mae_test["force"]

        elif method == "rmse":
            rmse_test, rmse_train = dict.fromkeys(vrbls, "NaN"), dict.fromkeys(
                vrbls, "NaN"
            )
            for v in vrbls:
                try:
                    mse_train_tmp = [
                        a * b
                        for a, b in zip(
                            [
                                float(i)**2
                                for i in dfs[f"train_{v}"]["rmse"].tolist()
                            ],
                            [
                                float(i)
                                for i in dfs[f"train_{v}"]["ncount"].tolist()
                            ],
                        )
                    ]
                    rmse_train[v] = np.sqrt(sum(mse_train_tmp) / sum(
                        [float(i) for i in dfs[f"train_{v}"]["ncount"].tolist()]
                    ))

                    mse_test_tmp = [
                        a * b
                        for a, b in zip(
                            [
                                float(i)**2
                                for i in dfs[f"test_{v}"]["rmse"].tolist()
                            ],
                            [
                                float(i)
                                for i in dfs[f"test_{v}"]["ncount"].tolist()
                            ],
                        )
                    ]
                    rmse_test[v] = np.sqrt(sum(mse_test_tmp) / sum(
                        [float(i) for i in dfs[f"test_{v}"]["ncount"].tolist()]
                    ))

                except:
                    continue

            os.chdir(path)
            with open(
                f"./{self.project_name}/{sweep_name}/atom_errors.csv",
                "a+"
            ) as f:
                f.writelines(
                    f"{iteration},{rmse_train['energy']},{rmse_test['energy']},{rmse_train['force']},"
                    f"{rmse_test['force']}\n"
                )

            self.atom_error = self.loss_ratio * rmse_test["energy"] + (
                                    1 - self.loss_ratio) * rmse_test["force"]

        else:
            print("Error: Method not recognized - please use 'mae' or 'rmse'")

    def val_snap_pytorch(self, sweep_name, iteration, method="mae"):
        f_train = np.loadtxt("force_comparison.dat")
        f_test = np.loadtxt("force_comparison_val.dat")
        e_train = np.loadtxt("energy_comparison.dat")
        e_test = np.loadtxt("energy_comparison_val.dat")
        #s_train = np.loadtxt("stress_comparison.dat")
        #s_test = np.loadtxt("stress_comparison_val.dat")

        def calc_mae(a1, a2, vector=False):
            if vector == True:
                a1 = a1.reshape(int(len(a1)/3), 3)
                a2 = a2.reshape(int(len(a2)/3), 3)
                diff = a1 - a2
                norm = np.linalg.norm(diff, axis=1)
                return np.mean(norm)
            else:
                abs_diff = np.abs(a1 - a2)
                return np.mean(abs_diff)
       
        def calc_rmse(a1, a2):
            rms_diff = np.square(a1 - a2)
            return np.sqrt(np.mean(rms_diff))


        if method == "mae":
            print(f_test)
            mae_f_val = calc_mae(f_test[:, 0], f_test[:, 1])
            print(mae_f_val)
            mae_f_train = calc_mae(f_train[:, 0], f_train[:, 1])
            mae_e_val = calc_mae(e_test[:, 0], e_test[:, 1])
            mae_e_train = calc_mae(e_train[:, 0], e_train[:, 1])

            ce = self.loss_ratio * mae_e_val + mae_f_val
            self.combined_error = ce

            os.chdir(path)
            with open(f"./{self.project_name}/{sweep_name}/atom_errors.csv","a+") as f:
                f.writelines(
                    f"{iteration},{mae_e_train},{mae_e_val},{mae_f_train},{mae_f_val}\n"
                )
        
        elif method == "rmse":
            rmse_f_val = calc_rmse(f_test[:, 0], f_test[:, 1])
            rmse_f_train = calc_rmse(f_train[:, 0], f_train[:, 1])
            rmse_e_val = calc_rmse(e_test[:, 0], e_test[:, 1], p_atom=True)
            rmse_e_train = calc_rmse(e_train[:, 0], e_train[:, 1], p_atom=True)

            ce = self.loss_ratio * rmse_e_val + rmse_f_val
            self.combined_error = ce

            os.chdir(path)
            with open(f"./{self.project_name}/{sweep_name}/atom_errors.csv","a+"
            ) as f:
                f.writelines(
                    f"{iteration},{rmse_e_train},{rmse_e_val},{rmse_f_train},"
                    f"{rmse_f_val}\n"
                )

    
    def final_val_external(self, path, input_folder):
        # Run validation via external pylammps runtime.
        subprocess.call(f"python {this_file.parent}/snap_validate.py {self.infile} {path} {input_folder}", shell=True)
        print(os.getcwd())
        with open("loss.txt", "r") as f:
                self.combined_error=float(f.read())
        os.remove("loss.txt")

    def validate_errors(self, error_list, sweep_name, iteration, method="mae"):
        err = namedtuple("err", "test_e train_e test_f train_f")
        if method == "mae":
            mae = namedtuple("mae", "label e f") # stress")
            maes = []
            for i in error_list:
                e = np.abs(i.dft_e - i.lmp_e)
                f = np.abs(i.dft_f - i.lmp_f)
                #s = np.abs(i.dft_s - i.lmp_s)
                maes.append(mae(i.label, e, f)) #, s))
            
            test_e = np.mean([i.e for i in maes if "test" in i.label])
            train_e = np.mean([i.e for i in maes if "train" in i.label])
            test_f_tmp = np.vstack([i.f for i in maes if "test" in i.label])
            train_f_tmp = np.vstack([i.f for i in maes if "train" in i.label])
            test_f = np.mean([np.linalg.norm(i) for i in test_f_tmp])
            train_f = np.mean([np.linalg.norm(i) for i in train_f_tmp])
            #test_s = np.mean([i.s for i in maes if i.label == "test"])
            #train_s = np.mean([i.s for i in maes if i.label == "train"])

            os.chdir(path)
            with open(f"./{self.project_name}/{sweep_name}/errors.csv","a+") as f:
                f.writelines(
                    f"{iteration},{train_e},{test_e},{train_f},"
                    f"{test_f},NaN,NaN\n"
                )
            return err(test_e, train_e, test_f, train_f) #, test_s, train_s)
        
        elif method == "rmse":
            rmse = namedtuple("rmse", "label e f") # stress")
            rmses = []
            for i in error_list:
                e = (i.dft_e - i.lmp_e)**2
                f = (i.dft_f - i.lmp_f)**2
                #s = (i.dft_s - i.lmp_s)**2
                rmses.append(rmse(i.label, e, f)) #, s))
            
            test_e = np.sqrt(np.mean([i.e for i in rmses if "test" in i.label]))
            train_e = np.sqrt(np.mean([i.e for i in rmses if "train" in i.label]))
            test_f_tmp = np.vstack([i.f for i in rmses if "test" in i.label])
            train_f_tmp = np.vstack([i.f for i in rmses if "train" in i.label])
            test_f = np.sqrt(np.mean([np.linalg.norm(i) for i in test_f_tmp]))
            train_f = np.sqrt(np.mean([np.linalg.norm(i) for i in train_f_tmp]))
            #test_s = np.sqrt(np.mean([i.s for i in rmses if i.label == "test"]))
            #train_s = np.sqrt(np.mean([i.s for i in rmses if i.label == "train"]))

            os.chdir(path)
            with open(f"./{self.project_name}/{sweep_name}/errors.csv","a+") as f:
                f.writelines(
                    f"{iteration},{train_e},{test_e},{train_f},"
                    f"{test_f},NaN,NaN\n"
                )

            return err(test_e, train_e, test_f, train_f) #, test_s, train_s)

    def write_snap_results_panda(self, sweep_name, data_dict):
        dfdata = pd.DataFrame(data_dict)
        try:
            print(dfdata(index=False))
        except:
            print("pandas dataframe error")
        dfdata.to_csv(
            f"./{self.project_name}/{sweep_name}/params.csv",
            index=False,
        )

    def clean_errors_xval(self, sweep_name, iteration):
        os.chdir(path)
        df = pd.read_csv(f"./{self.project_name}/{sweep_name}/errors.csv")
        df1 = df.groupby(["Iteration"], as_index=False).mean()
        df1.to_csv(f"./{self.project_name}/{sweep_name}/errors.csv", index=False)

        df2 = pd.read_csv(f"./{self.project_name}/{sweep_name}/atom_errors.csv")
        df3 = df2.groupby(["Iteration"], as_index=False).mean()
        df3.to_csv(f"./{self.project_name}/{sweep_name}/atom_errors.csv", index=False)

    def fit_optimal_xval(self, sweep_name, data_dict):
        os.chdir(f"{path}/{self.project_name}/{sweep_name}")
        loss_min_index = data_dict["loss"].index(min(data_dict["loss"]))
        with open(f"./{loss_min_index}/xval_0/snap_input.in", "r") as f:
            snap_input = f.readlines()
        for i, line in enumerate(snap_input):
            if "smartweights" in line:
                start_line = i+1
            if "[MEMORY]" in line:
                end_line = i
            if "dataPath" in line:
                snap_input[i] = line.replace("/xval_0", "")
        for i in range(start_line, end_line):
            snap_input[i] = line.replace("_xval", "")
            if "test" in snap_input[i]:
                snap_input.pop(i)
        with open(f"./optimal_input.in", "w+") as f:
            f.writelines(snap_input)
        subprocess.run(
            f"mpiexec -np {self.mpi_cores_per_node} python -m"
            f" {self.fit_executable} --printlammps -v optimal_input.in",
            shell=True)
        print("Optimal SNAP potential fitted to entire dataset")
        
