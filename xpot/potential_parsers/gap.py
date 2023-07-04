from pathlib import Path
import subprocess
import os
import sys
import csv
import copy
from ase.io import iread, read, write
from ase.calculators.lammpslib import LAMMPSlib
import numpy as np
import json
import skopt
import pandas as pd
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import hjson


path = os.getcwd()

this_file = Path(__file__).resolve()


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
    self.cutoff_lock = inputs[0]["cutoff_lock"]
    self.base_directory = inputs[0]["base_directory"]
    if type(self.elements) is list:
        self.elements = [eval(i) for i in self.elements]
    elif type(self.elements) is str:
        print(self.elements)
        self.elements = eval(self.elements) 
    

class GAP:
    all_options = {
        "e0_method": ["isolated", "average"],
        "kernel_regularisation_is_per_atom": ["T", "F"],
        "sigma_per_atom": ["T", "F"],
        "do_copy_atoms_file": ["T", "F"],
        "do_copy_at_file": ["T", "F"],
        "sparse_separate_file": ["T", "F"],
        "sparse_use_actual_gpcov": ["T", "F"],
        # "verbosity": ["NORMAL", "VERBOSE", "NERD", "ANALYSIS"],
        "do_ip_timing": ["T", "F"]
        # "sparsify_only_no_fit": ["T", "F"]
    }

    all_gap_options = {
        "covariance_type": [
            "gaussian",
            "ARD_SE",
            "dot_product",
            "bond_real_space",
            "pp",
        ],
        "mark_sparse": ["T", "F"],
        "add_species": ["T", "F"],
        "sparse_method": [
            "random",
            "pivot",
            "cluster",
            "uniform",
            "kmeans",
            "covariance",
            "cur_covariance",
            "cur_points",
        ],
    }

    gap_str_aliases = {
        "2b_f0": "f0",
        "2b_n_sparse": "n_sparse",
        "2b_delta": "delta",
        "2b_config_type_n_sparse": "config_type_n_sparse",
        "2b_sparse_method": "sparse_method",
        "2b_lengthscale_factor": "lengthscale_factor",
        "2b_lengthscale_uniform": "lengthscale_uniform",
        "2b_lenthscale_file": "lengthscale_file",
        "2b_sparse_file": "sparse_file",
        "2b_mark_sparse_atoms": "mark_sparse_atoms",
        "2b_add_species": "add_species",
        "2b_covariance_type": "covariance_type",
        "2b_theta": "theta",
        "2b_print_sparse_index": "print_sparse_index",
        "2b_unique_hash_tolerance": "unique_hash_tolerance",
        "2b_unique_descriptor_tolerance": "unique_descriptor_tolerance",
        "3b_f0": "f0",
        "3b_n_sparse": "n_sparse",
        "3b_delta": "delta",
        "3b_config_type_n_sparse": "config_type_n_sparse",
        "3b_sparse_method": "sparse_method",
        "3b_lengthscale_factor": "lengthscale_factor",
        "3b_lengthscale_uniform": "lengthscale_uniform",
        "3b_lenthscale_file": "lengthscale_file",
        "3b_sparse_file": "sparse_file",
        "3b_mark_sparse_atoms": "mark_sparse_atoms",
        "3b_add_species": "add_species",
        "3b_covariance_type": "covariance_type",
        "3b_theta": "theta",
        "3b_print_sparse_index": "print_sparse_index",
        "3b_unique_hash_tolerance": "unique_hash_tolerance",
        "3b_unique_descriptor_tolerance": "unique_descriptor_tolerance",
        "soap_f0": "f0",
        "soap_n_sparse": "n_sparse",
        "soap_delta": "delta",
        "soap_config_type_n_sparse": "config_type_n_sparse",
        "soap_sparse_method": "sparse_method",
        "soap_lengthscale_factor": "lengthscale_factor",
        "soap_lengthscale_uniform": "lengthscale_uniform",
        "soap_lenthscale_file": "lengthscale_file",
        "soap_sparse_file": "sparse_file",
        "soap_mark_sparse_atoms": "mark_sparse_atoms",
        "soap_add_species": "add_species",
        "soap_covariance_type": "covariance_type",
        "soap_theta": "theta",
        "soap_print_sparse_index": "print_sparse_index",
        "soap_unique_hash_tolerance": "unique_hash_tolerance",
        "soap_unique_descriptor_tolerance": "unique_descriptor_tolerance",
    }


    def __init__(self, infile):

        self.global_configs = {
            "at_file": "training.extxyz",
            "core_param_file": "",
            "core_ip_args": "",
            "energy_parameter_name": "energy",
            "local_property_parameter_name": "local_property",
            "force_parameter_name": "forces",
            "virial_parameter_name": "virial",
            "hessian_parameter_name": "hessian",
            "config_type_parameter_name": "config_type",
            "sigma_parameter_name": "sigma",
            "force_mask_parameter_name": "force_mask",
            "parameter_name_prefix": "",
            "gap_file": "gap.xml",
            "verbosity": "NORMAL",
            "template_file": "",
            "sparsify_only_no_fit": "F",
        }

        self.defaults_global = {
            "e0": "0.0",
            "local_property0": "0.0",
            "e0_offset": "0.0",
            "e0_method": "isolated",
            "default_sigma": "{0.002 0.2 0.2 0.0}",
            "sparse_jitter": "1.0e-8",
            "hessian_delta": "1.0e-2",
            "config_type_kernel_regularisation": "",
            "config_type_sigma": "",
            "kernel_regularisation_is_per_atom": "T",
            "sigma_per_atom": "T",
            "do_copy_atoms_file": "T",
            "do_copy_at_file": "T",
            "sparse_separate_file": "T",
            "sparse_use_actual_gpcov": "F",
            "rnd_seed": "1",
            "openmp_chunk_size": "1",
            "do_ip_timing": "F",
        }

        self.defaults_new = {
            "f0": 0.0,
            "n_sparse": 15,
            "delta": "(MANDATORY - YOU MUST ENTER A VALUE (2.0 is reasonable))",
            "config_type_n_sparse": "",
            "sparse_method": "random",
            "lengthscale_factor": 1.0,
            "lengthscale_uniform": 0.0,
            "lenthscale_file": "",
            "sparse_file": "",
            "mark_sparse_atoms": "F",
            "add_species": "T",
            "covariance_type": "gaussian",
            "theta": 1.0,
            "soap_exponent": 1.0,
            "print_sparse_index": "",
            "unique_hash_tolerance": 1.0e-10,
            "unique_descriptor_tolerance": 1.0e-10,
        }

        self.gap_cmd_dict = {
            "2b_type_and_order": "",
            "2b_f0": "",
            "2b_n_sparse": "",
            "2b_delta": "",
            "2b_config_type_n_sparse": "",
            "2b_sparse_method": "",
            "2b_lengthscale_factor": "",
            "2b_lengthscale_uniform": "",
            "2b_lenthscale_file": "",
            "2b_sparse_file": "",
            "2b_mark_sparse_atoms": "",
            "2b_add_species": "",
            "2b_covariance_type": "",
            "2b_theta": "",
            "2b_print_sparse_index": "",
            "2b_unique_hash_tolerance": "",
            "2b_unique_descriptor_tolerance": "",
            "3b_type_and_order": "",
            "3b_f0": "",
            "3b_n_sparse": "",
            "3b_delta": "",
            "3b_config_type_n_sparse": "",
            "3b_sparse_method": "",
            "3b_lengthscale_factor": "",
            "3b_lengthscale_uniform": "",
            "3b_lenthscale_file": "",
            "3b_sparse_file": "",
            "3b_mark_sparse_atoms": "",
            "3b_add_species": "",
            "3b_covariance_type": "",
            "3b_theta": "",
            "3b_print_sparse_index": "",
            "3b_unique_hash_tolerance": "",
            "3b_unique_descriptor_tolerance": "",
            "soap_type_and_order": "",
            "soap_f0": "",
            "soap_n_sparse": "",
            "soap_delta": "",
            "soap_config_type_n_sparse": "",
            "soap_sparse_method": "",
            "soap_lengthscale_factor": "",
            "soap_lengthscale_uniform": "",
            "soap_lenthscale_file": "",
            "soap_sparse_file": "",
            "soap_mark_sparse_atoms": "",
            "soap_add_species": "",
            "soap_covariance_type": "",
            "soap_theta": "",
            "soap_print_sparse_index": "",
            "soap_unique_hash_tolerance": "",
            "soap_unique_descriptor_tolerance": "",
        }

        self.potential_path = ""
        self.optimisation_space = {}
        self.opt_space = {}
        self.test_opt_space = []

        self.file_input(infile)
        path = self.base_directory
        self.define_opt_scikit()
        #self.test_set_parsing(self.sweep_name)

    def file_input(self, infile):
        # Read in input file | N.B. GAP is not mpi-parallelised in this code
        with open(infile, "r") as f:
            inputs = hjson.load(f)

        _load_general_params(self, inputs)
        self.testing_data = inputs[0]["testing_data"]
        gap_1 = inputs[1]
        gap_2 = inputs[2]
        gap_3 = inputs[3]
        if gap_1["USE"] == "T":
            for k, v in gap_1.items():
                self.gap_cmd_dict["2b_" + k] = v
        if gap_2["USE"] == "T":
            for k, v in gap_2.items():
                self.gap_cmd_dict["3b_" + k] = v
        if gap_3["USE"] == "T":
            for k, v in gap_3.items():
                self.gap_cmd_dict["soap_" + k] = v
        self.defaults_global = inputs[4]
        self.global_configs = inputs[5]

        (Path(".") / self.project_name / self.sweep_name).mkdir(
            parents=True, exist_ok=True
        )


    def define_opt_scikit(self):
        # Define optimization space for scikit-optimize.
        optimiser_keywords = ["range", "arange"]
        opt_space = []
        for k, v in self.gap_cmd_dict.items():
            if "skopt" in str(v):
                opt_space.append(eval(v[:-1] + f", name=k)"))
            if any(item in str(v) for item in optimiser_keywords):
                opt_space.append(
                    skopt.space.Real(min([*eval(v)]), max([*eval(v)]), name=k)
                )
            elif str(v).startswith("["):
                opt_space.append(skopt.space.Categorical(v, name=k))
            elif "opt" in str(v):
                for k1 in GAP.all_gap_options.keys():
                    if k1 in k:
                        opt_space.append(
                            skopt.space.Categorical(
                                GAP.all_gap_options[k1], name=k
                            )
                        )
            else:
                continue
        for k, v in self.defaults_global.items():
            if "skopt" in str(v) and not str(v).startswith("{"):
                opt_space.append(eval(v[:-1] + f", name=k)"))
            if any(str(v).startswith(item) for item in optimiser_keywords):
                opt_space.append(
                    skopt.space.Real(min([*eval(v)]), max([*eval(v)], name=k))
                )
            elif str(v).startswith("{"):
                keys = ["_energies", "_forces", "_virials", "_hessians"]
                v = v.replace("{", "")
                v = v.replace("}", "")
                a = v.split()
                print(a)
                if len(a) == len(keys):
                    for i in range(0, len(keys)):
                        if (
                            a[i].replace(".", "", 1).isdigit()
                        ):  # Check for whether the value is a fixed float
                            continue
                        else:
                            k = "default_sigma" + keys[i]
                            print(opt_space)
                            opt_space.append(eval(a[i][:-1] + f", name=k)"))
                else:
                    continue
            elif str(v).startswith("["):
                opt_space.append(skopt.space.Categorical(v, name=k))
            elif "opt" in str(v):
                opt_space.append(
                    skopt.space.Categorical(GAP.all_options[k], name=k)
                )
            else:
                continue

        cutoffs = [k for k in opt_space if "cutoff" in str(k).lower()]
        if len(cutoffs) > 1 and self.cutoff_lock == True:
            opt_space = [k for k in opt_space if k not in cutoffs[1:]]

        self.test_opt_space = opt_space 
        

    def file_parse_config_to_gap(self, config):
        os.chdir(path)
        print(config)
        cutoffs = [v for k,v in config.items() if "cutoff" in str(k).lower()]
        if len(cutoffs) >= 1 and self.cutoff_lock == True:
            config["2b_cutoff"] = cutoffs[0]
            config["3b_cutoff"] = cutoffs[0]
            config["soap_cutoff"] = cutoffs[0]
        gap_str = "gap={"
        prefixes = ["2b", "3b", "soap"]
        refactors = ["default_sigma", "config_type_sigma"]
        keys = ["_energies", "_forces", "_virials", "_hessians"]
        b = "type_and_order"
        dict1 = {}
        for k, v in config.items():
            if k in self.gap_cmd_dict:
                oldval = self.gap_cmd_dict[k]
                self.gap_cmd_dict[k] = v
                dict1[k] = oldval
            else:
                pass
        for a in prefixes:
            if self.gap_cmd_dict[f"{a}_type_and_order"] != "":
                if gap_str[-1] != "{":
                    gap_str += ": "
                gap_str += self.gap_cmd_dict[f"{a}_type_and_order"] + " "
                for key in self.gap_cmd_dict:
                    if a in key and b not in key:
                        if self.gap_cmd_dict[key] != "":
                            gap_str += (
                                f"{key[len(a)+1:]}={self.gap_cmd_dict[key]} "
                            )
            else:
                continue
        gap_str = gap_str[:-1] + "}"
        for k, v in self.defaults_global.items():
            if v != "":
                if k in refactors:
                    val_ref = self.defaults_global[k]
                    val = val_ref.replace("{", "")
                    val = val.replace("}", "")
                    val = val.split()
                    print(val)

                    for key in keys:
                        for c_k, c_v in config.items():
                            if c_k == k + key:
                                index = keys.index(key)
                                #print(index)
                                print(c_k, c_v)
                                val[index] = c_v
                                print(val)
                            else:
                                pass
                    val = "{" + " ".join(map(str, val)) + "}"
                    #self.defaults_global[refac] = val
                    gap_str += f" {k}={val}"
                        
                else:
                    gap_str += f" {k}={v}"

        for k, v in self.global_configs.items():
            if v != "":
                gap_str += f" {k}={v}"
        self.gap_str = gap_str
        for k, v in dict1.items():
            self.gap_cmd_dict[k] = v
        print(self.gap_str)


    def mkdir_and_move(self, sweep_name, iteration):
        Path(f"./opt_GAP/{sweep_name}/{iteration}").mkdir(
            parents=True, exist_ok=True
        )
        subprocess.run(
            [
                "cp",
                f"{self.global_configs['at_file']}",
                f"./opt_GAP/{sweep_name}/{iteration}/",
            ]
        )
        os.chdir(f"./opt_GAP/{sweep_name}/{iteration}")


    def train(self, config, sweep_name, iteration, data_dict):
        os.chdir(path)
        print("run starting")
        self.file_parse_config_to_gap(config)
        print("done parsing")
        self.mkdir_and_move(sweep_name, iteration)
        print("done moving")
        for k, v in config.items():
            data_dict[k].append(v)
        data_dict["loss"].append(np.NaN)
        gap_cmd = self.fit_executable + " " + self.gap_str
        subprocess.run(gap_cmd, shell=True)
        self.new_lammps_run(self.global_configs["at_file"], self.testing_data, iteration, method=self.error_method)
        print("Errors Calculated")
        data_dict["loss"].pop()
        data_dict["loss"].append(self.loss)
        data_dict["iteration"].append(iteration)
        self.write_results_panda(sweep_name, data_dict)
        print("Run complete - continuing")
        return self.loss


    def train_xval(self):
        print("Placeholder Function, not working yet")


    def single_train(self, sweep_name, data_dict):
        iteration = "single_fit"
        os.chdir(path)
        self.file_parse_config_to_gap({})
        self.mkdir_and_move(sweep_name, iteration)
        data_dict["loss"].append(np.NaN)
        gap_cmd = self.fit_executable + " " + self.gap_str
        subprocess.run(gap_cmd, shell=True)
        self.new_lammps_run(self.global_configs["at_file"], self.testing_data, iteration, method=self.error_method)
        data_dict["loss"].pop()
        data_dict["loss"].append(self.loss)
        data_dict["iteration"].append(iteration)
        self.write_results_panda(sweep_name, data_dict)
        return self.loss

    def test_set_parsing(self, sweep_name):
        testing_data = self.testing_data
        tests = iread(testing_data)
        num = 1
        Path(f"./opt_GAP/{sweep_name}/testing/xyzs").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"./opt_GAP/{sweep_name}/testing/datas").mkdir(
            parents=True, exist_ok=True
        )
        for atoms in tests:
            write(f"./opt_GAP/{sweep_name}/testing/xyzs/{num}.extxyz", atoms)
            write(
                f"./opt_GAP/{sweep_name}/testing/datas/{num}.data",
                atoms,
                format="lammps-data",
            )
            num += 1
        subprocess.run(
            f"cat {path}/opt_GAP/{sweep_name}/testing/xyzs/*.extxyz > "
            f"{path}/opt_GAP/{sweep_name}/testing/xyzs/cat.xyz",
            shell=True,
        )

    def new_lammps_run(self, train, test, iteration, method="mae", power=0.5):
        # Calculate errors for current GAP iteration through LAMMPS + ASE.
        print(os.getcwd())
        with open(f"./{self.global_configs['gap_file']}") as f:
            gap_id = f.readline().replace("<", "").replace(">", "")
            gap_id = gap_id.strip('\n')
        print(gap_id)

        cmds = ["pair_style quip", 
                f"pair_coeff * * ./{self.global_configs['gap_file']} 'Potential xml_label={gap_id}' {self.elements}"]

        Structure = namedtuple("Structure", 
                ["label", "dft_e", "lmp_e", "dft_f", "lmp_f"])

        structs = []
        pa_e_train, pa_f_train, pa_e_test, pa_f_test = [], [], [], []
        train_list = iread(train)
        for atoms in train_list:
            atoms.info["config_type"] = "train"
            dft_e = atoms.info[self.global_configs["energy_parameter_name"]]/len(atoms)**power
            dft_e_pa = atoms.info[self.global_configs["energy_parameter_name"]]/len(atoms)
            dft_f = atoms.arrays[self.global_configs["force_parameter_name"]]
            try:
                dft_s = atoms.info[self.global_configs["virial_parameter_name"]]
            except:
                dft_s = np.zeros(6)
            lmp = LAMMPSlib(lmpcmds=cmds)
            atoms.calc = lmp
            lmp_e = atoms.get_potential_energy()/len(atoms)**power
            lmp_f = atoms.get_forces()

            pa_e_train.append(atoms.get_potential_energy()/len(atoms) - dft_e_pa)
            pa_f_train.append(((lmp_f.flatten() - dft_f.flatten())))
            structs.append(Structure(atoms.info['config_type'], dft_e, 
                    lmp_e, dft_f, lmp_f))

        test_list = iread(test)
        for atoms in test_list:
            atoms.info["config_type"] = "test"
            dft_e = atoms.info[self.global_configs["energy_parameter_name"]]/len(atoms)**power
            dft_e_pa = atoms.info[self.global_configs["energy_parameter_name"]]/len(atoms)
            dft_f = atoms.get_forces()
            try:
                dft_s = atoms.get_stress()
            except:
                dft_s = np.zeros(6)
            lmp = LAMMPSlib(lmpcmds=cmds)
            atoms.calc = lmp
            lmp_e = atoms.get_potential_energy()/len(atoms)**power
            lmp_f = atoms.get_forces()

            pa_e_test.append(atoms.get_potential_energy()/len(atoms) - dft_e_pa)
            pa_f_test.append(np.flatten((lmp_f.flatten() - dft_f.flatten())))
            structs.append(Structure(atoms.info['config_type'], dft_e, 
                    lmp_e, dft_f, lmp_f))
            #print(lmp_e, dft_e)

        pa_f_train = np.vstack(pa_f_train)
        pa_f_test = np.vstack(pa_f_test)

        if method == "rmse":
            e_train = np.sqrt(np.mean(np.square(pa_e_train)))
            e_test = np.sqrt(np.mean(np.square(pa_e_test)))
            f_train = np.sqrt(np.mean(np.square(pa_f_train)))
            f_test = np.sqrt(np.mean(np.square(pa_f_test)))

            with open(f"{path}/{self.project_name}/{self.sweep_name}/atom_errors.csv", "a+") as f:
                f.write(f"{iteration},{e_train},{e_test},{f_train},{f_test}\n")


        elif method == "mae":
            e_train = np.mean(np.abs(pa_e_train))
            e_test = np.mean(np.abs(pa_e_test))
            f_train = np.mean(np.abs(pa_f_train))
            f_test = np.mean(np.abs(pa_f_test))

            with open(f"{path}/{self.project_name}/{self.sweep_name}/atom_errors.csv", "a+") as f:
                f.write(f"{iteration},{e_train},{e_test},{f_train},{f_test}\n")


        errors = self.validate_errors(structs, self.sweep_name, iteration, method=method)
        self.loss = self.loss_ratio * errors.test_e + (1-self.loss_ratio)* errors.test_f
        return self.loss
                   
    def validate_errors(self, error_list, sweep_name, iteration, method="mae"):
        err = namedtuple("err", "test_e train_e test_f train_f")
        if method == "mae":
            mae = namedtuple("mae", "label e f") # stress")
            maes = []
            for i in error_list:
                e = np.abs(i.dft_e - i.lmp_e)
                f = np.square(np.subtract(i.dft_f,i.lmp_f))
                f = np.sqrt(np.sum(f, axis=1))
                #s = np.abs(i.dft_s - i.lmp_s)
                maes.append(mae(i.label, e, f)) #, s))
            
            test_e = np.mean([i.e for i in maes if i.label == "test"])
            train_e = np.mean([i.e for i in maes if i.label == "train"])
            test_f_tmp = np.vstack([i.f for i in maes if i.label == "test"])
            train_f_tmp = np.vstack([i.f for i in maes if i.label == "train"])
            test_f = np.mean([np.abs(i) for i in test_f_tmp])
            train_f = np.mean([np.abs(i) for i in train_f_tmp])
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
                f = np.square(np.subtract(i.dft_f,i.lmp_f))
                f = np.cbrt(np.sum(f, axis=1))
                #s = (i.dft_s - i.lmp_s)**2
                rmses.append(rmse(i.label, e, f)) #, s))
            
            test_e = np.sqrt(np.mean([i.e for i in rmses if i.label == "test"]))
            train_e = np.sqrt(np.mean([i.e for i in rmses if i.label == "train"]))
            test_f_tmp = np.vstack([i.f for i in rmses if i.label == "test"])
            train_f_tmp = np.vstack([i.f for i in rmses if i.label == "train"])
            test_f = np.sqrt(np.mean([i**2 for i in test_f_tmp]))
            train_f = np.sqrt(np.mean([i**2 for i in train_f_tmp]))
            #test_s = np.sqrt(np.mean([i.s for i in rmses if i.label == "test"]))
            #train_s = np.sqrt(np.mean([i.s for i in rmses if i.label == "train"]))

            os.chdir(path)
            with open(f"./{self.project_name}/{sweep_name}/errors.csv","a+") as f:
                f.writelines(
                    f"{iteration},{train_e},{test_e},{train_f},"
                    f"{test_f},NaN,NaN\n"
                )

            return err(test_e, train_e, test_f, train_f) #, test_s, train_s)

    def write_results_panda(self, sweep_name, data_dict):
        # Write results to the pandas dataframe
        print(data_dict)
        dfdata = pd.DataFrame(data_dict)
        try:
            print(dfdata(index=False))
        except:
            print("pandas dataframe error")
        dfdata.to_csv(
            f"./{self.project_name}/{sweep_name}/params.csv",
            index=False,
        )