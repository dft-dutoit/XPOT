import glob
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from ase.io import read, write
from xpot.potential_parsers.ace import ACE
from xpot.potential_parsers.gap import GAP
from xpot.potential_parsers.snap import SNAP


def _parse_method(method: str, args) -> object:
    if method == "GAP":
        ml_class = GAP(args)
    elif method == "SNAP":
        ml_class = SNAP(args)
    elif method == "ACE":
        ml_class = ACE(args)
    else:
        print("Error: method not recognized")
        sys.exit()
    return ml_class


def _get_args(input_script=None):
    print(sys.argv)
    if len(sys.argv) >= 2:
        return sys.argv[-1]
    else:
        return input_script


def _load_json(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)


def _load_general_params(self, inputs: list):
    self.fit_executable = inputs[0]["fitting_executable"]
    self.lammps_executable = inputs[0]["lammps_executable"]
    self.elements = inputs[0]["atomic_numbers"]
    self.project_name = inputs[0]["project_name"]
    self.sweep_name = inputs[0]["sweep_name"]
    self.nodes = inputs[0]["mpi_nodes"]
    self.mpi_cores_per_node = inputs[0]["mpi_cores_per_node"]
    self.loss_ratio = float(inputs[0]["error_energy_ratio"])
    self.error_method = inputs[0]["error_method"]


def prep_x_validation(input_directory: str, bins: int, include_test=True):
    files = []
    for path in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, path)):
            files.append(path)

    train_files = []
    test_files = []

    for file in files:
        if "train" in file:
            train_files.append(file)
            print(train_files)
        elif "test" in file and include_test == True:
            test_files.append(file)
        else:
            continue

    os.chdir(input_directory)
    for i in range(bins):
        os.mkdir(f"xval_{i}")
    for file in train_files:
        inputs = read(file, index=":")
        name = file.split("train")
        name = name[0] + "test_xval" + name[1]
        print(inputs)
        np.random.RandomState(42).shuffle(inputs)
        print(inputs)
        chunk_size = math.ceil(len(inputs) / bins)
        chunky_list = []
        for i in range(0, len(inputs), chunk_size):
            chunky_list.append(inputs[i : i + chunk_size])
        for i in range(len(chunky_list)):
            write(f"xval_{i}/{name}", chunky_list[i])

    for file in test_files:
        for i in range(len(chunky_list)):
            shutil.copy(file, f"xval_{i}/{file}")

    xval_files = []
    for path in os.listdir(f"{input_directory}/xval_0"):
        if os.path.isfile(os.path.join(f"{input_directory}/xval_0", path)):
            xval_files.append(path)

    for i in range(bins):
        os.chdir(f"{input_directory}/xval_{i}")
        with open(f"./test_xval.xyz", "w") as wfd:
            for f in os.listdir("."):
                if f.endswith(".xyz") and f != "test_xval.xyz":
                    with open(f, "r") as fd:
                        shutil.copyfileobj(fd, wfd)
                        # wfd.write("\n")

        for j in range(bins):
            if j == i:
                pass
            elif j != i:
                for fl in xval_files:
                    fl = str(f"../xval_{j}/{fl}")
                    newfl = fl.replace("test", "train")
                    newfl = newfl.replace(f"xval_{j}", f"xval_{i}")
                    with open(fl, "r") as f1:
                        with open(newfl, "a+") as f2:
                            shutil.copyfileobj(f1, f2)
                            # f2.write("\n")

    for i in range(bins):
        os.chdir(f"{input_directory}/xval_{i}")
        for file in os.listdir("."):
            if "_test_xval" in file:
                os.remove(file)
            else:
                continue

    os.chdir(input_directory)


def convert_xyz_pickle(
    data_files: str,
    ref_e: float | dict = 0,
    energy_key: str = "energy",
    force_key: str = "forces",
):
    # Read in the data
    dataset = {
        "energy": [],
        "forces": [],
        "ase_atoms": [],
        "energy_corrected": [],
    }

    structures = read(data_files, index=":")
    t_ref_e = 0

    if isinstance(ref_e, dict):
        elems = ref_e.keys()
        for atoms in structures:
            dataset["energy"].append(atoms.info[energy_key])
            dataset["forces"].append(atoms.arrays[force_key])
            dataset["ase_atoms"].append(atoms)
            for elem in elems:
                if elem in atoms.get_chemical_symbols():
                    chem_syms = atoms.get_chemical_symbols()
                    t_ref_e += ref_e[elem] * chem_syms.count(elem)
            dataset["energy_corrected"].append(atoms.info[energy_key] - t_ref_e)
            t_ref_e = 0

    else:
        for atoms in structures:
            dataset["energy"].append(atoms.info[energy_key])
            dataset["forces"].append(atoms.arrays[force_key])
            dataset["ase_atoms"].append(atoms)
            E = atoms.info[energy_key] - (len(atoms) * ref_e)
            dataset["energy_corrected"].append(E)

    df = pd.DataFrame(dataset)
    df.to_pickle(f"{data_files}.pkl.gzip", compression="gzip")


def get_best_potential(directory: str, method: str = "ACE"):
    # Gets the best potential from the directory sweep specified. Used at the
    # end of a sweep to get the best potential and copy it to a new folder.
    with open(f"{directory}/params.csv") as f:
        df = pd.read_csv(f)
        # best = df[df["loss"] == df["loss"].min()]
        best_iteration = df.index[df["loss"] == df["loss"].min()].tolist()
        best_iteration = best_iteration[0]
        shutil.copytree(
            f"{directory}/{best_iteration}",
            f"{directory}/best_pot_no{best_iteration}",
        )
        if method == "GAP":
            pot_file = glob.glob(
                f"{directory}/best_pot_no{best_iteration}/*.xml"
            )

        elif method == "ACE":
            subprocess.run(
                [
                    "pace_yaml2yace",
                    "-o",
                    f"{directory}/best_pot_no{best_iteration}/*.yace",
                    f"{directory}/best_pot_no{best_iteration}/*.yaml",
                ]
            )
            pot_file = glob.glob(
                f"{directory}/best_pot_no{best_iteration}/*.yace"
            )

        elif method == "SNAP":
            pot_files = glob.glob(
                f"{directory}/best_pot_no{best_iteration}/*.snap*"
            )
            with open(
                f"{directory}/best_pot_no{best_iteration}/snap_input.in", "r"
            ) as f:
                ref_setup = []
                for line in f.readlines():
                    if line.strip() == "[REFERENCE]":
                        copy = True
                        continue
                    elif line.startswith("["):
                        copy = False
                    elif copy:
                        ref_setup.append(line.split(" = "))
            pot_file = {}
            for entry in ref_setup:
                if entry[0] == "units" or entry[0] == "atom_style":
                    pot_file["lammps_header"] = f"{entry[0]} {entry[1]}"
                elif entry[0] == ("pair_style"):
                    pot_file["lmpcmds"] = [
                        f"pair_style snap",
                        f"pair_coeff * * {pot_files[0]} {pot_files[1]} ",
                    ]
                elif (
                    entry[0].startswith("pair_style")
                    and entry[0] != "pair_style"
                ):
                    pot_file["lmpcmds"] = [
                        f"pair_style hybrid/overlay {entry[1]} snap"
                    ]
                    for i in [
                        a for a in ref_setup if a[0].startswith("pair_coeff")
                    ] and i[1] != "* * zero":
                        pot_file["lmpcmds"].append(f"pair_coeff {i[1]}")
                    pot_file["lmpcmds"].append(
                        f"pair_coeff * * snap {pot_files[0]} {pot_files[1]} "
                    )

    return pot_file


def parse_previous_sweep(directory: str) -> list:
    # Returns the parameters and loss values of the specified sweep.
    # results[0] = parameters, results[1] = loss values
    with open(f"{directory}/params.csv", "r") as f:
        df = pd.read_csv(f)
        y0 = df["loss"].values.tolist()
        value_1 = df.drop(columns="loss")
        values = value_1.drop(columns="iteration").values.tolist()
    return values, y0


def potential_to_calculator(method: str, directory: str) -> object:
    # Returns the best potential from a sweep as an ASE calculator object.
    # Get the best potential from the sweep, return the folder
    # Modules are only loaded if they are needed for the method type
    # to maintain compatibility across installs.
    pot_file = get_best_potential(directory)
    if method == "GAP":
        from quippy.potential import Potential

        calc = Potential(param_filename=str(pot_file))
    elif method == "ACE":
        from pyace import PyACECalculator

        calc = PyACECalculator(str(pot_file))
    elif method == "SNAP":
        from ase.calculators.lammpslib import LAMMPSlib

        calc = LAMMPSlib(**pot_file)
    return calc
