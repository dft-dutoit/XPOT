import glob
import random
import warnings

import pandas as pd
from ase.io import write

from xpot import loaders


def xyz2pkl(
    data_file: str,
    ref_e: float | dict = 0,
    energy_key: str = "energy",
    force_key: str = "forces",
) -> None:
    """
    Convert XYZ databases into pacemaker compatible pickle format. The reference
    energy can be either a float (for single element systems) or a dictionary
    (for multi-element systems). This function returns a pickle file with the
    same name as the input file, with the extension `.pkl.gzip`.

    Parameters
    ----------
    data_files : str
        File path to the XYZ database.
    ref_e : float
        Reference energy value(s).
    energy_key : str
        Key for the energy.
    force_key : str
        Key for the forces.
    """

    # Read in the data
    dataset = {
        "energy": [],
        "forces": [],
        "ase_atoms": [],
        "energy_corrected": [],
    }

    structures = loaders.load_structures(data_file)

    if isinstance(ref_e, dict):
        elems = ref_e.keys()
        for atoms in structures:
            t_ref_e = 0
            try:
                dataset["energy"].append(atoms.info[energy_key])
            except KeyError:
                dataset["energy"].append(atoms.get_potential_energy())
                warnings.warn(
                    "Energy key not found, using inbuilt energy method.",
                    stacklevel=2,
                )
            try:
                dataset["forces"].append(atoms.arrays[force_key])
            except KeyError:
                dataset["forces"].append(atoms.get_forces())
                warnings.warn(
                    "Force key not found, using inbuild forces method.",
                    stacklevel=2,
                )
            dataset["ase_atoms"].append(atoms)
            for elem in elems:
                if elem in atoms.get_chemical_symbols():
                    chem_syms = atoms.get_chemical_symbols()
                    t_ref_e += ref_e[elem] * chem_syms.count(elem)
            try:
                dataset["energy_corrected"].append(
                    atoms.info[energy_key] - t_ref_e
                )
            except KeyError:
                dataset["energy_corrected"].append(
                    atoms.get_potential_energy() - t_ref_e
                )

    else:
        for atoms in structures:
            try:
                dataset["energy"].append(atoms.info[energy_key])
            except KeyError:
                dataset["energy"].append(atoms.get_potential_energy())
                warnings.warn(
                    "Energy key not found, using inbuilt energy method.",
                    stacklevel=2,
                )
            try:
                dataset["forces"].append(atoms.arrays[force_key])
            except KeyError:
                dataset["forces"].append(atoms.get_forces())
                warnings.warn(
                    "Force key not found, using inbuild forces method.",
                    stacklevel=2,
                )
            dataset["ase_atoms"].append(atoms)
            try:
                E = atoms.info[energy_key] - (len(atoms) * ref_e)
            except KeyError:
                E = atoms.get_potential_energy() - (len(atoms) * ref_e)
            dataset["energy_corrected"].append(E)

    df = pd.DataFrame(dataset)
    df.to_pickle(f"{data_file}.pkl.gzip", compression="gzip")


def pkl2xyz(
    data_file: str,
    energy_key: str = "energy_corrected",
    force_key: str = "forces",
) -> None:
    """
    Convert pickle file into XYZ database.

    Parameters
    ----------
    data_files : str
        File path to the pickle file.
    e_offset : float
        Reference energy value(s).
    energy_key : str
        Key for the energy in the pickle file.
    force_key : str
        Key for the forces in the pickle file.
    """

    database = pd.read_pickle(data_file, compression="gzip")
    structures = database["ase_atoms"].to_list()
    energies = database[energy_key].to_list()
    forces = database[force_key].to_list()

    # data_file = data_file.split(".")[0]

    for i, atoms in enumerate(structures):
        atoms.calc = None
        atoms.info["energy"] = energies[i]
        atoms.arrays["forces"] = forces[i]
        print(data_file)
        atoms.write(f"{data_file}.xyz", append=True)


def pot2ase_calc(
    path: str,
    method: str,
) -> object:
    """
    Convert best potential from sweep into ASE calculator.

    Parameters
    ----------
    path : str
        File path of sweep to extract potential from.
    method : str
        The class of potential [SNAP, ACE, GAP, etc.].
    """

    best_pot_path = loaders.return_best_potential(path)

    if method == "ACE":
        from pyace import PyACECalculator

        best_pot = best_pot_path + "output_potential.yaml"
        calc = PyACECalculator(best_pot)

    elif method == "GAP":
        from quippy.potential import Potential

        files = glob.glob(best_pot_path + "*.xml")
        best_pot = files[0]
        calc = Potential(best_pot)

    elif method == "SNAP":
        from ase.calculators.lammpslib import LAMMPSlib

        files = glob.glob(best_pot_path + "*.snap*")
        coeff = files[0]
        param = files[1]
        with open(f"{best_pot_path}/snap_input.in") as f:
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
                    "pair_style snap",
                    f"pair_coeff * * {coeff} {param} ",
                ]
            elif entry[0].startswith("pair_style") and entry[0] != "pair_style":
                pot_file["lmpcmds"] = [
                    f"pair_style hybrid/overlay {entry[1]} snap"
                ]
                for i in [  # noqa: B020
                    a for a in ref_setup if a[0].startswith("pair_coeff")
                ] and i[1] != "* * zero":
                    pot_file["lmpcmds"].append(f"pair_coeff {i[1]}")
                pot_file["lmpcmds"].append(
                    f"pair_coeff * * snap {coeff} {param} "
                )

        calc = LAMMPSlib(
            lmpcmds=pot_file["lmpcmds"], lammps_header=pot_file["lammps_header"]
        )
    else:
        raise ValueError("Method not recognized.")

    return calc


def split_xyz(
    data_file: str,
    test_frac: float = 0.2,
    by_config: bool = True,
    config_key: str = "config_type",
    seed: int = 42,
) -> None:
    """
    Split a dataset into training and testing sets.

    Parameters
    ----------
    data_file : str
        File path to the XYZ database.
    test_frac : float
        Fraction of the dataset to use for testing.
    by_config : bool
        Whether to split the dataset by configuration.
    seed : int
        Random seed for reproducibility.
    """

    dataset = loaders.load_structures(data_file)
    configs = {}
    if by_config:
        configs = {}
        for atoms in dataset:
            if atoms.info[config_key] not in configs:
                configs[atoms.info[config_key]] = [atoms]
            else:
                configs[atoms.info[config_key]].append(atoms)

    else:
        configs = {"all": list(dataset)}

    train_set = []
    test_set = []
    for i in configs:
        # Randomly split the indexes of the configurations
        random.seed(seed)
        random.shuffle(configs[i])
        split = int(len(configs[i]) * test_frac)
        test_set.extend(configs[i][:split])
        train_set.extend(configs[i][split:])

    file_comps = data_file.split("/")
    file_name = file_comps[-1]
    train_data = "/".join(file_comps[:-1]) + f"/train-{file_name}"
    test_data = "/".join(file_comps[:-1]) + f"/test-{file_name}"

    write(train_data, train_set)
    write(test_data, test_set)


# def split_pkl(
#     data_file: str,
#     test_frac: float = 0.2,
#     by_config: bool = True,
#     config_key: str = "config_type",
#     seed: int = 42,
# ) -> None:
#     """
#     Split a dataset into training and testing sets.

#     Parameters
#     ----------
#     data_file : str
#         File path to the pickle database.
#     test_frac : float
#         Fraction of the dataset to use for testing.
#     by_config : bool
#         Whether to split the dataset by configuration.
#     seed : int
#         Random seed for reproducibility.
#     """

#     dataset = pd.read_pickle(data_file, compression="gzip")
#     print(dataset.columns)
#     configs = {}
#     if by_config:
#         configs = {}
#         for i in range(len(dataset)):
#             config = dataset["ase_atoms"][i].info["config_type"]
#             if config not in configs:
#                 configs[config] = [dataset.iloc[i]]
#             else:
#                 configs[config].append(dataset.iloc[i])

#     else:
#         configs = {"all": dataset}

#     train_set = []
#     test_set = []
#     for i in configs:
#         # Randomly split the indexes of the configurations
#         random.seed(seed)
#         random.shuffle(configs[i])
#         split = int(len(configs[i]) * test_frac)
#         test_set.extend(configs[i][:split])
#         train_set.extend(configs[i][split:])

#     train_set = pd.DataFrame(train_set)
#     test_set = pd.DataFrame(test_set)

#     data_file = data_file.split("/")
#     train_data = "/".join(data_file[:-1]) + f"/train-{data_file[-1]}"
#     test_data = "/".join(data_file[:-1]) + f"/test-{data_file[-1]}"

#     train_set.to_pickle(train_data, compression="gzip")
#     test_set.to_pickle(test_data, compression="gzip")
