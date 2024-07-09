import glob

import pandas as pd
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import read
from pyace import PyACECalculator
from quippy.potential import Potential

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

    structures = read(data_file, index=":")

    if isinstance(ref_e, dict):
        elems = ref_e.keys()
        for atoms in structures:
            t_ref_e = 0
            dataset["energy"].append(atoms.info[energy_key])
            dataset["forces"].append(atoms.arrays[force_key])
            dataset["ase_atoms"].append(atoms)
            for elem in elems:
                if elem in atoms.get_chemical_symbols():
                    chem_syms = atoms.get_chemical_symbols()
                    t_ref_e += ref_e[elem] * chem_syms.count(elem)
            dataset["energy_corrected"].append(atoms.info[energy_key] - t_ref_e)

    else:
        for atoms in structures:
            dataset["energy"].append(atoms.info[energy_key])
            dataset["forces"].append(atoms.arrays[force_key])
            dataset["ase_atoms"].append(atoms)
            E = atoms.info[energy_key] - (len(atoms) * ref_e)
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
    energies = database["energy_key"].to_list()
    forces = database["force_key"].to_list()

    data_file = data_file.split(".")[0]

    for i, atoms in enumerate(structures):
        atoms.info["energy"] = energies[i]
        atoms.arrays["forces"] = forces[i]
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
        best_pot = best_pot_path + "output_potential.yaml"
        calc = PyACECalculator(best_pot)
    
    elif method == "GAP":
        files = glob.glob(best_pot_path + "*.xml")
        best_pot = files[0]
        calc = Potential(best_pot)

    elif method == "SNAP":
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
            elif (
                entry[0].startswith("pair_style")
                and entry[0] != "pair_style"
            ):
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
            lmpcmds=pot_file["lmpcmds"],
            lammps_header=pot_file["lammps_header"]
            )
    else:
        raise ValueError("Method not recognized.")
        
    return calc



