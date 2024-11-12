from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import iread
from quippy.potential import Potential

import xpot.loaders as load
import xpot.maths as maths
from xpot.models.model import MLP

_this_file = Path(__file__).resolve()
_exec_path = Path(os.getcwd()).resolve()


class GAP(MLP):
    def __init__(self, infile: str) -> None:
        """
        Class for optimizing ACE potentials using XPOT, for use with pacemaker.

        Parameters
        ----------
        infile: str
            Path to input .hjson file containing XPOT and ML parameters.
        """

        super().__init__(infile, "gap_defaults.json")
        if not isinstance(self.mlp_total, dict):
            raise ValueError(
                """Model error: mlp_total is not a dictionary!
                Initialisation failed."""
            )

    def write_input_file(self) -> str:
        """
        Function to write out the command required for GAP. Named
        write_input_file to match the interface of other ML classes in XPOT.

        Parameters
        ----------
        params: dict
            Dictionary of parameters to be written to the input line.

        Returns
        -------
        str
            Command to pass to gap_fit.
        """

        self.mlp_total = load.trim_empty_values(self.mlp_total)

        gap_cmd = ""

        for k, v in self.mlp_total.items():
            if k == "gap":
                gap_cmd += "gap={"
                for term, term_dict in v.items():  # type: ignore
                    gap_cmd += f"{term} "
                    for term_params, param_vals in term_dict.items():
                        gap_cmd += f"{term_params}={param_vals} "
                    gap_cmd += ": "

                gap_cmd = gap_cmd[:-2] + "}"
            else:
                gap_cmd += f"{k}={v} "

        return gap_cmd

    def fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
    ) -> float:
        """
        The main fitting function for the SNAP potential. This function does
        the following:
        1. Replace old values with new hyperparameters from the optimiser.
        2. Reconstitute lists in the hyperparameter dictionary.
        3. Write the input file.
        4. Run fitsnap.
        5. Calculate errors from the validation set, write outputs, and return
        the loss value.

        Parameters
        ----------
        opt_values : dict
            Dictionary of parameter names and values returned by the optimiser
            for the current iteration of fitting.
        iteration : int
            The current iteration number.
        filename : str
            Path/Name for the input file to be written to.
        xval_set : int
            The current cross-validation set number.

        Returns
        -------
        float
            The loss value.
        """

        # Prepare fitting path
        self.prep_fit(opt_values, iteration)

        # Prepare input file
        self.mlp_total = self.list_to_reg(self.mlp_total)
        cmd_params = self.write_input_file()

        subprocess.run(f"gap_fit {cmd_params}", shell=True)

        tmp_train = self.calculate_errors(validation=False)
        tmp_test = self.calculate_errors(validation=True)

        train_errors = self.validate_errors(tmp_train, maths.get_rmse, [1, 0.5])
        test_errors = self.validate_errors(tmp_test, maths.get_rmse, [1, 0.5])

        _ = self.process_errors(
            train_errors[0], test_errors[0], "atomistic_errors.csv"
        )

        loss = self.process_errors(
            train_errors[1], test_errors[1], "loss_function_errors.csv"
        )

        return loss

    def collect_raw_errors(self, filename: str) -> Iterable[Atoms]:
        """
        Collect errors from the fitting process.

        Parameters
        ----------
        filename : str
            The file to read errors from. (XYZ format)

        Returns
        -------
        Iterable[Atoms]
            The structures to calculate errors on.
        """
        return load.load_structures(filename)

    def calculate_errors(
        self,
        validation: bool = False,
    ) -> dict[str, list[float]]:
        """
        Calculate test errors for GAP potential on testing set.

        Parameters
        ----------
        validation : bool
            Whether to use the validation or training set.
        xval : bool
            Whether cross-validation is being used.
        xval_set : int
            The cross-validation set in use.

        Returns
        -------
        dict
            Dictionary of errors.
        """
        if validation:
            structures = iread(self.xpot["validation_data"])  # type: ignore
        else:
            structures = iread(self.mlp_total["at_file"])  # type: ignore

        n_per_structure = []
        ref_energy = []
        ref_forces = []
        pred_energy = []
        pred_forces = []
        quip = Potential(param_filename=self.mlp_total["gap_file"])
        for atoms in structures:
            if isinstance(atoms, Atoms):
                n_per_structure.append(len(atoms))
                ref_energy.append(atoms.get_potential_energy())
                ref_forces.extend(atoms.get_forces().ravel())

                atoms.calc = quip
                pred_energy.append(atoms.get_potential_energy())
                pred_forces.extend(atoms.get_forces().ravel())
            else:
                raise ValueError(
                    "ASE error: atoms object is not an Atoms object"
                )

        energy_diff = [pred - ref for pred, ref in zip(pred_energy, ref_energy)]
        forces_diff = [pred - ref for pred, ref in zip(pred_forces, ref_forces)]
        print(energy_diff, forces_diff)

        errors = {
            "energies": energy_diff,
            "forces": forces_diff,
            "atom_counts": n_per_structure,
        }

        return errors

    def list_to_reg(self, d: load.NestedDict) -> load.NestedDict:
        """
        Convert lists to regularisation format for GAP.

        Parameters
        ----------
        d : dict
            The nested dictionary to be converted.

        Returns
        -------
        dict
            The dictionary with lists converted to strings.
        """
        for k, v in d.items():
            if isinstance(v, dict):
                self.list_to_reg(v)
            elif isinstance(v, list):
                d[k] = "{" + " ".join(str(x) for x in v) + "}"
        return d
