from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import xpot.loaders as load
import xpot.maths as maths
from xpot.models.model import MLP

_this_file = Path(__file__).resolve()
_exec_path = Path(os.getcwd()).resolve()


class PACE(MLP):
    def __init__(self, infile: str) -> None:
        """
        Class for optimizing ACE potentials using XPOT, for use with pacemaker.

        Parameters
        ----------
        infile: str
            Path to input .hjson file containing XPOT and ML parameters.
        """
        MLP.__init__(self, infile, "ace_defaults.json")

    def write_input_file(
        self,
        filename: str = "xpot-ace.yaml",
    ) -> None:
        """
        Write the input file for the ACE potential from the hyperparameter +
        dictionary.

        Parameters
        ----------
        filename : str
            Path to the input file.
        """
        with open(filename, "w+") as f:
            yaml.safe_dump(dict(self.mlp_total), f)  # type: ignore

    def fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        filename: str = "xpot-ace.yaml",
    ) -> float:
        """
        The main fitting function for creating an ACE potential. This function
        does the following:
        1. Replace old values with new hyperparameters from the optimiser.
        2. Reconstitute lists in the hyperparameter dictionary.
        3. Write the input file.
        4. Run pacemaker.
        5. Collect errors from the fitting process, write outputs, and return
        the loss value.

        Parameters
        ----------
        filename : str
            Path/Name for the input file to be written to.

        Returns
        -------
        float
            The loss value.
        """
        self.prep_fit(opt_values, iteration)
        self.write_input_file(filename)

        subprocess.run(["pacemaker", filename])

        tmp_train = self.calculate_errors(validation=False)
        tmp_test = self.calculate_errors(validation=True)

        train_errors = self.validate_errors(tmp_train, maths.get_rmse, [1, 0.5])
        test_errors = self.validate_errors(tmp_test, maths.get_rmse, [1, 0.5])

        self.process_errors(
            train_errors[0], test_errors[0], "atomistic_errors.csv"
        )

        loss = self.process_errors(
            train_errors[1], test_errors[1], "loss_function_errors.csv"
        )

        return loss

    def collect_raw_errors(
        self,
        filename: str,
    ) -> pd.DataFrame:
        """
        Collect errors from the fitting process.

        Parameters
        ----------
        filename : str
            The file to read errors from.

        Returns
        -------
        pd.DataFrame
            The dataframe of the errors from the fitting process.

        """
        df = pd.read_pickle(filename, compression="gzip")
        return df

    def calculate_errors(
        self,
        validation: bool = False,
    ) -> dict[str, list[float]]:
        """
        Validate the potential from pickle files produced by :code:`pacemaker`
        during the fitting process.

        Parameters
        ----------
        validation : bool
            If True, calculate validation errors, otherwise calculate training
            errors.

        Returns
        -------
        dict
            The errors as a dictionary of lists.
        """
        if validation:
            errors = self.collect_raw_errors("test_pred.pckl.gzip")
        else:
            errors = self.collect_raw_errors("train_pred.pckl.gzip")

        n_per_structure = errors["NUMBER_OF_ATOMS"].values.tolist()

        ref_energy = errors["energy_corrected"].values.tolist()
        pred_energy = errors["energy_pred"].values.tolist()

        energy_diff = [pred - ref for pred, ref in zip(pred_energy, ref_energy)]

        ref_forces = np.concatenate(errors["forces"].to_numpy(), axis=None)
        pred_forces = np.concatenate(
            errors["forces_pred"].to_numpy(), axis=None
        )
        forces_diff = [pred - ref for pred, ref in zip(pred_forces, ref_forces)]

        errors = {
            "energies": energy_diff,
            "forces": forces_diff,
            "atom_counts": n_per_structure,
        }

        return errors
