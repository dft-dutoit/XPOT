from __future__ import annotations

import csv
import os
import subprocess
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import xpot.loaders as load
import xpot.maths as maths

_this_file = Path(__file__).resolve()
_exec_path = Path(os.getcwd()).resolve()


class PACE:
    def __init__(self, infile: str) -> None:
        """
        Class for optimizing ACE potentials using XPOT, for use with pacemaker.

        Parameters
        ----------
        infile: str
            Path to input .hjson file containing XPOT and ML parameters.
        """

        defaults = load.get_defaults(
            join(_this_file.parent / "defaults" / "ace_defaults.json")
        )
        # Load input hyperparameters and remove the XPOT parameters.
        os.chdir(_exec_path)
        hypers = load.get_input_hypers(infile)
        self.xpot = hypers["xpot"]
        hypers.pop("xpot")

        self.project = self.xpot["project_name"]
        self.sweep = self.xpot["sweep_name"]
        self.sweep_path = join(_exec_path / self.project / self.sweep)
        os.makedirs(self.sweep_path)

        self.ace_total = load.merge_hypers(defaults, hypers)
        load.validate_subdict(self.ace_total, hypers)
        self.optimisation_space = load.get_optimisable_params(self.ace_total)

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
        self.ace_total = load.trim_empty_values(self.ace_total)
        self.ace_total = load.convert_numpy_str(self.ace_total)

        with open(filename, "w+") as f:
            yaml.safe_dump(dict(self.ace_total), f)

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
        opt_values : dict
            Dictionary of parameter names and values returned by the optimiser
            for the current iteration of fitting.
        iteration : int
            The current iteration number.
        filename : str
            Path/Name for the input file to be written to.

        Returns
        -------
        float
            The loss value.
        """

        # Prepare fitting path
        self.iter = iteration
        self.iter_path = join(self.sweep_path, str(self.iter))
        os.mkdir(self.iter_path)
        os.chdir(self.iter_path)

        # Prepare input file
        self.ace_total = load.reconstitute_lists(self.ace_total, opt_values)
        self.ace_total = load.prep_dict_for_dump(self.ace_total)
        self.write_input_file(filename)

        subprocess.run(["pacemaker", filename])

        self.run_error_metrics(
            filename="../atomistic_errors.csv",
            metric=maths.get_rmse,
            n_scaling=1,
        )

        e_test, f_test = self.run_error_metrics(
            filename="../loss_function_errors.csv",
            metric=maths.get_rmse,
            n_scaling=0.5,
        )

        loss = self.calculate_loss(e_test, f_test)

        os.chdir(self.sweep_path)

        return loss

    def xval_fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        filename: str = "xpot-ace.yaml",
        xval_set: int = 0,
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
        self.iter = iteration
        self.iter_path = join(
            self.sweep_path, str(self.iter), "xval_" + str(xval_set)
        )
        if "/" in self.ace_total["data"]["filename"]:
            raise ValueError(
                "The data filename in the input file cannot be a"
                " path. The path must be defined in the datapath"
                " key of the input file."
            )

        self.ace_total["data"]["filename"] = f"xval_{xval_set}.pckl.gzip"
        self.ace_total["data"]["test_filename"] = (
            f"test_xval_{xval_set}.pckl.gzip"
        )

        os.mkdir(self.iter_path)
        os.chdir(self.iter_path)

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
        print(os.getcwd())
        df = pd.read_pickle(filename, compression="gzip")
        return df

    def validate_errors(
        self,
        test: bool = False,
        metric: callable = maths.get_rmse,
        xval: bool = False,
        xval_set: int = 0,
        n_exp: float = 0.5,
    ) -> tuple[float, float]:
        """
        Validate the potential from pickle files produced by :code:`pacemaker`
        during the fitting process.

        Parameters
        ----------

        test : bool
            If True, calculate validation errors, otherwise calculate training
            errors.
        metric : callable
            The error metric to use. Default is RMSE, MAE is also available.
            Look at the :code:`maths` module for more information.
        xval : bool
            If True, enable cross-validation handling and error calculation.
        xval_set : int
            The cross-validation set currently being calculated.
        n_scaling : float
            The exponent to which number of atoms is raised for error metrics
            to be passed to the loss function.

        Returns
        -------
        tuple
            The errors as a tuple of floats.
        """
        if test:
            errors = self.collect_raw_errors("test_pred.pckl.gzip")
        else:
            errors = self.collect_raw_errors("train_pred.pckl.gzip")

        n_per_structure = errors["NUMBER_OF_ATOMS"].values.tolist()

        ref_energy = [
            v / (n**n_exp)
            for v, n in zip(
                errors["energy_corrected"].values.tolist(), n_per_structure
            )
        ]
        pred_energy = [
            v / (n**n_exp)
            for v, n in zip(
                errors["energy_pred"].values.tolist(), n_per_structure
            )
        ]

        energy_diff = [pred - ref for pred, ref in zip(pred_energy, ref_energy)]

        ref_forces = np.concatenate(errors["forces"].to_numpy(), axis=None)
        pred_forces = np.concatenate(
            errors["forces_pred"].to_numpy(), axis=None
        )
        # print(ref_forces.shape, pred_forces.shape)
        # print(ref_forces)
        forces_diff = [pred - ref for pred, ref in zip(pred_forces, ref_forces)]

        energy_error = metric(energy_diff)
        forces_error = metric(forces_diff)

        return energy_error, forces_error

    def calculate_loss(
        self,
        e_error=float,
        f_error=float,
    ) -> float:
        """
        Determine the loss value from the validation errors.

        Parameters
        ----------
        e_error : float
            The overall energy error.
        f_error : float
            The overall force error.

        Returns
        -------
        float
            The loss value.
        """
        alpha = self.xpot["alpha"]

        return (alpha * e_error) + ((1 - alpha) * f_error)

    def write_error_file(
        self,
        e_train: float,
        f_train: float,
        e_test: float,
        f_test: float,
        filename: str,
    ) -> None:
        """
        Write the error values to a file.

        Parameters
        ----------
        e_train : float
            The energy training error.
        f_train : float
            The force training error.
        e_test : float
            The energy validation error.
        f_test : float
            The force validation error.
        filename : str
            The file to write to.
        """
        errors = [
            self.iter,
            e_train,
            e_test,
            f_train,
            f_test,
        ]
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(errors)

    def run_error_metrics(
        self,
        filename: str,
        metric: callable = maths.get_rmse,
        n_scaling: float = 0.5,
    ) -> tuple[float, float]:
        """
        Run an error metric sweep for the potential.

        Parameters
        ----------
        filename : str
            The file to write to.
        metric : callable
            The error metric to use. Default is RMSE, MAE is also available.
            Look at the :code:`maths` module for more information.
        n_scaling : float
            The exponent to which number of atoms is raised for error metrics
            to be passed to the loss function.

        Returns
        -------
        tuple
            The errors as a tuple of floats.
            Scaled by 1 / num_atoms ** n_scaling.
        """
        e_train, f_train = self.validate_errors(
            test=False, metric=metric, n_exp=n_scaling
        )
        e_test, f_test = self.validate_errors(
            test=True, metric=metric, n_exp=n_scaling
        )
        errors = [e_train, f_train, e_test, f_test]

        self.write_error_file(*errors, filename)

        return e_test, f_test
