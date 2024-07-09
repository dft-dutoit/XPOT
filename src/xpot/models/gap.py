from __future__ import annotations

import csv
import os
import subprocess
from os.path import join
from pathlib import Path

import numpy as np
from ase.io import iread
from quippy.potential import Potential

import xpot.loaders as load
import xpot.maths as maths

_this_file = Path(__file__).resolve()
_exec_path = Path(os.getcwd()).resolve()


class GAP:
    def __init__(self, infile: str) -> None:
        """
        Class for optimizing ACE potentials using XPOT, for use with pacemaker.

        Parameters
        ----------
        infile: str
            Path to input .hjson file containing XPOT and ML parameters.
        """

        defaults = load.get_defaults(
            join(_this_file.parent / "defaults" / "gap_defaults.json")
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

        self.val_set = self.xpot["validation_data"]

        self.gap_cmd = self.xpot["fitting_executable"]

        self.gap_total = load.merge_hypers(defaults, hypers)
        load.validate_subdict(self.gap_total, hypers)
        self.train_set = self.gap_total["at_file"]
        self.optimisation_space = load.get_optimisable_params(self.gap_total)

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

        self.gap_total = load.trim_empty_values(self.gap_total)

        gap_cmd = ""

        for k, v in self.gap_total.items():
            if k == "gap":
                gap_cmd += "gap={"
                for term, term_dict in v.items():
                    gap_cmd += f"{term} "
                    for term_params, param_vals in term_dict.items():
                        gap_cmd += f"{term_params}={param_vals} "
                    gap_cmd += ": "

                gap_cmd = gap_cmd[:-2] + "}"
            else:
                gap_cmd += f"{k}={v} "

        return gap_cmd

    def calculate_val_errors(
        self,
        validation: bool = True,
        xval: bool = False,
        xval_set: int = 0,
    ) -> dict[str:list]:
        """
        Calculate test errors for FitSNAP potential on testing set.

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
            structures = iread(self.val_set)
        else:
            structures = iread(self.train_set)

        delta_e = []
        delta_f = []
        n_atoms = []
        quip = Potential(param_filename=self.gap_total["gap_file"])
        for atoms in structures:
            ref_e = atoms.info["energy"]
            ref_f = list(atoms.arrays["forces"].flat)
            n_atoms.append(len(atoms))
            atoms.calc = quip
            e = atoms.get_potential_energy() - ref_e
            f = np.subtract(list(atoms.get_forces().flat), ref_f)
            delta_e.append(e)
            delta_f.append(f)
        d = {
            "energies": delta_e,
            "forces": [x for xs in delta_f for x in xs],
            "atom_counts": n_atoms,
        }

        return d

    def collect_errors(
        self,
        errors: dict[str:list],
        metric: callable = maths.get_rmse,
        xval: bool = False,
        xval_set: int = 0,
        n_exp: float = 0.5,
    ) -> tuple[float, float]:
        """
        Collect the errors from the validation set and calculate the overall
        energy and force errors.

        Parameters
        ----------
        errors : dict
            Dictionary of errors.
        metric : callable
            The error metric to use. Check :code:`xpot.maths` for available
            metrics.
        xval : bool
            Whether cross-validation is being used.
        xval_set : int
            The cross-validation set in use.
        n_exp : float
            The exponent to scale the number of atoms by for error metrics.

        Returns
        -------
        tuple(float, float)
            The energy and force errors.
        """

        energy_error = [
            v / n**n_exp
            for v, n in zip(errors["energies"], errors["atom_counts"])
        ]

        force_error = errors["forces"]

        energy_error = metric(energy_error)
        force_error = metric(force_error)

        return energy_error, force_error

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
        metric: callable = maths.get_rmse,
        n_exp: float = 0.5,
    ) -> tuple[float, float]:
        """
        Run the error metrics on the validation set.

        Parameters
        ----------
        filename : str
            The file to write the error metrics to.
        metric : callable
            The error metric to use.
        n_scaling : float
            The exponent to scale the number of atoms by for error metrics.

        Returns
        -------
        tuple(float, float)
            The energy and force errors.
        """
        val_errors = self.calculate_val_errors()
        e_err, f_err = self.collect_errors(val_errors, metric, n_exp)
        at_e_err, at_f_err = self.collect_errors(val_errors, metric, n_exp=1)

        train_errors = self.calculate_val_errors(validation=False)
        e_train, f_train = self.collect_errors(train_errors, metric, n_exp)
        at_e_train, at_f_train = self.collect_errors(
            train_errors, metric, n_exp=1
        )

        errors = [e_train, f_train, e_err, f_err]
        at_errors = [at_e_train, at_f_train, at_e_err, at_f_err]

        self.write_error_file(*errors, "../loss_function_errors.csv")
        self.write_error_file(*at_errors, "../atomistic_errors.csv")

        return e_err, f_err

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
        self.iter = iteration
        self.iter_path = join(self.sweep_path, str(self.iter))
        os.mkdir(self.iter_path)
        os.chdir(self.iter_path)

        # Prepare input file
        self.gap_total = load.reconstitute_lists(self.gap_total, opt_values)
        self.gap_total = load.prep_dict_for_dump(self.gap_total)
        self.gap_total = self.list_to_reg(self.gap_total)
        cmd_params = self.write_input_file()
        # print(self.gap_cmd, cmd_params)

        subprocess.run(f"{self.gap_cmd} {cmd_params}", shell=True)

        # Process errors & return loss
        e_err, f_err = self.run_error_metrics()

        loss = self.calculate_loss(e_err, f_err)

        os.chdir(self.sweep_path)

        return loss
