from __future__ import annotations

import csv
import os
import subprocess
from os.path import join
from pathlib import Path

import numpy as np
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import iread

import xpot.loaders as load
import xpot.maths as maths

_this_file = Path(__file__).resolve()
_exec_path = Path(os.getcwd()).resolve()


class SNAP:
    def __init__(self, infile: str) -> None:
        """
        Class for optimizing SNAP potentials using XPOT, for use with FitSNAP3.

        Parameters
        ----------
        infile: str
            Path to input .hjson file containing XPOT and ML parameters.
        """

        defaults = load.get_defaults(
            join(_this_file.parent / "defaults" / "snap_defaults.json")
        )
        # Load input hyperparameters and remove the XPOT parameters.
        os.chdir(_exec_path)
        hypers = load.get_input_hypers(infile)
        self.xpot = hypers["xpot"]
        hypers.pop("xpot")

        self.val_set = self.xpot["validation_data"]
        self.train_set = self.xpot["training_data"]

        self.project = self.xpot["project_name"]
        self.sweep = self.xpot["sweep_name"]
        self.sweep_path = join(_exec_path / self.project / self.sweep)
        os.makedirs(self.sweep_path)

        self.snap_total = load.merge_hypers(defaults, hypers)
        load.validate_subdict(self.snap_total, hypers)
        self.optimisation_space = load.get_optimisable_params(self.snap_total)

    def prepare_group_params(
        self,
        params: load.NestedDict,
    ) -> load.NestedDict:
        """
        Prepare the parameters for the SNAP potential, converting the groups
        block into the requisite format.

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameters.

        Returns
        -------
        dict
            Dictionary of hyperparameters.
        """
        group_param_keys = self.snap_total["[GROUPS]"].keys()
        exclusions = [
            "group_sections",
            "group_types",
            "random_sampling",
            "smartweights",
            "BOLTZT",
        ]

        group_blocks = {
            key: self.snap_total["[GROUPS]"][key]
            for key in group_param_keys
            if key not in exclusions
        }

        new_groups = load.list_to_string(group_blocks)
        print(new_groups)
        final_groups = {
            "[GROUPS]": load.merge_hypers(
                self.snap_total["[GROUPS]"], new_groups
            )
        }
        return load.merge_hypers(self.snap_total, final_groups)

    def write_input_file(
        self,
        filename: str = "xpot-snap.yaml",
    ) -> None:
        """
        Write the input file for the SNAP potential from the hyperparameter +
        dictionary.

        Parameters
        ----------
        filename : str
            Path to the input file.
        """
        self.snap_total = load.trim_empty_values(self.snap_total)
        self.snap_total = self.prepare_group_params(self.snap_total)
        with open(filename, "w+") as f:
            for k, v in self.snap_total.items():
                if isinstance(v, dict):
                    f.write(f"{k}\n")
                    for k1, v1 in v.items():
                        f.write(f"{k1} = {v1}\n")
                    f.write("\n")
                else:
                    f.write(f"{k} = {v}\n")

    def fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        filename: str = "xpot-snap.yaml",
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
        self.snap_total = load.reconstitute_lists(self.snap_total, opt_values)
        self.snap_total = load.prep_dict_for_dump(self.snap_total)
        # self.snap_total = load.list_to_string(self.snap_total)
        self.write_input_file(filename)

        subprocess.run(["python", "-m", "fitsnap3", filename])

        # Process errors & return loss
        e_err, f_err = self.run_error_metrics()

        loss = self.calculate_loss(e_err, f_err)

        os.chdir(self.sweep_path)

        return loss

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
        for atoms in structures:
            a = self.determine_lmp_args()
            lammps = LAMMPSlib(lmpcmds=a)
            ref_e = atoms.info["energy"]
            ref_f = list(atoms.arrays["forces"].flat)
            n_atoms.append(len(atoms))
            atoms.calc = lammps
            e = atoms.get_potential_energy() - ref_e
            f = np.subtract(list(atoms.get_forces().flat), ref_f)
            delta_e.append(e)
            delta_f.append(f)

        d = {
            "energies": delta_e,
            "forces": delta_f,
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

    def determine_lmp_args(self) -> list[str]:
        """
        Determine the LAMMPS arguments to be used for the SNAP potential
        calculator during validation.

        Returns
        -------
        list
            List of LAMMPS commands compatible with ASE LAMMPSlib calculator.
        """

        pot = self.snap_total["[OUTFILE]"]["potential"]

        with open(f"{pot}.mod") as f:
            lmp_cmds = [i.strip("\n") for i in f.readlines()[3:]]

        return lmp_cmds

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
