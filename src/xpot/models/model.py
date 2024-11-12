from __future__ import annotations

import os
from collections.abc import Callable
from os.path import join
from pathlib import Path

import numpy as np

import xpot.loaders as load
import xpot.maths as maths

_this_file = Path(__file__).resolve()
_exec_path = Path(os.getcwd()).resolve()


class MLP:
    def __init__(self, infile: str, default_file: str) -> None:
        """
        Parent class for all MLPs

        Parameters
        ----------
        infile : str
            Path to input .hjson file containing XPOT and ML parameters.
        defaults : str
            Name of input .json file containing default parameters for the MLP.
        """

        defaults = load.get_defaults(
            join(_this_file.parent / "defaults" / default_file)
        )

        os.chdir(_exec_path)
        hypers = load.get_input_hypers(infile)
        self.xpot = hypers["xpot"]
        self.project = str(self.xpot["project_name"])
        self.sweep = str(self.xpot["sweep_name"])
        self.alpha = float(self.xpot["alpha"])  # type: ignore
        self.sweep_path = join(_exec_path / self.project / self.sweep)
        os.makedirs(self.sweep_path)

        hypers.pop("xpot")
        self.mlp_total = load.merge_hypers(defaults, hypers)
        load.validate_subdict(self.mlp_total, hypers)
        self.optimisation_space = load.get_optimisable_params(self.mlp_total)

    def write_input_file(self) -> None:
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )

    def prep_fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
    ) -> None:
        """
        Call the relevant fitting routine for the MLP architecture required.

        Parameters
        ----------
        opt_values : dict
            Dictionary of parameter names and values returned by the optimiser
            for the current iteration of fitting.
        iteration : int
            The current iteration number.
        """
        self.iter = iteration
        self.iter_path = join(self.sweep_path, str(self.iter))
        os.mkdir(self.iter_path)
        os.chdir(self.iter_path)

        self.mlp_total = load.reconstitute_lists(self.mlp_total, opt_values)
        self.mlp_total = load.prep_dict_for_dump(self.mlp_total)
        self.mlp_total = load.trim_empty_values(self.mlp_total)  # type: ignore
        self.mlp_total = load.convert_numpy_types(self.mlp_total)

    def fit(self) -> None:
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )

    def collect_raw_errors(self) -> None:
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )

    def validate_errors(
        self,
        errors: dict[str, list[float]],
        metric: Callable = maths.get_rmse,
        n_scaling: float | list[float] = 0.5,
    ) -> list[tuple[float, float]]:
        """
        Calculate the training and validation error values specific to the loss
        function of XPOT from the MLP.

        Parameters
        ----------
        errors : tuple
            Tuple of training and validation errors.
        metric : Callable
            The error metric to use. Default is RMSE, MAE is also available.
        """
        if isinstance(n_scaling, float):
            energy_diff = (
                errors["energies"]
                / np.array(errors["atom_counts"]) ** n_scaling
            )
            forces_diff = errors["forces"]
            print(forces_diff)
            energy_error = metric(energy_diff)
            forces_error = metric(forces_diff)

            return [(energy_error, forces_error)]

        elif isinstance(n_scaling, list):
            energy_diffs = []
            for i in n_scaling:
                energy_diff = (
                    errors["energies"] / np.array(errors["atom_counts"]) ** i
                )
                energy_error = metric(energy_diff)
                energy_diffs.append(energy_error)
            forces_diff = errors["forces"]
            forces_error = metric(forces_diff)
            forces_diffs = [forces_error] * len(energy_diffs)
            output = []
            for i in range(len(energy_diffs)):
                output.append((energy_diffs[i], forces_diffs[i]))
            return output

        else:
            raise ValueError("n_scaling must be a float or list of floats.")

    def process_errors(
        self,
        train_errors: tuple[float, float],
        test_errors: tuple[float, float],
        filename: str,
    ) -> float:
        """
        Write the error metrics to a file.

        Parameters
        ----------
        train_errors : tuple
            Tuple of training errors.
        test_errors : tuple
            Tuple of test errors.
        filename : str
            The file to write to.
        """
        errors = [
            train_errors[0],
            test_errors[0],
            train_errors[1],
            test_errors[1],
        ]
        load.write_error_file(self.iter, errors, filename)

        loss = maths.calculate_loss(test_errors[0], test_errors[1], self.alpha)
        os.chdir(self.sweep_path)
        return loss
