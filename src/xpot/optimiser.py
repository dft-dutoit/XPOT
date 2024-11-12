from __future__ import annotations

import csv
import os
import pickle
from pathlib import Path
from typing import Generic, TypeVar, Union

import skopt
from matplotlib import pyplot as plt
from skopt import plots
from tabulate import tabulate

import xpot.loaders as load

DictValue = str | list[str, int] | int | float
NestedDict = dict[str, Union["NestedDict", DictValue]]

Key = TypeVar("Key")

_exec_path = Path(os.getcwd()).resolve()


class NamedOptimiser(Generic[Key]):
    def __init__(
        self,
        optimisable_params: dict[Key, skopt.space.Dimension],
        sweep_path: str,
        skopt_kwargs: dict[str, str | int | float] = None,
    ):
        """
        The NamedOptimiser class is a wrapper around the skopt.Optimiser class
        that allows for the use of named parameters, and implements the
        ask-tell interface, dump and load methods, and result recording.

        This class can be used to initialise any optimiser to be used by XPOT
        for optimising hyperparameters for fitting ML potentials. This class is
        used for all classes

        Parameters
        ----------
        optimisable_params : dict
            Dictionary of parameter names and skopt.space.Dimension objects.
        skopt_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the skopt.Optimiser
            class, by default None. You should define any non-default parameters
        """
        self._optimiser = skopt.Optimizer(
            dimensions=list(optimisable_params.values()),
            random_state=42,
            **skopt_kwargs,
        )
        self.sweep_path = sweep_path
        self._optimisable_params = optimisable_params
        keys = [" ".join(i[0]) for i in self._optimisable_params]
        load.initialise_csvs(sweep_path, keys)
        self.iter = 1

    def ask(self) -> dict[Key, DictValue]:
        """
        Ask the optimizer for a new set of parameters based on the current
        results of the optimisation.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        param_values: list[float | int | str] = self._optimiser.ask()  # type: ignore
        return {
            name: value
            for name, value in zip(
                self._optimisable_params.keys(), param_values
            )
        }

    def tell(self, params: dict[Key, DictValue], result: float) -> None:
        """
        Tell the optimiser the result of the last iteration, as well as the
        parameter values used to achieve it.

        Parameters
        ----------
        params : dict
            Dictionary of parameter names and values.
        result : float
            Result (loss value) of the last iteration.
        """

        # 1. make sure that we get the order correct
        locations = [params[name] for name in self._optimisable_params]
        # 2. tell the optimiser
        self._optimiser.tell(locations, result)

    def dump_optimiser(self, path: str) -> None:
        """
        Dump the optimiser to a file.

        Parameters
        ----------
        path : str
            Path of directory to write file to.
        """
        with open(f"{path}/xpot-optimiser.pkl", "wb") as f:
            pickle.dump(self._optimiser, f)

    def load_optimiser(self, path: str) -> None:
        """
        Load the optimiser from a file.

        Parameters
        ----------
        path : str
            File path.
        """
        with open(path, "rb") as f:
            self._optimiser = pickle.load(f)

        if len(self._optimiser.space.dimensions) != len(
            self._optimisable_params
        ):
            raise ValueError(
                "The optimiser and the optimisable parameters "
                "have different lengths. The optimiser cannot be "
                "loaded."
            )

    def initialise_csvs(self, path: str) -> None:
        keys = [" ".join(i[0]) for i in self._optimisable_params]
        with open(f"{path}/parameters.csv", "w+") as f:
            f.write("iteration,loss," + ",".join(map(str, keys)) + "\n")
        with open(f"{path}/atomistic_errors.csv", "w+") as f:
            f.write(
                "Iteration,"
                + "Train Δ Energy,"
                + "Test Δ Energy,"
                + "Train Δ Force,"
                + "Test Δ Force"
                + "\n"
            )
        with open(f"{path}/loss_function_errors.csv", "w+") as f:
            f.write(
                "Iteration,"
                + "Train Δ Energy,"
                + "Test Δ Energy,"
                + "Train Δ Force,"
                + "Test Δ Force"
                + "\n"
            )
        print("Initialised CSV Files")

    def write_param_result(
        self,
        path: str,
        iteration: int,
    ) -> None:
        """
        Write the current iteration and loss to the parameters.csv file.

        Parameters
        ----------

        path : str
            Path of directory with parameters.csv file.
        iteration : int
            Current iteration.
        loss : float
            Loss of current iteration.
        params : dict
            List of parameter values for the current iteration.
        """
        # Raise error if there is no parameters.csv file
        if not os.path.isfile(f"{path}/parameters.csv"):
            raise FileNotFoundError(
                f"parameters.csv file does not exist at {path}"
            )
        with open(f"{path}/parameters.csv", "a") as f:
            f.write(
                f"{iteration},"
                + f"{self._optimiser.yi[-1]},"
                + ",".join([str(i) for i in self._optimiser.Xi[-1]])
                + "\n"
            )
        print(f"Iteration {iteration} written to parameters.csv")

    def tabulate_final_results(
        self,
        path: str,
    ) -> None:
        """
        Tabulate the final results of the optimisation into pretty tables, with
        filenames

        Parameters
        ----------
        path : str
            Path of directory for all error files.
        """

        def tabulate_csv(file):
            with open(f"{file}.csv") as csv_file:
                reader = csv.reader(csv_file)
                rows = [row for row in reader]
                table = tabulate(rows, headers="firstrow", tablefmt="github")
            with open(f"{file}_final", "a+") as f:
                f.write(table)

        tabulate_csv(f"{path}/parameters")
        tabulate_csv(f"{path}/atomistic_errors")
        tabulate_csv(f"{path}/loss_function_errors")

    def run_optimisation(
        self,
        objective: callable,
        path=_exec_path,
        **kwargs,
    ) -> float:
        """
        Function for running optimisation sweep.

        Parameters
        ----------

        objective : callable
            Function to be optimised. Must return a float (loss value).
        path : str, optional
            Path of directory to get files from, by default "./".
        **kwargs
            Keyword arguments to pass to objective function.

        Returns
        -------
        loss
            Loss value of the current iteration.
        """
        next_params = self.ask()
        loss = objective(next_params, iteration=self.iter, **kwargs)
        self.tell(next_params, loss)
        self.write_param_result(path, self.iter)
        self.iter += 1
        return loss

    def plot_results(self, path: str) -> None:
        """
        Function to create scikit-optimize results using inbuilt functions.

        Parameters
        ----------
        path : str
            Path of directory to save plots to.
        """
        data = self._optimiser.get_result()
        a = plots.plot_objective(data, levels=20, size=3)
        plt.tight_layout()
        a.figure.savefig(f"{path}/objective.pdf")

        b = plots.plot_evaluations(data)
        b.figure.savefig(f"{path}/evaluations.pdf")
