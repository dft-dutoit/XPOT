from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from ase.calculators.lammpslib import LAMMPSlib

import xpot.loaders as load
import xpot.maths as maths
from xpot.models.model import MLP

_this_file = Path(__file__).resolve()
_exec_path = Path(os.getcwd()).resolve()


class SNAP(MLP):
    def __init__(self, infile: str) -> None:
        """
        Class for optimizing SNAP potentials using XPOT, for use with FitSNAP3.

        Parameters
        ----------
        infile: str
            Path to input .hjson file containing XPOT and ML parameters.
        """

        super().__init__(infile, "snap_defaults.json")
        if not isinstance(self.mlp_total, dict):
            raise ValueError(
                """Model error: mlp_total is not a dictionary!
                Initialisation failed."""
            )

        self.val_set = self.xpot["validation_data"]
        self.train_set = self.xpot["training_data"]

    def prepare_group_params(
        self,
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
        group_param_keys = self.mlp_total.get("[GROUPS]", {}).keys()  # type: ignore
        exclusions = [
            "group_sections",
            "group_types",
            "random_sampling",
            "smartweights",
            "BOLTZT",
        ]

        group_blocks = {
            key: self.mlp_total["[GROUPS]"][key]  # type: ignore
            for key in group_param_keys
            if key not in exclusions
        }

        new_groups = load.list_to_string(group_blocks)
        print(new_groups)
        final_groups = {
            "[GROUPS]": load.merge_hypers(
                self.mlp_total["[GROUPS]"],  # type: ignore
                new_groups,
            )
        }
        return load.merge_hypers(self.mlp_total, final_groups)

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
        self.mlp_total = load.trim_empty_values(self.mlp_total)
        self.mlp_total = self.prepare_group_params()
        with open(filename, "w+") as f:
            for k, v in self.mlp_total.items():
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
        self.prep_fit(opt_values, iteration)
        self.mlp_total = self.prepare_group_params()
        self.write_input_file(filename)

        subprocess.run(["python", "-m", "fitsnap3", filename])

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

    def calculate_errors(
        self,
        validation: bool = True,
    ) -> dict[str, list[float]]:
        """
        Calculate test errors for FitSNAP potential on testing set.

        Parameters
        ----------
        validation : bool
            Whether to calculate validation errors.

        Returns
        -------
        dict
            Dictionary of errors.
        """
        if validation:
            structures = load.load_structures(str(self.val_set))
        else:
            structures = load.load_structures(str(self.train_set))

        n_per_structure = []
        ref_energy = []
        ref_forces = []
        pred_energy = []
        pred_forces = []

        a = self.determine_lmp_args()
        lammps = LAMMPSlib(lmpcmds=a)
        for atoms in structures:
            n_per_structure.append(len(atoms))
            ref_energy.append(atoms.get_potential_energy())
            ref_forces.extend(atoms.get_forces().ravel())

            atoms.calc = lammps
            pred_energy.append(atoms.get_potential_energy())
            pred_forces.extend(atoms.get_forces().ravel())

        energy_diff = [pred - ref for pred, ref in zip(pred_energy, ref_energy)]
        forces_diff = [pred - ref for pred, ref in zip(pred_forces, ref_forces)]
        forces_diff = np.concatenate(forces_diff).ravel().tolist()

        errors = {
            "energies": energy_diff,
            "forces": forces_diff,
            "atom_counts": n_per_structure,
        }

        return errors

    def determine_lmp_args(self) -> list[str]:
        """
        Determine the LAMMPS arguments to be used for the SNAP potential
        calculator during validation.

        Returns
        -------
        list
            List of LAMMPS commands compatible with ASE LAMMPSlib calculator.
        """

        pot = self.mlp_total["[OUTFILE]"]["potential"]  # type: ignore

        with open(f"{pot}.mod") as f:
            lmp_cmds = [i.strip("\n") for i in f.readlines()[3:]]

        return lmp_cmds
