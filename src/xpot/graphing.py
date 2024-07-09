import csv

import matplotlib.pyplot as plt
from ase.atoms import Atoms


def plot_convergence(
    data: str,
    format: str = "png",
) -> None:
    """
    Plot the line graph with points for the convergence of the optimisation.
    Saves graph to a file.

    Parameters
    ----------
    data : str
        File path to the CSV file.
    """
    with open(data) as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)

    iteration = [int(i[0]) for i in data]
    loss = [float(i[1]) for i in data]

    plt.plot(iteration, loss, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Convergence")

    plt.savefig(f"{data.split(".")[0]}.{format}")


def plot_e_f_convergence(
    conv_data: str,
    e_f_data: str,
    format: str = "png",
    style: str = None,
) -> None:
    """
    Plot line graphs for the convergence, energy, and force errors throughout
    optimization. Saves the graph to a file.

    Parameters
    ----------
    conv_data : str
        File path to the convergence data CSV file.
    e_f_data : str
        File path to the energy and force error data CSV file.
    format : str
        File format to save the graph as.
    style : str
        Path to a matplotlib style file.
    """
    if style:
        plt.style.use(style)

    with open(conv_data) as f:
        reader = csv.reader(f)
        next(reader)
        conv_data = list(reader)

    with open(e_f_data) as f:
        reader = csv.reader(f)
        next(reader)
        e_f_data = list(reader)

    iteration = [int(i[0]) for i in conv_data]
    loss = [float(i[1]) for i in conv_data]
    e_val = [float(i[2]) * 1000 for i in e_f_data]
    f_val = [float(i[4]) for i in e_f_data]

    # Plot energy, force, and convergence on subplots stacked vertically.
    fig, axs = plt.subplots(3)
    fig.suptitle("Convergence and Errors")
    axs[0].plot(iteration, loss)
    axs[0].set_ylabel("Loss")
    axs[1].plot(iteration, e_val)
    axs[1].set_ylabel("Energy Error (meV/at.)")
    axs[2].plot(iteration, f_val)
    axs[2].set_ylabel("Force Error (eV/Å)")
    axs[2].set_xlabel("Iteration")

    def plot_dimers(
        method: str,
        potential: str,
        elements: list[str],
        min_sep: float,
        max_sep: float,
        step: float,
    ) -> None:
        """

        Plot the energy of a dimer scan for a given potential.

        Parameters
        ----------
        method : str
            The class for the potential [SNAP, ACE, GAP, etc.].
        potential : str
            For ACE/GAP: the path to the potential file.
            For SNAP: the string of the LAMMPS setup for LAMMPSlib.
        elements : list[str]
            List of elements to consider in the dimer scans. e.g. ["Si", "C"]
        min_sep : float
            Minimum separation between atoms in the dimer scan.
        max_sep : float
            Maximum separation between atoms in the dimer scan.
        step : float
            Step size between min_sep and max_sep.

        Returns
        -------
        None
        """
        if method == "GAP":
            from quippy.potential import Potential

            calc = Potential(param_filename=potential)

        elif method == "ACE":
            from pyace import PyACECalculator

            calc = PyACECalculator(potential)

        elif method == "SNAP":
            from ase.calculators.lammpslib import LAMMPSlib

            calc = LAMMPSlib(**potential)

        # Create a list of all possible dimers from the elements list.
        dimer_scans = [
            [a, b]
            for idx, a in enumerate(elements)
            for b in elements[idx + 1 :]
        ]

        for dim in dimer_scans:
            energies = []
            distances = [i for i in range(min_sep, max_sep, step)]
            for dist in distances:
                atoms = Atoms(
                    positions=[[0, 0, 0], [0, 0, dist]],
                    symbols=dim,
                    calculator=calc,
                )
                energies.append(atoms.get_potential_energy() / 2)

            plt.plot(distances, energies, label=f"{dim[0]}-{dim[1]} dimer scan")
            plt.savefig(
                f"{potential.split('.')[0]}_dimer_scan_{dim[0]-dim[1]}.png"
            )
