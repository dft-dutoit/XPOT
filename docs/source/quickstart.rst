Installation
============

Standalone Installation
-----------------------

For installation, the :code:`pip` package manager is recommended:

.. code-block:: console

    $ pip install xpot

Due to the fact that :code:`xpot` is a wrapper around various ML potential fitting 
codes, you will need to install the fitting packages relevant for you. 

Currently supported fitting codes:
- `pacemaker <https://pacemaker.readthedocs.io/en/latest/pacemaker/install/>`_
- `FitSNAP <https://fitsnap.github.io/Installation.html>`_
- `GAP <https://github.com/libAtoms/QUIP>`_

In order to fit potentials using any of these codes, you will need to install 
them in the same environment as :code:`xpot`. All other dependencise will be
installed automatically with :code:`xpot`.

In order to test whether xpot has installed, you can run the following
code in a python shell:

.. code-block:: python

    from xpot.loaders import load_defaults
    from xpot.maths import get_rmse

    defaults = load_defaults("/path/to/any/valid/json")
    print(defaults)
    test_list = [5, 5, 4]
    print(get_rmse(test_list))

If the installation was successful, you should see the json file printed to
console, as well as a return of 22 for the RMSE. To test the fitting codes, you
will first need to install them, and then run the relevant jupyter notebooks 
included in the :code:`xpot` repository.

ACE Functionality
-----------------

For ACE functionality, you need to install :code:`pacemaker`. The package (and 
installation steps) are available at `this link 
<https://pacemaker.readthedocs.io/en/latest/pacemaker/install/>`_. The
installation steps can be completed either before or after pip installation of
:code:`xpot`. 

.. warning::
    In our experience, it is required to run: :code:`pip install protobuf==3.20.*`

    after installation of :code:`pacemaker`.

SNAP Functionality
------------------
SNAP functionality is provided by the :code:`fitsnap` package. The package (and
installation steps) are available at `this link 
<https://fitsnap.github.io/Installation.html>`_. If not installing via :code:`pip` 
you must use the following: :code:`export PYTHONPATH=/path/to/fitsnap:PYTHONPATH`

SNAP functionality also requires :code:`LAMMPS` to be installed. It must be 
compiled with ML_SNAP and PYTHON packages enabled. Steps are laid out in the 
:code:`fitsnap` documentation, but also at the `LAMMPS website 
<https://docs.lammps.org/Python_install.html>`_.

GAP Functionality
-----------------

GAP functionality is provided by :code:`QUIP`. The package (and installation
steps) are available at `this link <https://github.com/libAtoms/QUIP>`_. 
