Models
======

Documentation for each of the models used in XPOT. Each
class has the same base functionality which can be described as follows:

#. Read input file, sort into optimisable and unoptimisable hyperparameters.
#. Export optimisable hyperparameters to the optimiser.
#. Read optimiser output and write model fitting input. 
#. Fit the model.
#. Collect validation errors from the model.
#. Pass the output to the optimiser object and write output files.

.. toctree::
   :titlesonly:
   
   ace
   snap
   gap