Documentation of the `hypermodern-screening` package
====================================================


The ``hypermodern-screening`` package provides tools for efficient global sensitivity analyses based on elementary effects. Its unique feature is the option to compute these effects for models with correlated input parameters. The underlying conceptual approach is developed by Stenzel (2020). The fundamental idea comes from Ge and Menendez (2017). It is the combination of inverse transform sampling with an intelligent juggling of parameter positions in the input vector to create different dependency hierarchies. The package does also include a variety of sampling methods.


Install ``hypermodern-screening`` from PyPI with

.. code-block:: bash

    $ pip install hypermodern_screening

|

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~~

The documentation consists of two main parts. The first part is close to the implementation and the second part provides some background starting from basic definitions of uncertainty qunatification.

The first part describes the concepts that are implemented in this package. It comprises three sections. These are :doc:`sampling schemes <sampling>` tailored to the computation of elementary effects (EEs), :doc:`EEs <ee>` for correlated parameters and the importance of :doc:`sigma normalization <sigma_norm>`.
The first two sections are accompanied by tutorials written in jupyter notebooks that demonstrate the usage of the package:

- Sampling Schemes:

.. image:: https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464
 :target: https://nbviewer.jupyter.org/github/tostenzel/hypermodern-screening/blob/documentation/docs/notebooks/sampling.ipynb

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/tostenzel/hypermodern-screening/documentation?filepath=docs%2Fnotebooks%2Fsampling.ipynb

- Elementary Effects:

.. image:: https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464
 :target: https://nbviewer.jupyter.org/github/tostenzel/hypermodern-screening/blob/documentation/docs/notebooks/elementary_effects.ipynb

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/tostenzel/hypermodern-screening/documentation?filepath=docs%2Fnotebooks%2Felementary_effects.ipynb

|

The second part embedds EEs within uncertainty quantification and, in particular, sensitivity analysis. Thereby, the motivation to compute EEs is clarified. It also contains a short discussion of methods for models with correlated input parameters.


.. toctree::
   :hidden:
   :maxdepth: 1

   sampling
   ee
   sigma_norm
   uq
   sa
   quantgsa
   qualgsa
   corrs
   references
   license
   modules

|

References
~~~~~~~~~~

    Stenzel, T. (2020): `Uncertainty Quantification for an Eckstein-Keane-Wolpin model with
    correlated input parameters <https://github.com/tostenzel/thesis-projects-tostenzel/blob/master/latex/main.pdf>`_.
    *Master's thesis, University of Bonn*.

    Ge, Q. and Menendez, M. (2017). `Extending Morris method for qualitative global sensitivity
    analysis of models with dependent inputs <https://doi.org/10.1016/j.ress.2017.01.010>`_. *Reliability Engineering & System Safety 100(162)*,
    28-39.
