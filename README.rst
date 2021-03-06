.. image:: https://badge.fury.io/py/hypermodern-screening.svg
  :target: https://pypi.org/project/hypermodern-screening

.. image:: https://github.com/tostenzel/hypermodern-screening/workflows/Continuous%20Integration/badge.svg?branch=master
  :target: https://github.com/tostenzel/hypermodern-screening/actions

.. image:: https://readthedocs.org/projects/hypermodern-screening/badge/?version=latest
   :target: https://hypermodern-screening.readthedocs.io/en/latest/?badge=latest

.. image:: https://codecov.io/gh/tostenzel/hypermodern-screening/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/tostenzel/hypermodern-screening

.. image:: https://api.codacy.com/project/badge/Grade/87ad6f0069fd45d9afd5ad97a579329b
   :alt: Codacy Badge
   :target: https://app.codacy.com/manual/tostenzel/hypermodern-screening?utm_source=github.com&utm_medium=referral&utm_content=tostenzel/hypermodern-screening&utm_campaign=Badge_Grade_Dashboard


|

The ``hypermodern-screening`` package provides tools for efficient global sensitivity analyses based on elementary effects. Its unique feature is the option to compute these effects for models with correlated input parameters. The underlying conceptual approach is developed by Stenzel (2020). The fundamental idea comes from Ge and Menendez (2017). It is the combination of inverse transform sampling with an intelligent juggling of parameter positions in the input vector to create different dependency hierarchies. The package does also include a variety of sampling methods.

The name ``hypermodern-screening`` is inspired by the brilliant series of `blog <https://cjolowicz.github.io/posts/>`_ articles about cutting-edge tools for python development by Claudio Jolowicz in 2020. He calls his guide "Hypermodern Python"`*`. Its corner stones are ``poetry`` for packaging and dependency management and ``nox`` for automated testing. Another recommendation is rigorous typing. The repository framework widely follows the blog series.

Read the documentation `here <https://hypermodern-screening.readthedocs.io>`_ and install ``hypermodern-screening`` from PyPI with

.. code-block:: bash

    $ pip install hypermodern_screening


.. image:: docs/images/albert_robida_1883.jpg
   :width: 40pt

`**`

References
~~~~~~~~~~

    Stenzel, T. (2020): `Uncertainty Quantification for an Eckstein-Keane-Wolpin Model with
    Correlated Input Parameters <https://github.com/tostenzel/thesis-projects-tostenzel/blob/master/latex/main.pdf>`_.
    *Master's Thesis, University of Bonn*.

    Ge, Q. and Menendez, M. (2017). `Extending Morris method for qualitative global sensitivity
    analysis of models with dependent inputs <https://doi.org/10.1016/j.ress.2017.01.010>`_. *Reliability Engineering & System Safety 100(162)*,
    28-39.

Quick start
~~~~~~~~~~~

.. code-block:: bash

    import hypermodern_screening as hms

    # Define the model in pseudocode.
    def qoi_model(input_parameters)
        return qoi

    # Generate list of input samples in radial design.
    rad_list, step_list = hms.radial_sample(n_sample, n_inputs, normal=True)

    # Compute uncorrelated and correlated elementary effects and statistics thereof.
    measures_list, ees_list = hms.screening_measures(
        qoi_model,
        rad_list,
        step_list,
        cov_inputs,
        mu_inputs,
        radial=True
    )

    # Compute sigma-normalized statistics of elementary effects.
    measures_sigma_norm = hms.compute_measures(ees_list, sd_qoi, sd_inputs, sigma_norm=True)

|

-----

`*`: Claudio, in turn, was inspired by the chess book "Die hypermoderne Schachpartie" (1925) by Savielly Tartakower.

`**`: The image is a detail from the photogravure *Paris by night* by Albert Robida, 1883 (via `Old Book Illustrations <https://www.oldbookillustrations.com/illustrations/paris-night>`_).
