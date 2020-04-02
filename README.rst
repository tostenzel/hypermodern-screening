

.. image:: https://github.com/tostenzel/hypermodern-screening/workflows/Tests/badge.svg
  :target: https://github.com/tostenzel/hypermodern-screening/actions

.. image:: https://codecov.io/gh/tostenzel/hypermodern-screening/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/tostenzel/hypermodern-screening

.. image:: https://readthedocs.org/projects/hypermodern-screening/badge/?version=latest
   :target: https://hypermodern-screening.readthedocs.io/en/latest/?badge=latest



The ``hypermodern-screening`` package provides tools for efficient global sensitivity analyses based on elementary effects. Its unique feature is the possibility to compute these effects for models with correlated input parameters. The underlying conceptual approach is developed by Stenzel (2020). The fundamental idea comes from Ge and Menendez (2017). It is the combination of inverse transform sampling with an intelligent juggling of parameter positions in the input vector to create different dependency hierarchies. The package does also include a variety of sampling methods.

The name ``hypermodern-screening`` is inspired by the brilliant series of blog articles about cutting-edge tools for python development by Claudio Jolowicz in 2020. He calls his guide "Hypermodern Python". [*]_ Its corner stones are ``poetry`` for packaging and dependency management and ``nox`` for automated testing. Another recommendation is rigorous typing. The repository framework widely follows the blog series.


.. [*] Claudio, in turn, was inspired by the chess book "Die hypermoderne Schachpartie" (1925) by Savielly Tartakower.
