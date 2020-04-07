Quantitative GSA
================

Quantitative GSA aims to determine the precise effect size of each input parameter and its variation on the output variation. The most common measures in quantitative GSA are the Sobol' sensitivity indices. The next equation gives the general expression for the first order index. Let :math:`\text{Var}_{X_i} (Y|X_i)` denote the variance of the model output :math:`Y` conditional on input parameter :math:`X_i`. Then,

.. math::
    S_i = \frac{\text{Var}_{X_i}(Y|X_i)}{\text{Var}(Y)}.

The equation becomes clearer with the following equivalent expression in the next one.
For this purpose, let :math:`\sim i` denote the set of indices except :math:`i`. The expectation of :math:`Y` for one specific value of :math:`X_i` equals the average of the model evaluations from a sample, :math:`\pmb{\chi_{\sim i}}`,  of :math:`\pmb{X_{\sim i}}` and a given value
:math:`X_i = x_i^*`. Then, we use :math:`E[f(X_i = x_i^*,\pmb{\chi_{\sim i}} )] = E_{\pmb{X_{\sim i}}} [Y|X_i ]` to write the first-order Sobol' index as the variance of :math:`E_{\pmb{X_{\sim i}}} [Y|X_i ]` over all :math:`x_i^*` as

.. math::
    S_i = \frac{\text{Var}_{X_i}\big( E_{\pmb{X_{\sim i}}} [Y|X_i ]\big)}{\text{Var}(Y)}.


The first-order index does not include the additional variance in :math:`Y` that may occur from the interaction of :math:`\pmb{X_{\sim i}}` with :math:`X_i`. This additional variance is included in the total-order Sobol' index given by the next equation. It is the same expression as above except that the positions for :math:`X_i` and :math:`\pmb{X_{\sim i}}` are interchanged. Conditioning on :math:`\pmb{X_{\sim i}}` accounts for the inclusion of the interaction effects of :math:`X_i`.


.. math::
    S_{i}^T = \frac{\text{Var}_{\pmb{X_{\sim i}}}\big( E_{X_{\sim i}}[Y|\pmb{X_{\sim i}}] \big)}{\text{Var}(Y)}

Computing these measures requires many function evaluations, even if an estimator is used as a shortcut ([Saltelli.2004]_). The more time-intense one function evaluation is, the more utility does factor fixing based on qualitative measures provide.
