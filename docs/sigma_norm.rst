Sigma Normalization
===================

The aim of this section is to show how important it can be to sigma-normalize the Elementary Effects or the derived statistics thereof. A code example is given by:

.. code-block:: bash

    # Compute sigma-normalized statistics of Elementary Effects.
    measures_sigma_norm = hms.compute_measures(ees_list, sd_qoi, sd_inputs, sigma_norm=True)

Let :math:`g(X_1, ..., X_k) = \sum_{i = 1}^{k} c_i X_i` be an arbitrary linear function. Let :math:`\rho_{i,j}` be the linear correlation between :math:`X_i` and :math:`X_j`. Then, for all :math:`i \in 1, ..., k`, I expect

.. math::
    d_i^{u,*} = c_i,\\
    d_i^{c,*} = \sum_{j = 1}^{k} \rho_{i,j} c_{j}.

These results correspond to the intuition provided by the example in [Saltelli.2008]_, p. 123.
Both equations state that, conceptually, the result does not depend on the sampling scheme.


Let us consider the case without any correlations between the inputs. Additionally, let :math:`c_i = \{3,2,1\}` and :math:`\sigma^2_{X_i}=\{1,4,9\}` for :math:`i \in \{1,2,3\}`. The following results are derived from [Saltelli.2008]_. Let us first compute the Sobol' indices. As :math:`g` does not include any interactions, :math:`S_i^T = S_i`. Additionally, we have :math:`\text{Var}(Y)=\sum_{i=1}^k c_i^2 \sigma_{X_i}^2` and :math:`\text{Var}_{\pmb{X_{\sim i}}}\big( E_{X_{\sim i}}[Y|\pmb{X_{\sim i}}] \big) = c_i^2 \sigma_{X_i}^2`. Table 1 compares three different sensitvity measures. These are the total Sobol' indices, :math:`S_i^T` (Measure I, the mean absolute EE, :math:`\gamma_i^*` (Measure II), and the *squared* sigma-normalized mean absolute EE, :math:`(\mu_i^* \frac{\sigma_{X_i}}{\sigma_Y})^2` (Measure III).


.. csv-table:: Table 1: Importance measures for parametric uncertainty
   :header: "Parameter", "Measure I", "Measure II", "Measure III"
   :widths: 10, 10, 10, 10

   "X_1", 9, 3, 9
   "X_2", 8, 2, 8
   "X_3", 9, 1, 9



In context of screening, :math:`S_i^T` is the objective measure that we would like to predict approximately. We observe that :math:`\gamma_i^*` ranks the parameters incorrectly. The reason is that :math:`\gamma_i^*` is only a measure of the influence of :math:`X_i` on :math:`Y` and not of the influence of the variation and the level of :math:`X_i` on the variation of :math:`Y`. We also see that :math:`(\mu_i^* \frac{\sigma_{X_i}}{\sigma_Y})^2` is an exact predictor for :math:`S_i^T` as it does not only generate the correct ranking but also the right effect size. Importantly, this result is specific to a linear function without any interactions and correlations. However, it underlines the point that :math:`\gamma_i^*` alone is not sufficient for screening. Following Ge.2017, one approach would be to additionally consider the EE variation, :math:`\sigma_i`. However, analysing two measures at once is difficult for models with a large number of input parameters. Table 1 indicates that :math:`(\mu_i^* \frac{\sigma_{X_i}}{\sigma_Y})^2` and also :math:`\mu_i^* \frac{\sigma_{X_i}}{\sigma_Y}` can be an appropriate alternative. The actual derivative version of this measure is also recommended by guidelines of the Intergovernmental Panel for Climate Change ([IPCC.1999]_).
