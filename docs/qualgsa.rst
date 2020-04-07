Qualitative GSA
===============

Qualitative GSA deals with the computation of measures that can rank random input parameters in terms of their impact on the function output and the variability thereof. This is done to a degree of accuracy that allows distinguishing between influential and non-influential parameters. If the measures for some input parameters are negligibly small, these parameters can be fixed so that the number of random input parameters decreases for a subsequent quantitative GSA. This section explains the qualitative measures and the trade-off between computational costs and accuracy.

The most commonly used measures in qualitative GSA is the mean EE, :math:`\mu`, the mean absolute EEs, :math:`\mu^*`, and the standard deviation of the EEs, :math:`\sigma`. The EE of :math:`X_i` is given by one individual function derivative with respect to :math:`X_i`. The "change in", or the "step of" the input parameter, denoted by :math:`\Delta`. The only restriction is that :math:`X_i + \Delta` is in the sample space of :math:`X_i`. The Elementary Effect, or derivative, is denoted by

.. math::
    d_i^{(j)} =  \frac{f(\pmb{X_{\sim i}^{(j)}}, X_i^{(j)} + \Delta^{(i,j)})- f(\pmb{X})}{\Delta^{(i,j)}},

where :math:`j` is an index for the number of :math:`r` observations of :math:`X_i`.
Note, that the EE, :math:`d_i^{(j)}`, is equal to the aforementioned local measure, the system derivate :math:`S_i = \frac{\partial Y}{\partial X_i}`, except that the value :math:`\Delta` has not to be infinitesimally small. To offset the third drawback of :math:`d_i` and :math:`S_i`, that base vector :math:`X_i` does not represent the whole input space, one computes the mean EE, :math:`\mu_i`, based on a random sample of :math:`X_i` from the input space. The second drawback, that interaction effects are disregarded, is also offset because elements :math:`X_{\sim i}` are also resampled for each new :math:`X_i`. The mean EE is given by

.. math::
    \mu_i = \frac{1}{r} \sum_{j=1}^{r} d_i^{(j)}.

Thus, :math:`\mu_i` is the global version of :math:`d_i^{(j)}`. The standard deviation of the EEs writes :math:`\sigma_i = \sqrt{\frac{1}{r} \sum_{j=1}^{r} (d_i^{(j)} - \mu_i)^2}`. The mean absolute EE, :math:`\mu_i^*`, is used to prevent observations of opposite sign to cancel each other out:

.. math::
    \mu_i^* = \frac{1}{r} \sum_{j=1}^{r} \big| d_i^{(j)} \big|.

Step :math:`\Delta^{(i,j)}` may or may not vary depending on the sample design that is used to draw the input parameters.


One last variant is provided in [Smith.2014]_. That is, the scaling of :math:`\mu_{i}^* by \frac{\sigma_{X_i}}{\sigma_Y}`. This measure is called the sigma-normalized mean absolute EE:


.. math::
    \mu_{i,\sigma}^* = \mu_i^* \frac{\sigma_{X_i}}{\sigma_Y}.


This improvement is necessary for a consistent ranking of :math:`X_i`. Otherwise, the ranking would be distorted by differences in the level of the the input parameters. The reason is that the input space constrains :math:`\Delta`. If the input space is larger, the base value of :math:`X_i` can be changed by a larger :math:`\Delta`.


From the aforementioned set of drawbacks of the local derivate :math:`D_i = \frac{\partial Y}{\partial X_i}`, two drawbacks are remaining for the EE method. The first drawback is the missing direct link to the variation in :math:`Var(Y)`. The second drawback is that the choice of :math:`\Delta` is somewhat arbitrary if the derivative is not analytic. To this date, the literature has not developed convincing solutions for these issues.

In an attempt to establish a closer link between EE-based measures and Sobol' indices, [Kucherenko.2009]_ made two conclusions: the first conclusion is that there is an upper bound for the total index, :math:`S_i^T`, such that

.. math::
    S_i^T \leq \frac{\frac{1}{r} \sum_{j=1}^{r} {d_i^2}^{(j)}}{\pi^2 \sigma_Y}.

This expression makes use of the squared EE. In light of this characteristic, the use of :math:`\sigma_i` as a measure that aims to include the variation of :math:`d_i^{j}` appears less relevant. Nevertheless, this rescaling makes the interpretation more difficult. The second conclusion is that the Elementary Effects method can lead to false selections for non-monotonic functions. This is also true if functions are non-linear. The reason is linked to the aforementioned second drawback, the arbitrary choice of step :math:`\Delta`. More precisely, depending on the sampling scheme, :math:`\Delta` might be random instead of arbitrary and constant. In both cases, :math:`\Delta` can be too large to approximate a derivative. If, for example, the function is highly non-linear of varying degree with respect to the input parameters :math:`\pmb{X}`, :math:`\Delta > \epsilon` can easily distort the results. Especially if the characteristic length of function variation is much smaller than :math:`\Delta`.
