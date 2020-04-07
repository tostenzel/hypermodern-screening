Uncertainty Quantification
==========================

According to [Smith.2014]_, Model-based forecasting includes two main steps: the first step is calibration. In this step, the input parameters of the model are estimated. The second step is the prediction. The prediction contains the model evaluation at the estimated parameters. This allows us to make statements about potential scenarios. These statements are made in a probabilistic way. Thereby, the uncertainty of these statements is emphasised.

There are four sources of uncertainty in modern forecasting that are based on complex computational models ([Smith.2014]_). The first source, the model uncertainty, is the uncertainty about whether the mathematical model represents the reality sufficiently. The second source, the input uncertainty, is the uncertainty about the size of the input parameters of the model. The third one, the numerical uncertainty, comes from potential errors and uncertainties introduced by the translation of a mathematical to a computational model. The last source of uncertainty, the measurement uncertainty, is the accuracy of the experimental data that is used to approximate and calibrate the model.

We deal with the second source of uncertainty, the input uncertainty. In my view, this is the source for which UQ offers the most and also the strongest instruments. This might result from the fact that the estimation step produces standard errors as basic measures for the variation or uncertainty in the input parameter estimates. These can then be used to compute a variety of measures for the impact of the input uncertainty on the variation in the model output.

The following explains the basic notation. It is essential to define the quantity that one wants to predict with a model. This quantity is called output, or quantity of interest, and is denoted by :math:`Y`. For instance, the QoI can be the impact of a 500 USD tuition subsidy for higher education on average schooling years. The uncertain model parameters :math:`X_1, X_2, ..., X_k` are denoted by vector :math:`\pmb{X}`. The function that computes QoI Y by evaluating a  model and, if necessary, post-processing the model output is denoted by :math:`f(X_1, X_2, ..., X_k)`. Thus,

.. math::
    \begin{align}
    Y = f(\pmb{X}).
    \end{align}

From this point forward, I also refer to :math:`f` as the model. Large-scale UQ applications draw from various fields such as probability, statistics, analysis, and numerical mathematics. These disciplines are used in a combined effort for parameter estimation, surrogate model construction, parameter selection, uncertainty analysis, local sensitivity analysis (LSA), and GSA, amongst others. Drawing from [Smith.2014]_) I briefly sketch the first four components. The last two components, local and especially global sensitivity analysis, are discussed more extensively after that.

Parameter estimation covers the calibration step. There is a large number of estimation techniques for various types of models.

If the run time of a model is too long to compute the desired UQ measures, surrogate models are constructed to substitute the original model :math:`f` ([McBrider.2019]_). These surrogate models are functions of the model input parameters which are faster to evaluate. The functions are also called interpolants because they are computed from a random sample of input vectors, drawn from the input distribution and evaluated by the model. Typically, a surrogate model is computed by minimising a distance measure between a predetermined type of function and the model evaluations at the sample points. Therefore, the surrogate model interpolates this sample. Some specifications, like orthogonal polynomials, have properties which can simplify the computation of UQ measures tremendously ([Xiu.2010]_).

Another way to reduce the computation time, not directly of the model but of UQ measures, is to reduce the number of uncertain input parameters as part of a parameter selection. The decision which parameters to fix is made based on sensitivity measures. This is called **screening** or *factor fixing* ([Saltelli.2008]_). This point will be taken up again in the next section.

Uncertainty analysis is the core of the prediction step. It comprises two parts. The first part is the construction of the QoI's probability distribution by propagating the input uncertainty through the model. For instance, this can be achieved by evaluating a sample of random input parameters (as also required for the construction of a surrogate model). The second part is the computation of descriptive statistics like the probabilities for a set of specific events in the QoI range using this distribution. Both steps are conceptually simple. The construction of the probability distribution is also important for designing subsequent steps like a sensitivity analysis. For example, if the distribution is unimodal and symmetric, variance-based UQ measures are meaningful. If the distribution has a less tractable, then density-based measures are better suited ([Plischke.2013]_)).

The next section is a short and general presentation of sensitivity analysis.
