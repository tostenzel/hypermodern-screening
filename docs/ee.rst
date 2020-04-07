Elementary Effects
==================

Click at the following `nbviewer` or `mybinder` badges to view the tutorial notebook that accompanies this section.

.. image:: https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464
 :target: https://nbviewer.jupyter.org/github/tostenzel/hypermodern-screening/blob/master/docs/notebooks/elementary_effects.ipynb

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/tostenzel/hypermodern-screening/master?filepath=docs%2Fnotebooks%2Felementary_effects.ipynb

This section describes the approach to extend the EE-based measures to input parameters that are correlated. It largely follows [Ge.2017]_. Their main achievement is to outline a transformation of samples in radial and trajectory design that incorporates the correlation between the input parameters. This implies, that the trajectory and radial samples cannot be written as before. The reason is that the correlations of parameter :math:`X_i`, to which step :math:`\Delta^i` is added, imply that all other parameters :math:`\pmb{X_{\sim i}}` in the same row with non-zero correlation with :math:`X_i` are changed as well. Therefore, the rows cannot be denoted and compared as easily by :math:`a`'s and :math:`b`'s as in the last section. Transforming these matrices allows to re-define the EE-based measures accordingly, such that they sustain the main properties of the ordinary measures for uncorrelated parameters. The property is being a function of the mean derivative. The section covers the approach in a simplified form, focussing on normally distributed input parameters. Yet, [Ge.2017]_ do not fully develop these measures. I explain how their measures can lead to arbitrary rankings for correlated input parameters. I first cover the inverse transform sampling method that incorporates correlations between random draws from the parameter distributions. Second, I describe the Elementary Effects that I redesigned based on my analysis of [Ge.2017]_ and the drawbacks therein. Lastly, I comment on these drawbacks in more detail.

**************************
Inverse transform sampling
**************************

The section deals with developing a recipe for transforming draws :math:`\pmb{u} = \{u_1, u_2, ..., u_k\}` from :math:`[0,1]` for an input parameter vector to draws :math:`\pmb{x} = \{x_1, x_2, ..., x_k\}` from an arbitrary joint normal distribution. We will do this in three steps.

For this purpose, let :math:`\pmb{\Sigma}` be a non-singular variance-covariance matrix and let :math:`\pmb{\mu}` be the mean vector. The :math:`k`-variate normal distribution is denoted by :math:`\mathcal{N}_k(\pmb{\mu}, \pmb{\Sigma})`.

Creating potentially correlated draws :math:`\pmb{x}` from :math:`\mathcal{N}_k(\pmb{\mu}, \pmb{\Sigma})` is simple. Following [Gentle.2006]_, this can be achieved the following way: draw a :math:`k`-dimensional row vector of independent and identically distributed (i.i.d.) standard normal deviates from the univariate :math:`N(0,1)` distribution, such that  :math:`\pmb{z} = \{z_1, z_2, ..., z_k\}`, and compute the Cholesky decomposition of :math:`\Sigma`, such that :math:`\pmb{\Sigma} = \pmb{T^T T}`. The lower triangular matrix is denoted by :math:`\pmb{T^T}`. Then apply the operation in the equation below to obtain the correlated deviates :math:`\pmb{x}` from :math:`\mathcal{N}_k(\pmb{\mu}, \pmb{\Sigma})`. Then,

.. math::
    \pmb{x} = \pmb{\mu} + \pmb{T^T z^T}.

The next step is to understand that we can split the operation in thje above equation into two subsequent operations. The separated first part allows to potentially map correlated standard normal deviates to other distributions than the normal one. For this, let :math:`\pmb{\sigma}` be the vector of standard deviations and let :math:`\pmb{R_k}` be the correlation matrix of :math:`\pmb{x}`.

The first operation is to transform the standard deviates :math:`\pmb{z}` to correlated standard deviates :math:`\pmb{z_c}` by using :math:`\pmb{z_c}=\pmb{Q^T z^T}`. In this equation, :math:`\pmb{Q^T}` is the lower matrix from the Cholesky decomposition :math:`\pmb{R_k}=\pmb{Q^T Q}`. This is equivalent to the above approach in [Gentle.2006]_ for the specific case of the multivariate standard normal distribution :math:`\mathcal{N}_k(0, R_k)`. This is true because for multivariate standard normal deviates, the correlation matrix is equal to the covariance matrix.

The second operation is to scale the correlated standard normal deviates: :math:`\pmb{z}=\pmb{z_c(i)}\pmb{\sigma}\pmb{(i)} + \pmb{\mu}`., where the :math:`i` s indicate an element-wise multiplication. This equation is specific to normally distributed parameters.

The last step to construct the final approach is to recall the inverse transform sampling method. Therewith, we can transform the input parameter draws :math:`\pmb{u}` to uncorrelated standard normal draws :math:`\pmb{z}`. Then we will continue with the two operations in the above paragraph. The transformation from :math:`\pmb{u}` to :math:`\pmb{z}` is denoted by :math:`F^{-1}(\Phi^c)`, where the :math:`c` in  :math:`\Phi^c` stands for the introduced correlations. This transformation is summarized by the following three steps:

.. math::

	\text{Step 1: }\pmb{z} = \pmb{\Phi({u})}

	\text{Step 2: }\pmb{z_c} = \pmb{Q^T z^T}

	\text{Step 3: }\pmb{x} = \pmb{\mu} + \pmb{z_c(i)}\pmb{\sigma(i)}

I denote these three steps by :math:`F^{-1}(\Phi^c) = \mathcal{T}_2`.



To map :math:`u` to different sample spaces, Step 3 can be substituted. For instance, this could be achieved by applying :math:`\Phi^{-1,u}` and the inverse CDF of the desired distribution to :math:`z_c`. [The procedure described by the three steps above is equivalent to an inverse Rosenblatt transformation and a linear inverse Nataf transformation for parameters in normal sample space and connects to Gaussian copulas. For the first two transformations, see [Lemaire.2013]_, p. 78 - 113. These concepts can be used to transform deviates in [0,1] to the sample space of arbitrary distributions by using the properties sketched above under different conditions.]

The one most important point to understand is that the transformation comprised by the three steps is not unique for correlated input parameters. Rather, the transformation changes with the order of parameters in vector :math:`\pmb{u}`. This can be seen from the lower triangular matrix :math:`\pmb{Q^T}`. To prepare the next equation, let :math:`\pmb{R_k} = (\rho_{ij})_{ij=1}^k` and sub-matrix :math:`\pmb{R_h} = (\rho_{ij})_{ij=1}^h`, :math:`h \leq k`. Also let :math:`\pmb{\rho_i^{*,j}} = (\rho_{1,j}, \rho_{2,j}, ..., \rho_{i-1,j})` for :math:`j \geq i` with the following abbreviation :math:`\pmb{\rho_i}:=\pmb{\rho_i^{*,i}}`. Following [Madar.2015]_, the lower matrix can be written as

.. math::
	\pmb{Q^T} =
	\begin{pmatrix}
		\\ 1 & 0 & 0 & ... & 0
		\\\rho_{1,2} & \sqrt{1-\rho_{1,2}^2} & 0 & ... & 0
		\\ \rho_{1,3} & \frac{\rho_{2,3}-\rho_{1,2}\rho_{1,3}}{\sqrt{1-\rho_{1,2}^2}} & \sqrt{1-\pmb{\rho_{3}}\pmb{R^{-1}_2}\pmb{\rho_{3}^T}} & ... & 0
		\\\vdots & \vdots & \vdots & 	\ddots & \vdots
		\\ \rho_{1,k} &\frac{\rho_{2,k}-\rho_{1,2}\rho_{1,k}}{\sqrt{1-\rho_{1,2}^2}} & \frac{\pmb{\rho_{3,k}}-\pmb{\rho_{3}^{*,k}}\pmb{R^{-1}_2}\pmb{\rho_{3}^T}}{\sqrt{1-\pmb{\rho_{3}}\pmb{R^{-1}_2}\pmb{\rho_{3}^T}}}  &
		... & \sqrt{1-\pmb{\rho_{k}}\pmb{R^{-1}_2}\pmb{\rho_{k}^T}}
	\end{pmatrix}.

This equation, together with Step 2, implies that the order of the uncorrelated standard normal deviates :math:`\pmb{z}` constitutes a hierarchy amongst the correlated deviates :math:`\pmb{z_c}` in the following manner: the first parameter is not subject to any correlations, the second parameter is subject to the correlation with the first parameter, the third parameter is subject to the correlations with the parameters before, etc. Therefore, if parameters are correlated, typically :math:`\pmb{Q^T z^T} \neq \pmb{Q^T (z')^T}` and :math:`F^{-1}(\Phi)(\pmb{u}) \neq F^{-1}(\Phi)(\pmb{u'})`, where :math:`\pmb{z'}` and :math:`\pmb{u'}` denote :math:`\pmb{z}` and :math:`\pmb{u}` in different orders. The next section shows how the three sampling steps play in the design of the Elementary Effects for correlated parameters.

******************
Elementary Effects
******************

I redesign the measures in [Ge.2017]_ by scaling the :math:`\Delta` in the denominator according to the nominator. I refer to the redesigned measures as the correlated and the uncorrelated Elementary Effects, :math:`d_i^{c}` and :math:`d_i^{u}`. The first measure includes the respective parameters effect on the other parameters and the second measure excludes it. The idea is, that both parameters have to be *very small* for one parameter to be fixed as constant or non-random. It is still a open reasearch question what *very small* should be. The measures are given below for arbitrary input distributions and for samples in trajectory and radial design.

.. math::

	d_i^{c,T} = \frac{f\big(\mathcal{T}(\pmb{T_{i+1,*}}; i-1)\big) - f\big(\mathcal{T}(\pmb{T_{i-1,*}}; i)\big)}{\cal{T} (b_i) - \cal{T}(a_i)}

	d_i^{u, T} = \frac{f\big(\mathcal{T}(\pmb{T_{i+1,*}}; i)\big) - f\big(\mathcal{T}(\pmb{T_{i,*}}; i)\big)}{\cal{T} (b_i) - \cal{T}(a_i)}

	d_i^{c, R} = \frac{f\big(\mathcal{T}(\pmb{R_{i+1,*}}; i-1)\big) - f\big(\mathcal{T}(\pmb{R_{1,*}}; i-1)\big)}{\cal{T} (b_i) - \cal{T}(a_i)}

	d_i^{u, R} = \frac{f\big(\mathcal{T}(\pmb{R_{i+1,*}}; i)\big) - f\big(\mathcal{T}(\pmb{R_{1,*}}; i)\big)}{\cal{T} (b_i) - \cal{T}(a_i)}.

In the above equations, :math:`\mathcal{T}(\cdot; i) :=\mathcal{T}_3\bigg(\mathcal{T}_2\big(\mathcal{T}_1(\cdot; i)\big); i\bigg)`. :math:`\mathcal{T}_1(\cdot; i)` orders the parameters, or row elements, to establish the right correlation hierarchy. :math:`\mathcal{T}_2`, or :math:`F^{-1}(\Phi^c)`, correlates the draws in :math:`[0,1]` and transforms them to the sample space. :math:`\mathcal{T}_3(\cdot; i)` reverses the element order back to the start, to be able to apply the subtraction in the numerator of the EEs row-by-row. Index :math:`i` in :math:`\mathcal{T}_1(\cdot; i)` and :math:`\mathcal{T}_3(\cdot; i)` stands for the number of initial row elements that are cut and moved to the back of the row in the same order. Applying :math:`\mathcal{T}(\pmb{T_{i+1,*}}; i-1)` and :math:`\mathcal{T}(\pmb{T_{i+1,*}}; i)` to all rows :math:`i` of trajectory :math:`\pmb{T}` gives the following two transformed trajectories:

.. math::

	\mathcal{T}_1(\pmb{T_{i+1,*}}; i-1)
	=
	\begin{pmatrix}
	a_k & a_1 & ... & ... &  a_{k-1} \\
	\pmb{b_1} & a_2 & ... & ... &  a_k \\
	\pmb{b_2} & a_3 & ... & ... &  \pmb{b_1} \\
	\vdots & \vdots & \vdots & 	\ddots &  \vdots\\
	\pmb{b_k} & \pmb{b_{1}} & ... & ... &  \pmb{b_{k-1}}
	\end{pmatrix}


	\mathcal{T}_1(\pmb{T_{i,*}}; i-1)=
	\begin{pmatrix}
	a_1 & a_2 & ... & ... &  a_k \\
	a_2 & ... & ... &  a_k & \pmb{b_1} \\
	a_3 & ... & ... &  \pmb{b_1} & \pmb{b_2} & \\
	\vdots & \vdots & \vdots & 	\ddots &  \vdots\\
	\pmb{b_1} & \pmb{b_{2}} & ... & ... &  \pmb{b_{k}}
	\end{pmatrix}


Two points can be seen from Equation above equations. First, :math:`\mathcal{T}_1(\pmb{T_{i+1,*}}; i-1)` and :math:`\mathcal{T}_1(\pmb{T_{i,*}}; i)`, i.e. the :math:`(i+1)`-th row in the first equation and the :math:`(i)`-th row in the second equation, only differ in the :math:`i`-th element. The difference is :math:`b_i - a_i`. Thus, these two rows can be used to compute the EEs like in the uncorrelated case in the qualitative GSA section. However, in this order, the parameters are in the wrong positions to be directly handed over to the function, as the :math:`i`-th parameter is always in front. The second point is that in :math:`\mathcal{T}_1(\pmb{T_{i+1,*}}; i-1)`, :math:`b_i` is in front of the :math:`i`-th row. This order prepares the establishing of the right correlation hierarchy by :math:`\mathcal{T}_2`, such that the :math:`\Delta` in :math:`a_i + \Delta` is included to transform all other elements representing :math:`X_{\sim i}`. Importantly, to perform :math:`\mathcal{T}_2`, mean vector :math:`\pmb{x}` and covariance matrix :math:`\pmb{\Sigma}` and its transformed representatives have always to be re-ordered according to each row.
Then, :math:`\mathcal{T}_3` restores the original row order and :math:`d_i^{full}` can comfortably be computed by comparing function evaluations of row :math:`i+1` in :math:`\mathcal{T}(\pmb{T_{i+1,*}}; i-1)` with function evaluations of row :math:`i` in :math:`\mathcal{T}(\pmb{T_{i,*}}; i-1)`. Now, the two transformed trajectories only differ in the :math:`i`-th element in each row :math:`i`.


Assuming samples in trajectory design, one can also write the denominator equivalently but more explicitly for all four EEs as :math:`F^{-1}\big({Q^T}_{k,*k-1}(j)R_{i+1,*k-1}^T(j) + {Q^T}_{k,k} \Phi^u(b_i)\big) - F^{-1}\big({Q^T}_{k,*k-1}(j)R_{i,*k-1}^T(j)+{Q^T}_{k,k} \Phi^u(a_i)\big)`. Not accounting for :math:`Q_t` like [Ge.2017]_ leads to arbitrarily decreased independent Elementary Effects for input parameters with higher correlations.

The transformation for the samples in radial design are equivalent except that the subtrahend samples do only consist of reordered first rows.

For :math:`X_1, ..., X_k \sim \mathcal{N}_k(\pmb{\mu}, \pmb{\Sigma})`, the denominator of :math:`d_i^{u,*}` simplifies drastically to

.. math::

	\begin{split}
	\big(\mu_i + \sigma_i\big({Q^T}_{k,*k-1}(j)T_{i+1,*k-1}^T(j) + {Q^T}_{k,k} \Phi^u(b_i)\big) \\-  \big(\mu_i + \sigma_i\big({Q^T}_{k,*k-1}(j)T_{i+1,*k-1}^T(j) + {Q^T}_{k,k} \Phi^u(a_i)\big)\\= \sigma_i{Q^T}_{k,k}\big(\Phi^u(b_i)-\Phi^u(a_i)\big).
	\end{split}

**In this package, the implementation restricts itself to uniform and normally distributed input parameters.** It resembles the expression in the last equation.

***********************
Drawbacks in [Ge.2017]_
***********************

For the following explanation, I refer to a normal sample space. The drawback in the EE definitions by [Ge.2017]_ is that :math:`\Delta` is transformed multiple times in the numerator but not in the denominator. Therefore, these measures are not Elementary Effects in the sense of a derivative. To understand the intuition, it is easier to view :math:`\Delta` as :math:`b-a`. The transformation in the numerator is performed by applying :math:`F^{-1}(\Phi^c)` to :math:`u_i^j = a_i^j + \Delta^{(i,j)}`. The implications of this approach is twofold. The first implication is that function :math:`f` is evaluated at arguments that are non-linearily transformed by Step 1 and Step 3. Then, the difference in the numerator is divided by the  difference between the changed input parameter and the base input parameter -- in unit space. Therefore, numerator and denominator refer to different sample spaces. This makes the results hard to interpret. It also increases the influence of more extreme draws in :math:`[0,1]` because :math:`\Phi^{-1}` is very sensitive to these. Therefore, it will take a larger sample for the aggregate EE measures in [Ge.2017]_ to converge. Additionally, these problems are amplified if there are large differences between the inputs' standard deviation through the subsequent multiplicative scaling. The large sensitivity to more extreme draws implies also a higher sensitivity to larger differences in :math:`\Delta=b-a`. Therefore, the results will differ in their level depending on the sampling scheme. The largest drawback, however, is that :math:`b_i-a_i` in the denominator of :math:`d_i^{ind}` does not account for the transformation of the :math:`b_i-a_i` in the nominator by the establishing of correlations in Step 2. This transformation decreases :math:`b_i-a_i` proportional to the degree of correlations of the respective input parameter as can be seen by the last row of the lower Cholesky matrix. Hence, :math:`d_i^{ind}` is inflated by the the input parameters' correlations even tough this measure is introduced as an independent effect. It actually is a dependent effect as well. Because the full EE has to be interpreted with regards to the independent EE, this problem spills over to :math:`d_i^{full}`. For these reasons, I argue that these measures can not be used for screening.

The next section explains why it is important to sigma-normalize elementary effects.
