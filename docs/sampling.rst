Sampling Schemes
================

Click at the following `nbviewer` or `mybinder` badges to view the tutorial notebook that accompanies this section.

.. image:: https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464
 :target: https://nbviewer.jupyter.org/github/tostenzel/hypermodern-screening/blob/master/docs/notebooks/sampling.ipynb

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/tostenzel/hypermodern-screening/master?filepath=docs%2Fnotebooks%2Fsampling.ipynb


The two presented sampling schemes are the trajectory and the radial design. Although the schemes are tailored to the computation of EEs, positional differences between them cause differences in their post-processing.

According to several experiments with common test functions by [Campolongo.2011]_, the best design is the radial design ([Saltelli.2002]_) and the most commonly used is the trajectory design ([Morris.1991]_).
Both designs are comprised by a :math:`(k + 1) \times k`-dimensional matrix. The elements are generated in :math:`[0,1]`. Afterwards, they can potentially be transformed to the distributions of choice. The columns represent the different input parameters and each row is a complete input parameter vector. To compute the aggregate qualitative measures, a set of multiple matrices, or sample subsets, of input parameters has to be generated.

A matrix in radial design is generated the following way: draw a vector of length :math:`2k` from a quasi-random sequence. The first row, or parameter vector, is the first half of the sequence. Then, copy the first row to the remaining :math:`k` rows. For each row :math:`k'` of the remaining 2, ..., :math:`k+1` rows, replace the :math:`k'`-th element by the :math:`k'`-th element of the second half of the vector. This generates a matrix of the following form:

.. math::

	\underset{(k+1)\times k}{\pmb{R}} =
	\begin{pmatrix}
	a_1 & a_2 & ... & a_k \\
	\pmb{b_1} & a_2 & ... & a_k \\
	a_1 & \pmb{b_2} & ... & a_k \\
	\vdots & \vdots & 	\ddots & \vdots\\
	a_1 & a_2 & ... & \pmb{b_k}
	\end{pmatrix}.

Note here, that each column consists only of the respective first row element, except in one row.
From this matrix, one EE can be obtained for each parameter :math:`X_i`. This is achieved by using the :math:`(i+1)`-th row as function argument for the minuend and the first row as the subtrahend in the EE formula in Equation (\ref{eq:EE}). Then, :math:`\Delta^{(i,j)} = b_i^{(j)} - a_i^{(j)}`. The asterisk is an index for all elements of a vector.

.. math::

    d_i =  \frac{Y(\pmb{a_{\sim i}}, b_i) - Y(\pmb{a})}{b_i - a_i} = \frac{Y(\pmb{R_{i+1,*}}) -  Y(\pmb{R_{1,*}})}{b_i - a_i}.


If the number of radial subsamples is high, the draws from the quasi-random sequence lead to a fast coverage of the input space (compared to a random sequence). However, a considerable share of steps will be large, the maximum is :math:`1-\epsilon` in a sample space of :math:`[0,1]`. This amplifies the aforementioned problem of EE-based measures with non-linear functions. The quasi-random sequence considered here is the Sobol' sequence. It is comparably successful in the dense coverage of the unit hypercube, but also conceptually more involved. Therefore, its presentation is beyond the scope of this work. As the first elements of each Sobol' sequence, the direction numbers, are equal I draw the sequence at once for all sets of radial matrices.

Next, I present the trajectory design. As we will see, it can lead to a relatively representative coverage for a very small number of subsamples but also to repetitions of similar draws.
I skip the equations that generate a trajectory and instead present the method verbally.
There are different forms of trajectories. I focus on the common version presented in [Morris.1991]_ that generates equiprobable elements. The first step is to decide the number :math:`p` of equidistant grid points in interval :math:`[0,1]`. Then, the first row of the trajectory is composed of the lower half value of these grid points. Now, fix :math:`\Delta = p/[2(p-1)]`. This function implies, that adding :math:`\Delta` to the lowest point in the lowest half results in the lowest point of the upper half of the grid points, and so on. It also implies that 0.5 is the lower bound of :math:`\Delta`. The rest of the rows is constructed, first, by copying the row one above and, second, by adding :math:`\Delta` to the :math:`i`-th element of the :math:`i+1`-th row. The created matrix scheme is depicted below.

.. math::

	\underset{(k+1)\times k}{\pmb{T}} =
	\begin{pmatrix}
	a_1 & a_2 & ... & a_k \\
	\pmb{b_1} & a_2 & ... & a_k \\
	\pmb{b_1} & \pmb{b_2} & ... & a_k \\
	\vdots & \vdots & 	\ddots & \vdots\\
	\pmb{b_1} & \pmb{b_2} & ... & \pmb{b_k}
	\end{pmatrix}


In contrary to the radial scheme, each :math:`b_i` is copied to the subsequent row. Therefore, the EEs have to be determined by comparing each row with the row above instead of with the first row.
Importantly, two random transformations are common. These are, first, randomly switching rows, and second, randomly interchanging the :math:`i`-th column with the :math:`(k-i)`-th column and then reversing the column. The first transformation is skipped as it does not add additional coverage and because we need the stairs-shaped design to facilitate later transformations which account for correlations between input parameters. The second transformation is adapted because it is important to also have negative steps and because it sustains the stairs shape. Yet, this implies that :math:`\Delta` is also parameter- and trajectory-specific. Let :math:`f` and :math:`h` be additional indices representing the input parameters. The derivative formula is adapted to the trajectory design as follows:

.. math::

    d_i =  \frac{Y(\pmb{b_{f \leq i}}, \pmb{a_{h>i}}) - Y(\pmb{b_{f<i}}, \pmb{a_{h \geq i}})}{b_i - a_i} = \frac{Y(\pmb{T_{i+1,*})} -  Y(\pmb{T_{i,*}})}{b_i - a_i}.

The trajectory design involves first, a fixed grid, and second and more importantly, a fixed step :math:`\Delta`, s.t. :math:`\{\Delta\} = \{\pm \Delta\}`. This implies less step variety and less space coverage vis-á-vis the radial design for a larger number of draws.

To improve the sample space coverage by the trajectory design, [Campolongo.2007]_ develop a post-selection approach based on distances. The approach creates enormous costs for more than a small number of trajectories. This problem is effectively mitigated by [Ge.2014]_. The following describes the main ideas of both contributions.

The objective of both works is to select :math:`k`trajectories from a set of :math:`N` matrices. [Campolongo.2007]_ assign a pair distance to each pair of trajectories in the start set. Thereafter, they identify each possible combination of :math:`k` from :math:`N` trajectories. Then, they compute an aggregate distance for each combination based on the single pair distances. Finally, the optimized trajectory set is the subset with the highest aggregate distance.

This is computationally very costly because each aggregate distance is a sum of a binomial number of pair distances. For example, :math:`\binom{30}{15} = 155117520`. To decrease the computation time, [Ge.2014]_ propose two improvements. First, in each iteration :math:`i`, they select only :math:`N(i)-1` matrices from a set containing :math:`N(i)` trajectories until the set size has decreased to :math:`k`. Second, they compute the pair distances in each iteration based on the aggregate distances and the pair distances from the first set. Due to numerical imprecisions, their improvement does not always result in the same set as obtained from [Campolongo.2007]. However, the sets are usually very similar in terms of the aggregate distance. This thesis only uses the first step in [Ge.2014] to post-select the trajectory set because the second step does not provide any gain. [This refers only to my implementation.]


So far, we have only considered draws in [0,1]. For uncorrelated input parameters from arbitrary distributions with well-defined cumulative distribution function (CDF), :math:`\Phi^{-1}`, one would simply evaluate each element (potentially including the addition of the step) by the inverse CDF, or quantile function, :math:`\Phi`, of the respective parameter. One intuition is, that :math:`\Phi^{-1}` maps the sample space to [0,1]. Hence :math:`\Phi` can be used to transform random draws in [0,1] to the sample space of the arbitrary distribution. This is a basic example of so-called inverse transform sampling ([Devroye.1986]_) which we recall in the next section.

The following section describes the computation of Elementary Effects for correlated input parameters from samples in trajectory and radial design.
