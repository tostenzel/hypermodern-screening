"""Tests for module `screening_measures.py`."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from hypermodern_screening.sampling_schemes import radial_sample, trajectory_sample
from hypermodern_screening.screening_measures import (
    compute_measures,
    screening_measures,
)


def test_compute_measures() -> None:
    """Tests the normalization option by `(sd_x / sd_y)` of `compute_measures`."""
    ee_i = np.array([1, 1] * 3).reshape(3, 2)

    sd_x_2 = np.array([2, 2, 2])

    means_unit, _, _ = compute_measures(ee_i)
    means_double, _, _ = compute_measures(ee_i, sd_x_2, sigma_norm=True)

    assert_array_equal(means_unit * 2, means_double)


def sobol_model(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    coeffs: np.ndarray,
    *args: float,
) -> float:
    """Test function with analytical solutions.

    Notes
    -----
    Strongly nonlinear, nonmonotonic, and nonzero interactions.
    Analytic results for Sobol Indices.

    See Also
    --------
    `test_screening_measures_trajectory_uncorrelated_g_function`

    """
    input_pars = np.array([a, b, c, d, e, f])

    def g_i(input_pars: np.ndarray, coeffs: np.ndarray) -> float:
        return (abs(4 * input_pars - 2) + coeffs) / (1 + coeffs)

    y = float(1)
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs[i])

    return y


def test_screening_measures_trajectory_uncorrelated_g_function() -> None:
    """
    Tests the screening measures for six uncorrelated parameters.

    Data and results taken from pages 123 - 127 in [1]. The data is
    four trajectories and the results are the Elementary Effects, the absolute
    Elementary Effects and the SD of the Elementary Effects for six paramters.

    Notes
    -----
    -Many intermediate results are given as well. `screening_measures_trajectory` is
    able to compute all of them precisely.
    -The function uses a lot of reorderings. The reason is that
    `screening_measures_trajectory` assumes that the first columns has the first step
    addition etc. This facilitates the necessary transformations to account for
    correlations. In this example the order of the paramters to which the step is added
    is different for each trajectory. To account for this discrepancy in trajectory
    format, the trajectories and `sobol_model` have to be changed accordingly.
    Additionally, the effects have to be recomputed for each trajectory because the
    reordered trajectories with columns in order of the step addition are still
    composed of columns that represent different paramters.

    References
    ----------
    [1] Saltelli, A., M. Ratto, T. Andres, F. Campolongo, J. Cariboni, D. Gatelli,
    M. Saisana, and S. Tarantola (2008). Global Sensitivity Analysis: The Primer.
    John Wiley & Sons.

    """
    # Covariance matrix
    cov = np.zeros(36).reshape(6, 6)
    np.fill_diagonal(cov, np.ones(5))

    # This is not the expectation for x \in U[0,1]. Yet, prevents transformation.
    mu = np.array([0, 0, 0, 0, 0, 0])

    # Data: Four trajectories.
    # The columns are randomly shuffled in contrary to what this program assumes
    traj_one = np.array(
        [
            [0, 2 / 3, 1, 0, 0, 1 / 3],
            [0, 2 / 3, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [2 / 3, 0, 1, 0, 0, 1],
            [2 / 3, 0, 1, 2 / 3, 0, 1],
            [2 / 3, 0, 1 / 3, 2 / 3, 0, 1],
            [2 / 3, 0, 1 / 3, 2 / 3, 2 / 3, 1],
        ]
    )
    traj_two = np.array(
        [
            [0, 1 / 3, 1 / 3, 1, 1, 2 / 3],
            [0, 1, 1 / 3, 1, 1, 2 / 3],
            [0, 1, 1, 1, 1, 2 / 3],
            [2 / 3, 1, 1, 1, 1, 2 / 3],
            [2 / 3, 1, 1, 1, 1, 0],
            [2 / 3, 1, 1, 1, 1 / 3, 0],
            [2 / 3, 1, 1, 1 / 3, 1 / 3, 0],
        ]
    )
    traj_three = np.array(
        [
            [1, 2 / 3, 0, 2 / 3, 1, 0],
            [1, 2 / 3, 0, 0, 1, 0],
            [1 / 3, 2 / 3, 0, 0, 1, 0],
            [1 / 3, 2 / 3, 0, 0, 1 / 3, 0],
            [1 / 3, 0, 0, 0, 1 / 3, 0],
            [1 / 3, 0, 2 / 3, 0, 1 / 3, 0],
            [1 / 3, 0, 2 / 3, 0, 1 / 3, 2 / 3],
        ]
    )
    traj_four = np.array(
        [
            [1, 1 / 3, 2 / 3, 1, 0, 1 / 3],
            [1, 1 / 3, 2 / 3, 1, 0, 1],
            [1, 1 / 3, 0, 1, 0, 1],
            [1, 1 / 3, 0, 1 / 3, 0, 1],
            [1, 1 / 3, 0, 1 / 3, 2 / 3, 1],
            [1, 1, 0, 1 / 3, 2 / 3, 1],
            [1 / 3, 1, 0, 1 / 3, 2 / 3, 1],
        ]
    )
    # The uncorrices show the order of columns to which the step is added.
    idx_one = [5, 1, 0, 3, 2, 4]
    idx_two = [1, 2, 0, 5, 4, 3]
    idx_three = [3, 0, 4, 1, 2, 5]
    idx_four = [5, 2, 3, 4, 1, 0]

    # Create stairs shape:
    # Transform trajectories so that the the step is first added to frist columns etc.
    traj_one = traj_one[:, idx_one]
    traj_two = traj_two[:, idx_two]
    traj_three = traj_three[:, idx_three]
    traj_four = traj_four[:, idx_four]

    coeffs = np.array([78, 12, 0.5, 2, 97, 33])

    # Define wrappers around `sobol_model` to account for different coeffient order
    # due to the column shuffling. Argument order changes.
    def wrapper_one(
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
        coeffs: np.ndarray = coeffs[idx_one],
    ) -> float:
        return sobol_model(f, b, a, d, c, e, coeffs[idx_one])

    def wrapper_two(
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
        coeffs: np.ndarray = coeffs[idx_two],
    ) -> float:
        return sobol_model(b, c, a, f, e, d, coeffs[idx_two])

    def wrapper_three(
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
        coeffs: np.ndarray = coeffs[idx_three],
    ) -> float:
        return sobol_model(d, a, e, b, c, f, coeffs[idx_three])

    def wrapper_four(
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
        coeffs: np.ndarray = coeffs[idx_four],
    ) -> float:
        return sobol_model(f, c, d, e, b, a, coeffs[idx_four])

    # Compute step sizes because rows are also randomly shuffeled.
    # The uncorrices account for the column order for stairs.
    positive_steps = np.array([2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3])
    steps_one = positive_steps * np.array([1, -1, -1, 1, 1, 1])[idx_one]
    steps_two = positive_steps * np.array([1, 1, 1, -1, -1, -1])[idx_two]
    steps_three = positive_steps * np.array([-1, -1, 1, -1, -1, +1])[idx_three]
    steps_four = positive_steps * np.array([-1, +1, -1, -1, 1, 1])[idx_four]

    # Compute the uncorrependent Elementary Effects.
    # Since there is no correlation, they equal their abolute versions.
    one_ee_uncorr, _ = screening_measures(
        wrapper_one, [traj_one], [steps_one], cov, mu, radial=False
    )

    two_ee_uncorr, _ = screening_measures(
        wrapper_two, [traj_two], [steps_two], cov, mu, radial=False
    )

    three_ee_uncorr, _ = screening_measures(
        wrapper_three, [traj_three], [steps_three], cov, mu, radial=False
    )

    four_ee_uncorr, _ = screening_measures(
        wrapper_four, [traj_four], [steps_four], cov, mu, radial=False
    )

    # `argsort` inverses the transformation that uncorruced the stairs shape
    # to the trajectories.
    ee_one = np.array(one_ee_uncorr[0]).reshape(6, 1)[np.argsort(idx_one)]
    ee_two = np.array(two_ee_uncorr[0]).reshape(6, 1)[np.argsort(idx_two)]
    ee_three = np.array(three_ee_uncorr[0]).reshape(6, 1)[np.argsort(idx_three)]
    ee_four = np.array(four_ee_uncorr[0]).reshape(6, 1)[np.argsort(idx_four)]

    ee_i = np.concatenate((ee_one, ee_two, ee_three, ee_four), axis=1)

    # Compute summary measures "by hand" because `screening_measures_trajectory`
    # takes only a list of one trajectory because the argument order is different.
    ee = np.mean(ee_i, axis=1).reshape(6, 1)
    abs_ee = np.mean(abs(ee_i), axis=1).reshape(6, 1)
    # `np.var` does not work because it scales by 1/n instead of 1/(n - 1).
    sd_ee = np.sqrt((1 / (4 - 1)) * (np.sum((ee_i - ee) ** 2, axis=1).reshape(6, 1)))

    expected_ee = np.array([-0.006, -0.078, -0.130, -0.004, 0.012, -0.004]).reshape(
        6, 1
    )
    expected_abs_ee = np.array([0.056, 0.277, 1.760, 1.185, 0.034, 0.099]).reshape(6, 1)
    expected_sd_ee = np.array([0.064, 0.321, 2.049, 1.370, 0.041, 0.122]).reshape(6, 1)

    assert_array_equal(np.round(ee, 3), expected_ee, 3)
    assert_allclose(np.round(abs_ee, 3), expected_abs_ee, 3, atol=0.01)
    assert_array_equal(np.round(sd_ee, 3), expected_sd_ee, 3)


def lin_portfolio(
    q1: float, q2: float, c1: float = 2, c2: float = 1, *args: float
) -> float:
    """Linear function with analytic EE solution for the next test."""
    return c1 * q1 + c2 * q2


def test_screening_measures_trajectory_uncorrelated_linear_function() -> None:
    """
    Test for a linear function with two paramters.

    Non-unit variance and EEs are coefficients. Results data taken from [1], page 335.

    Notes
    -----
    This test contains intuition for reasable results (including correlations) for
    the first two testcases in [2] that also use a linear function. The corresponding
    EE should be the coefficients plus the correlation times the coefficients of the
    correlated parameters.

    References
    ----------
    [1] Smith, R. C. (2014). Uncertainty Quantification: Theory, Implementation, and
    Applications. Philadelphia: SIAM-Society for Industrial and Applied Mathematics.
    [2] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global
    sensitivityanalysis of models with dependent inputs. Reliability Engineering &
    System Safety 100 (162), 28–39.

    """
    cov = np.array([[1, 0], [0, 9]])

    # mu does not matter because the function is linear. You subtract what you add.
    mu = np.array([0, 0])

    numeric_zero = 0.01
    seed = 2020
    n_levels = 10
    n_inputs = 2
    n_traj_sample = 10_000

    traj_list, step_list = trajectory_sample(
        n_traj_sample, n_inputs, n_levels, seed, True, numeric_zero
    )

    measures, _ = screening_measures(
        lin_portfolio, traj_list, step_list, cov, mu, radial=False
    )

    exp_ee = np.array([2, 1]).reshape(n_inputs, 1)
    exp_sd = np.array([0, 0]).reshape(n_inputs, 1)

    assert_array_equal(exp_ee, measures[0])
    assert_array_equal(exp_ee, measures[1])
    assert_array_equal(exp_ee, measures[2])
    assert_array_equal(exp_ee, measures[3])
    assert_allclose(exp_sd, measures[4], atol=1.0e-15)
    assert_allclose(exp_sd, measures[5], atol=1.0e-15)


def linear_function(a: float, b: float, c: float, *args: float) -> float:
    """Additive function for Test Case 1 and 2 in [1].

    References
    ----------
    [1] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global
    sensitivityanalysis of models with dependent inputs. Reliability Engineering &
    System Safety 100 (162), 28–39.

    """
    return a + b + c


def test_linear_model_equality_radial_trajectory() -> None:
    """
    Measures for linear model computed by radial and traj. design are equal.

    Tests whether `screening_measures` yields the same results for samples in radial
    and in trajectory design. This yields confidence in the radial option, as the
    trajectory option is already tested multiple times.

    Notes
    -----
    As the model is linear, both uncorrelated EEs should be equals to the coefficients
    and both correlated EEs should be equals to the sum of coefficients times the
    correlation betwen parameters.

    """
    mu = np.array([0, 0, 0])

    cov = np.array([[1.0, 0.9, 0.4], [0.9, 1.0, 0.0], [0.4, 0.0, 1.0]])
    numeric_zero = 0.01
    seed = 2020
    n_levels = 10
    n_inputs = 3
    n_sample = 100

    # Generate trajectories and steps. Then computes measures.
    traj_list, traj_step_list = trajectory_sample(
        n_sample, n_inputs, n_levels, seed, False, numeric_zero
    )
    measures_list_traj, _ = screening_measures(
        linear_function, traj_list, traj_step_list, cov, mu, radial=False
    )

    # Generate radial samples and steps. Then computes measures.
    rad_list, rad_step_list = radial_sample(n_sample, n_inputs, True)
    measures_list_rad, _ = screening_measures(
        linear_function, rad_list, rad_step_list, cov, mu, radial=True
    )

    assert_allclose(measures_list_traj, measures_list_rad, atol=1.0e-13)
