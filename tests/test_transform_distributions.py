import numpy as np
from numpy.testing import assert_allclose

from hypermodern_screening.transform_distributions import covariance_to_correlation
from hypermodern_screening.transform_distributions import transform_uniform_stnormal_uncorr
from hypermodern_screening.transform_distributions import transform_stnormal_normal_corr

from tests.resources.nataf_transformation import nataf_transformation
from tests.resources.distributions import distributions


def test_covariance_to_correlation():
    """Unit test for `covariance_to_correlation`."""
    cov = np.array([[10, 0.2, 0.5], [0.2, 40, 0], [0.5, 0, 50]])
    expected = np.array([[1, 0.01, 0.0223], [0.01, 1, 0], [0.0223, 0, 1]])
    corr = covariance_to_correlation(cov)

    assert_allclose(corr, expected, atol=0.0001)


def test_transform_stnormal_normal_corr():
    """
    Compares the implementation of the inverse Rosenblatt/inverse Nataf transformation
    for normally distributed deviates with the implementation [1] for several
    distributions by TUM Department of Civil, Geo and Environmental Engineering
    Technical University of Munich.

    References
    ----------
    [1] https://www.bgu.tum.de/en/era/software/eradist/.

    """
    # Expectation values.
    mu = np.array([10, 10, 10, 10, 10])

    # Covariance matrix.
    cov = np.array(
        [
            [10, 0, 0, 2, 0.5],
            [0, 20, 0.4, 0.15, 0],
            [0, 0.4, 30, 0.05, 0],
            [2, 0.15, 0.05, 40, 0],
            [0.5, 0, 0, 0, 50],
        ]
    )

    # Draws from U(0,1).
    row = np.array([0.1, 0.1, 0.2, 0.8, 0.5])
    # Transform draws to uncorrelated N(0,1).
    z = transform_uniform_stnormal_uncorr(row)

    # Create Nataf transformation from class for many distribution types.
    M = list()
    M.append(distributions("normal", "PAR", [mu[0], np.sqrt(cov[0, 0])]))
    M.append(distributions("normal", "PAR", [mu[1], np.sqrt(cov[1, 1])]))
    M.append(distributions("normal", "PAR", [mu[2], np.sqrt(cov[2, 2])]))
    M.append(distributions("normal", "PAR", [mu[3], np.sqrt(cov[3, 3])]))
    M.append(distributions("normal", "PAR", [mu[4], np.sqrt(cov[4, 4])]))

    # Covariance matrix
    corr = covariance_to_correlation(cov)

    T_Nataf = nataf_transformation(M, corr)

    x_lemaire09, _ = transform_stnormal_normal_corr(z, cov, mu)
    X = T_Nataf.U2X(z)

    assert_allclose(x_lemaire09, X.T, atol=1.0e-14)
