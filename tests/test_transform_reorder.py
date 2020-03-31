import numpy as np
import pytest

from numpy.testing import assert_array_equal

from hypermodern_screening.transform_reorder import ee_corr_reorder_sample
from hypermodern_screening.transform_reorder import reverse_ee_corr_reorder_sample
from hypermodern_screening.transform_reorder import ee_uncorr_reorder_sample
from hypermodern_screening.transform_reorder import reverse_ee_uncorr_reorder_sample
from hypermodern_screening.transform_reorder import reorder_mu
from hypermodern_screening.transform_reorder import reorder_cov
from hypermodern_screening.transform_reorder import reverse_reorder_mu
from hypermodern_screening.transform_reorder import reverse_reorder_cov


@pytest.fixture
def traj():
    """Fix sample for next two tests."""
    traj = np.array([[0, 0, 0], [1, 0, 0], [2, 3, 0], [4, 5, 6]])
    return traj


def test_ee_uncorr_reorder_sample(traj):
    """
    Unit tests for `ee_uncorr_reorder_sample` and
    `reverse_ee_uncorr_reorder_sample`.

    """
    assert_array_equal(
        ee_uncorr_reorder_sample(traj),
        np.array([[0, 0, 0], [0, 0, 1], [0, 2, 3], [4, 5, 6]]),
    )

    assert_array_equal(
        traj, reverse_ee_uncorr_reorder_sample(ee_uncorr_reorder_sample(traj))
    )

    assert_array_equal(
        ee_uncorr_reorder_sample(traj, row_plus_one=False),
        np.array([[0, 0, 0], [0, 1, 0], [2, 3, 0], [5, 6, 4]]),
    )

    assert_array_equal(
        traj,
        reverse_ee_uncorr_reorder_sample(
            ee_uncorr_reorder_sample(traj, row_plus_one=False), row_plus_one=False
        ),
    )


def test_ee_corr_reorder_sample(traj):
    """
    Unit tests for `ee_corr_reorder_sample` and
    `reverse_ee_corr_reorder_sample`.

    """
    assert_array_equal(
        ee_corr_reorder_sample(traj),
        np.array([[0, 0, 0], [1, 0, 0], [3, 0, 2], [6, 4, 5]]),
    )

    assert_array_equal(
        traj, reverse_ee_corr_reorder_sample(ee_corr_reorder_sample(traj))
    )


@pytest.fixture
def mu():
    """Fix expectation values for next test."""
    mu = np.arange(10)
    return mu


def test_reorder_mu(mu):
    """Unit tests for `reorder_mu` and `reverse_reorder_mu`."""
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    assert_array_equal(expected, reorder_mu(mu))

    expected = np.array([9, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert_array_equal(expected, reverse_reorder_mu(mu))

    assert_array_equal(mu, reverse_reorder_mu(reorder_mu(mu)))


@pytest.fixture
def cov():
    """Fix covariance matrix for next test."""
    cov = np.array(
        [
            [10, 2, 3, 4, 5],
            [2, 20, 6, 7, 8],
            [3, 6, 30, 9, 10],
            [4, 7, 9, 40, 11],
            [5, 8, 10, 11, 50],
        ]
    )
    return cov


def test_reorder_cov(cov):
    """Unit tests for `reorder_cov` and `reverse_reorder_mu`."""
    expected = np.array(
        [
            [20, 6, 7, 8, 2],
            [6, 30, 9, 10, 3],
            [7, 9, 40, 11, 4],
            [8, 10, 11, 50, 5],
            [2, 3, 4, 5, 10],
        ]
    )
    assert_array_equal(expected, reorder_cov(cov))
    expected = np.array(
        [
            [50, 5, 8, 10, 11],
            [5, 10, 2, 3, 4],
            [8, 2, 20, 6, 7],
            [10, 3, 6, 30, 9],
            [11, 4, 7, 9, 40],
        ]
    )
    assert_array_equal(expected, reverse_reorder_cov(cov))

    assert_array_equal(cov, reverse_reorder_cov(reorder_cov(cov)))
