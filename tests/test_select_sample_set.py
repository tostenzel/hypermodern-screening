"""
Tests select_sample_sets.py.
References
----------
[1] Campolongo, F., J. Cariboni, and A. Saltelli (2007). An effective screening design for
sensitivity analysis of large models. Environmental modelling & software 22 (10), 1509–
1518.
[2] Ge, Q. and M. Menendez (2014). An efficient sensitivity analysis approach for
computationally expensive microscopic traffic simulation models. International Journal of
Transportation 2 (2), 49–64.
"""
import sys

# Define parent folder as relative path.
sys.path.append("..")

import numpy as np
import pytest

from numpy.testing import assert_array_equal

from hypermodern_screening.sampling_schemes import morris_trajectory
from hypermodern_screening.select_sample_set import compute_pair_distance
from hypermodern_screening.select_sample_set import distance_matrix
from hypermodern_screening.select_sample_set import combi_wrapper
from hypermodern_screening.select_sample_set import select_trajectories
from hypermodern_screening.select_sample_set import campolongo_2007
from hypermodern_screening.select_sample_set import intermediate_ge_menendez_2014
from hypermodern_screening.select_sample_set import select_trajectories_wrapper_iteration
from hypermodern_screening.select_sample_set import total_distance
from hypermodern_screening.select_sample_set import final_ge_menendez_2014


def test_compute_pair_distance():
    """Unit test for function `compute_pair_distance`"""
    traj_0 = np.ones((3, 2))
    traj_1 = np.zeros((3, 2))
    assert 4 * np.sqrt(3) == compute_pair_distance(traj_0, traj_1)


def test_distance_matrix():
    """Unit test for function `distance_matrix`"""
    traj_list = [np.ones((3, 2)), np.zeros((3, 2))]
    expected = np.array([[0, 4 * np.sqrt(3)], [4 * np.sqrt(3), 0]])
    assert_array_equal(expected, distance_matrix(traj_list))


def test_combi_wrapper():
    """Unit test for function `combi_wrapper`"""
    expected_0 = [[0]]
    expected_1 = [[0], [1]]
    expected_2 = [[0, 1]]
    expected_3 = [[0, 1], [0, 2], [1, 2]]
    expected_4 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert expected_0 == combi_wrapper([0], 1)
    assert expected_1 == combi_wrapper([0, 1], 1)
    assert expected_2 == combi_wrapper([0, 1], 2)
    assert expected_3 == combi_wrapper([0, 1, 2], 2)
    assert expected_4 == combi_wrapper([0, 1, 2, 3], 2)


def test_select_trajectories_1():
    """
    Small unit test for `select_trajectories` by reducing the sample set by
    one sample. Tests only the combination that yields the maximal `total_distance`.
    """
    test_traj_dist_matrix = np.array(
        [[0, 1, 2, 4], [1, 0, 3, 100], [2, 3, 0, 200], [4, 100, 200, 0]]
    )

    test_indices, test_select = select_trajectories(test_traj_dist_matrix, 3)

    expected_dist_indices = [1, 2, 3]
    expected_fourth_row = [1, 2, 3, np.sqrt(3 ** 2 + 100 ** 2 + 200 ** 2)]

    assert test_indices == expected_dist_indices
    assert_array_equal(test_select[3, :], expected_fourth_row)


@pytest.fixture
def dist_matrix():
    """Fix dist_matrix for the next two tests."""
    dist_matrix = np.array([[0, 4, 5, 6], [4, 0, 7, 8], [5, 7, 0, 9], [6, 8, 9, 0]])
    return dist_matrix


def test_select_trajectories_2(dist_matrix):
    """The difference between sample and selection size is not large enough for high trust."""
    exp_max_dist_indices = [1, 2, 3]

    exp_combi_distance = np.array(
        [
            [0, 1, 2, np.sqrt(4 ** 2 + 5 ** 2 + 7 ** 2)],
            [0, 1, 3, np.sqrt(4 ** 2 + 6 ** 2 + 8 ** 2)],
            [0, 2, 3, np.sqrt(5 ** 2 + 6 ** 2 + 9 ** 2)],
            [1, 2, 3, np.sqrt(7 ** 2 + 8 ** 2 + 9 ** 2)],
        ]
    )

    max_dist_indices, combi_distance = select_trajectories(dist_matrix, 3)

    assert_array_equal(exp_max_dist_indices, max_dist_indices)
    assert_array_equal(exp_combi_distance, combi_distance)


def test_select_trajectories_3(dist_matrix):
    """The difference between sample and selection size is not large enough for high trust."""
    exp_max_dist_indices = [2, 3]

    exp_combi_distance = np.array(
        [
            [0, 1, np.sqrt(4 ** 2)],
            [0, 2, np.sqrt(5 ** 2)],
            [0, 3, np.sqrt(6 ** 2)],
            [1, 2, np.sqrt(7 ** 2)],
            [1, 3, np.sqrt(8 ** 2)],
            [2, 3, np.sqrt(9 ** 2)],
        ]
    )

    max_dist_indices, combi_distance = select_trajectories(dist_matrix, 2)

    assert_array_equal(exp_max_dist_indices, max_dist_indices)
    assert_array_equal(exp_combi_distance, combi_distance)


def test_select_trajectories_iteration_1():
    """
    Small unit test for `select_trajectories_wrapper_iteration` by reducing
    the sample set by one sample.
    """
    dist_matrix = np.array([[0, 4, 5, 6], [4, 0, 7, 8], [5, 7, 0, 9], [6, 8, 9, 0]])

    exp_max_dist_indices = [2, 3]

    # indices in the array below do not match the original dist_matrix.
    exp_combi_distance = np.array(
        [[0, 1, np.sqrt(7 ** 2)], [0, 2, np.sqrt(8 ** 2)], [1, 2, np.sqrt(9 ** 2)]]
    )

    max_dist_indices, combi_distance = select_trajectories_wrapper_iteration(
        dist_matrix, 2
    )

    assert_array_equal(exp_max_dist_indices, max_dist_indices)
    assert_array_equal(exp_combi_distance, combi_distance)


def test_select_trajectories_iteration_2():
    """
    Small unit test for `select_trajectories_wrapper_iteration` by reducing
    the sample set by one sample.
    """
    test_traj_dist_matrix = np.array(
        [[0, 1, 2, 4], [1, 0, 3, 100], [2, 3, 0, 200], [4, 100, 200, 0]]
    )

    max_dist_indices, _ = select_trajectories(test_traj_dist_matrix, 2)
    max_dist_indices_iter, _ = select_trajectories_wrapper_iteration(
        test_traj_dist_matrix, 2
    )

    assert_array_equal(max_dist_indices, max_dist_indices_iter)


@pytest.fixture
def numbers():
    """Fix numbers for the next four tests."""
    n_inputs = 4
    n_levels = 10
    n_traj_sample = 30
    n_traj = 5

    return [n_inputs, n_levels, n_traj_sample, n_traj]


@pytest.fixture
def sample_traj_list(numbers):
    """Fix sample set for the next four tests."""
    sample_traj_list = list()
    for traj in range(0, numbers[2]):

        seed = 123 + traj
        m_traj, _ = morris_trajectory(numbers[0], numbers[1], seed=seed)
        sample_traj_list.append(m_traj)

    return sample_traj_list


@pytest.fixture
def traj_selection(sample_traj_list, numbers):
    """Fix sample set and distance matrix for the next four tests."""
    select_list, select_distance_matrix, _ = campolongo_2007(sample_traj_list, numbers[3])

    return [select_list, select_distance_matrix]


def test_compare_camp_07_int_ge_men_14_2(numbers, sample_traj_list, traj_selection):
    """
    Tests wether the trajectory set computed by compolongo_2007
    and intermediate_ge_menendez_2014 are reasonably close in terms
    of their total distance.
    """
    select_list_2, select_distance_matrix_2, _ = intermediate_ge_menendez_2014(
        sample_traj_list, numbers[3]
    )

    selection = traj_selection
    dist_camp = total_distance(selection[1])
    dist_gm = total_distance(select_distance_matrix_2)

    assert dist_camp - dist_gm < 0.03 * dist_camp


def test_compare_camp_07_final_ge_men_14_2(numbers, sample_traj_list, traj_selection):
    """
    Tests wether the trajectory set computed by compolongo_2007
    and final_ge_menendez_2014 are reasonably close in terms
    of their total distance.
    Notes
    -----
    Very few times, the difference gets relatively large, see assert statement.
    """
    select_list_2, select_distance_matrix_2, _ = final_ge_menendez_2014(
        sample_traj_list, numbers[3]
    )

    selection = traj_selection
    dist_camp = total_distance(selection[1])
    dist_gm = total_distance(select_distance_matrix_2)

    assert dist_camp - dist_gm < 0.4 * dist_camp


@pytest.mark.skip(
    reason="The following behavior is expected by Ge/Menendez (2014). \
    Oftentimes the test works. \
    However, due to numerical reasons, sometimes intermediate_ge_menendez_2014 \
    selects a different, slightly worse trajectory set\
    compared to campolongo_2007."
)
def test_compare_camp_07_int_ge_men_14_1(numbers, sample_traj_list, traj_selection):
    """
    Tests wether the sample set and distance matrix of the [1] and the first part
    of the improvment in [2] are identical.
    """
    select_list_2, select_distance_matrix_2, _ = intermediate_ge_menendez_2014(
        sample_traj_list, numbers[3]
    )
    selection = traj_selection
    assert_array_equal(np.array(selection[0]), np.array(select_list_2))
    assert_array_equal(traj_selection[1], select_distance_matrix_2)


@pytest.mark.skip(
    reason="The following behavior is expected by Ge/Menendez (2014). \
    Oftentimes the test works. \
    However, due to numerical reasons, sometimes intermediate_ge_menendez_2014 \
    selects a different, slightly worse trajectory set\
    compared to campolongo_2007."
)
def test_compare_camp_07_final_ge_men_14_1(numbers, sample_traj_list, traj_selection):
    """
    Tests wether the sample set and distance matrix of the [1] and the both parts
    of the improvment in [2] are identical.
    """
    traj_list, diagonal_dist_matrix, _ = final_ge_menendez_2014(
        sample_traj_list, numbers[3]
    )
    test_list, test_diagonal_dist_matrix, _ = intermediate_ge_menendez_2014(
        sample_traj_list, numbers[3]
    )

    assert_array_equal(traj_list, test_list)
    assert_array_equal(diagonal_dist_matrix, test_diagonal_dist_matrix)

