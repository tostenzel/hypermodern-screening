"""Decrease a set of sample sets in order to increase its representativeness.

These approaches are developed in the context of the trajectory design because
it can not cover the space very densely.
The methods are taken from the effective screening design in [1] and the
efficient screening design in [2].

References
----------
[1] Campolongo, F., J. Cariboni, and A. Saltelli (2007). An effective screening design
for sensitivity analysis of large models. Environmental modelling & software 22 (10),
1509–1518.
[2] Ge, Q. and M. Menendez (2014). An efficient sensitivity analysis approach for
computationally expensive microscopic traffic simulation models. International Journal
of Transportation 2 (2), 49–64.

"""
from itertools import combinations
from typing import Iterable, List, Tuple

import numpy as np
from scipy.special import binom

from hypermodern_screening.transform_distributions import (
    transform_uniform_stnormal_uncorr,
)


def compute_pair_distance(sample_0: np.ndarray, sample_1: np.ndarray) -> float:
    """Compute the distance measure between a pair of samples.

    The aggregate distance between sum of the root of the square distance between each
    parameter vector of one sample to each vector of the other sample.

    Parameters
    ----------
    sample_0
        Sample with paramters in cols and draws as rows.
    sample_1
        Sample with paramters in cols and draws as rows.

    Returns
    -------
    distance
        Pair distance.

    Raises
    ------
    AssertionError
        If sample is not in trajectory or radial design shape.
    AssertionError
        If the sample shapes differ.

    Notes
    -----
    The distance between two samples is sum of the root of the square distance between
    each parameter vector of one sample to each vector of the other sample.

    """
    distance = 0
    assert np.size(sample_0, 0) == np.size(sample_0, 1) + 1
    assert sample_0.shape == sample_1.shape

    if np.any(np.not_equal(sample_0, sample_1)):
        for col_0 in range(0, np.size(sample_0, 1)):
            for col_1 in range(0, np.size(sample_1, 1)):
                distance += np.sqrt(sum((sample_0[:, col_0] - sample_1[:, col_1]) ** 2))
    else:
        pass

    return distance


def distance_matrix(sample_list: List[np.ndarray]) -> np.ndarray:
    """Compute symmetric matrix of pair distances for a list of samples.

    Parameters
    ----------
    sample_list
        Set of samples.

    Returns
    -------
    distance_matrix
        Symmatric matrix of pair distances.

    """
    distance_matrix = np.nan * np.ones(shape=(len(sample_list), len(sample_list)))
    for i in range(0, len(sample_list)):
        for j in range(0, len(sample_list)):
            distance_matrix[i, j] = compute_pair_distance(
                sample_list[i], sample_list[j]
            )

    return distance_matrix


def combi_wrapper(iterable: Iterable, r: int) -> List[List]:
    """Wrap `itertools.combinations`, written in C, see [1].

    Parameters
    ----------
    iterable : iterable object
        Hashable container like a list of distinct elements to combine.

    r : int
        Number to draw from `iterable` with putting back and regarding the order.

    Returns
    -------
    list_list
        All possible combinations in ascending order.

    Example
    -------
    >>> combi_wrapper([0, 1, 2, 3], 2)
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    References
    ----------
    [1] https://docs.python.org/2/library/itertools.html#itertools.combinations.

    """
    tup_tup = combinations(iterable, r)
    list_list = [list(x) for x in tup_tup]

    return list_list


def total_distance(distance_matrix: np.ndarray) -> float:
    """Compute the total distance measure of all pairs of samples in a set.

    The equation corresponds to Equation (10) in [2].

    Parameters
    ----------
    distance_matrix : ndarray
        diagonal matrix of distances for sample pairs.

    Returns
    -------
    total_distance: float
        total distance measure of all pairs of samples in a set.

    """
    # The `*0.5` is implemented by only considering the lower triangular.
    total_distance = np.sqrt(sum(sum(np.tril(distance_matrix ** 2))))  # type:ignore

    return total_distance


def select_trajectories(
    pair_dist_matrix: np.ndarray, n_traj: int
) -> Tuple[List, np.ndarray]:
    """Compute `total distance` for each `n_traj` combinations of a set of samples.

    Parameters
    ----------
    pair_dist_matrix
        `distance_matrix` for a sample set.
    n_traj
        Number of sample combinations for which the `total_distance` is computed.

    Returns
    -------
    max_dist_indices : list of ints
        Indices of samples in `pair_dist_matrix` that are part of the combination
        with the largest `total_distance`.
    combi_total_distance : ndarray
        Matrix with n_traj + 1 rows. The first n_traj cols are filled with indices
        of samples and the last column is the `total_distance` of the combinations
        of samples marked by indices in the same row and the columns before.

    Raises
    ------
    AssertionError
        If `pair_dist_matrix` is not symmetric.
    AssertionError
        If the number of combinations does not correspong to the combinations
        indicated by the size of `pair_dist_matrix`.

    Notes
    -----
    This function can be very slow because it computes distances
    between np.binomial(len(pair_dist_matrix, n_traj) pairs of trajectories.
    Example: `np.biomial(30,15)` = 155117520.
    This selection function yields precise results
    because each total distance for each possible combination of
    trajectories is computed directly. The faster, iterative methods
    can yield different results that are, however, close in the total
    distance. The total distances tend to differentiate clearly.
    Therefore, the optimal combination is precisely determined.

    """
    assert np.all(np.abs(pair_dist_matrix - pair_dist_matrix.T) < 1e-8)
    # Get all possible combinations of input parameters by their indices.
    combi = combi_wrapper(list(np.arange(0, np.size(pair_dist_matrix, 1))), n_traj)
    assert len(combi) == binom(np.size(pair_dist_matrix, 1), n_traj)
    # leave last column open for total distance
    combi_total_distance = np.ones([len(combi), n_traj + 1]) * np.nan
    combi_total_distance[:, 0:n_traj] = np.array(combi)

    # This loop could be parallelized.
    for row in range(0, len(combi)):
        # Assign last column
        combi_total_distance[row, n_traj] = 0
        pair_combi = combi_wrapper(combi[row], 2)
        for pair in pair_combi:
            # Aggreate the pair distance to the total distance of the
            # trajectory combination.
            # There is no * 0.5 in contrary to Ge/Menendez (2014) because
            # this only uses half of the matrix.
            combi_total_distance[row, n_traj] += (
                pair_dist_matrix[int(pair[0])][int(pair[1])] ** 2
            )
    combi_total_distance[:, n_traj] = np.sqrt(combi_total_distance[:, n_traj])
    # Select indices of combination that yields highest total distance.
    max_dist_indices_row = combi_total_distance[:, n_traj].argsort()[-1:][::-1].tolist()
    max_dist_indices = combi_total_distance[max_dist_indices_row, 0:n_traj]
    # Convert list of float indices to list of ints.
    max_dist_indices = [int(i) for i in max_dist_indices.tolist()[0]]

    return max_dist_indices, combi_total_distance


def select_trajectories_wrapper_iteration(
    pair_dist_matrix: np.ndarray, n_traj: int
) -> Tuple[List, np.ndarray]:
    """Select the set of samples minus one sample.

    Used for selecting iteratively rather than by brute force.
    Implements the main step of the essential of the two "improvements"
    from [2] to [1].

    Parameters
    ----------
    pair_dist_matrix : ndarray
        Distance matrix of all combinations and their total_distance.
    n_traj : int
        number of samples to choose from a set of samples based on their
        `total_distance`.

    Returns
    -------
    tracker_keep_indices : list
        Indices of samples part of the selection.
    combi_total_distance : ndarray
        Matrix with n_traj + 1 rows. The first n_traj cols are filled with indices
        of samples and the last column is the `total_distance` of the combinations
        of samples marked by indices in the same row and the columns before.

    See Also
    --------
    select_trajectories

    Notes
    -----
    Oftentimes this function leads to diffent combinations than
    `select_trajectories`. The reason seems to be that this function
    deviates from the optimal path due to numerical reasons as different
    combinations may be very close (see [2]).
    However, the total sum of the returned combinations are close.
    Therefore, the `total_distance` loss is negligible compared to the speed gain
    for large numbers of trajectory combinations.
    This implies that, `combi_total_distance` always differs from the one in
    `select_trajectories` because it only contains the combination indices from
    the last iteration if n_traj is smaller than the sample set minus 1.
    The trick using `tracker_keep_indices` is an elegant solution.

    """
    n_traj_sample = np.size(pair_dist_matrix, 0)
    tracker_keep_indices = np.arange(0, np.size(pair_dist_matrix, 0))
    for _i in range(0, n_traj_sample - n_traj):

        indices = np.arange(0, np.size(pair_dist_matrix, 0)).tolist()
        # get list of all indices
        # get list of surviving indices
        max_dist_indices, combi_total_distance = select_trajectories(
            pair_dist_matrix, np.size(pair_dist_matrix, 0) - 1
        )
        # lost index
        lost_index = [item for item in indices if item not in max_dist_indices][0]

        # delete pairs with dropped trajectory from distance matrix
        pair_dist_matrix = np.delete(pair_dist_matrix, lost_index, axis=0)
        pair_dist_matrix = np.delete(pair_dist_matrix, lost_index, axis=1)
        tracker_keep_indices = np.delete(
            tracker_keep_indices, lost_index, axis=0
        ).tolist()

    return tracker_keep_indices, combi_total_distance


def campolongo_2007(
    sample_traj_list: List, n_traj: int
) -> Tuple[List, np.ndarray, List]:
    """Implement the post-selected sample set in [1].

    Takes a list of Morris trajectories and selects the `n_traj` trajectories
    with the largest distance between them.
    Returns the selection as array with n_inputs at the verical and n_traj at the
    horizontal axis and as a list.
    It also returns the diagonal matrix that contains the pair distance
    between each trajectory pair.

    Parameters
    ----------
    sample_traj_list : list of ndarrays
        Set of samples.
    n_traj : int
        Number of samples to choose from `sample_traj_list`.

    Returns
    -------
    sample_traj_list : list of ndarrays
        Set of trajectories.
    select_dist_matrix : ndarray
        Symmetric `distance_matrix` of selection.
    select_indices : list
        Indices of selected samples.

    """
    pair_matrix = distance_matrix(sample_traj_list)
    select_indices, combi_total_distance = select_trajectories(pair_matrix, n_traj)

    select_trajs = [sample_traj_list[idx] for idx in select_indices]

    select_dist_matrix = distance_matrix(select_trajs)

    return select_trajs, select_dist_matrix, select_indices


def intermediate_ge_menendez_2014(
    sample_traj_list: List, n_traj: int
) -> Tuple[List, np.ndarray, List]:
    """Implement the essential of the two "improvements" in[2] vis-a-vis [1].

    This is basically a wrapper around `select_trajectories_wrapper_iteration`.

    Parameters
    ----------
    sample_traj_list : list of ndarrays
        Set of samples.
    n_traj : int
        Number of samples to choose from `sample_traj_list`.

    Returns
    -------
    sample_traj_list : list of ndarrays
        Set of trajectories.
    select_dist_matrix : ndarray
        Symmetric `distance_matrix` of selection.
    select_indices : list
        Indices of selected samples.

    See Also
    --------
    select_trajectories_wrapper_iteration

    Notes
    -----
    Oftentimes this function leads to diffent combinations than
    `select_trajectories`. However, their total distance is very close
    to the optimal solution.

    """
    pair_matrix = distance_matrix(sample_traj_list)
    # This function is the difference to Campolongo.
    select_indices, _ = select_trajectories_wrapper_iteration(pair_matrix, n_traj)

    select_trajs = [sample_traj_list[idx] for idx in select_indices]

    select_dist_matrix = distance_matrix(select_trajs)

    return select_trajs, select_dist_matrix, select_indices


def next_combi_total_distance_gm14(
    combi_total_distance: np.ndarray, pair_dist_matrix: np.ndarray, lost_index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the set of samples minus one sample.

    Based on the algorithmic computation of the `total_distance` proposed by [2].
    I.e. by re-using and adjusting the first `combi_total_distance` matrix each
    iteration. Used for selecting iteratively rather than by brute force.

    Parameters
    ----------
    combi_total_distance_next
        Matrix with n_traj + 1 rows. The first n_traj cols are filled with indices
        of samples and the last column is the `total_distance` of the combinations
        of samples marked by indices in the same row and the columns before.
    pair_dist_matrix
        Distance matrix of all combinations and their total_distance.
    lost_index
        index of the sample that will be dropped from the samples in the above objects.

    Returns
    -------
    combi_total_distance_next
        `combi_total_distance` without the dropped sample.
    pair_dist_matrix_next
        `pair_dist_matrix` without the dropped sample.
    lost_index
        `lost_index` without the dropped sample one iteration before.

    Notes
    -----
    The function computes the total distance of each trajectory
    combination by using the total distance of each combination in the previous step
    and subtracting each pair distance with the dropped trajectory, that yielded
    the lowest total distance combinations in the previous step.
    This function, is in fact much slower than
    `select_trajectories_wrapper_iteration` because it uses more for loops to get
    the pair distances from the right combinations that must be subtracted from the
    total distances.

    """
    old_combi_total_distance = combi_total_distance
    old_pair_dist_matrix = pair_dist_matrix
    # Want to select all trajectories but one which is given by the length of
    # the dist_matrix
    n_traj_sample_old = np.size(old_pair_dist_matrix, 0)
    n_traj_sample = np.size(old_pair_dist_matrix, 0) - 1
    n_traj = np.size(old_pair_dist_matrix, 0) - 2
    remained_indices = np.arange(0, n_traj_sample_old).tolist()
    # This step shows that the indices in combi_total_distance_next mapp
    # to the indices in the old version. The index of the worst traj is missing.
    remained_indices.remove(lost_index)
    # Get all n_traj_sample combintaions from the indices above.
    combi_next = combi_wrapper(remained_indices, n_traj)
    # The  + 1 to n_traj is for the total distance.
    combi_total_distance_next = np.ones([len(combi_next), n_traj + 1]) * np.nan
    combi_total_distance_next[:, 0:n_traj] = np.array(combi_next).astype(int)

    # Compute the sum of squared pair distances
    # that each trajectory in new combination has with the lost trajectory.
    for row in range(0, len(combi_next)):
        sum_dist_squared = 0
        # - 1 is to no spare the total ditance column.
        for col in range(0, n_traj):
            # Get the distance between lost index trajectory and present ones in row.
            sum_dist_squared += (
                old_pair_dist_matrix[
                    int(combi_total_distance_next[row, col]), lost_index
                ]
            ) ** 2

            # Map old total distance to total distance for new combination of
            # trajectories.
            for row_old in range(0, np.size(old_combi_total_distance, 0)):
                # Construct the specific indices of each combi
                # in the old combi_total_distance matrix from the new combi and the lost
                # trajectories.

                # For each traj combination of (n_traj_sample_old - 2) trajs,
                # the lost index is added to get the total distance from
                # old_combi_total_distance that contains (n_traj_sample_old - 1)
                # trajectory combinations and their total distances.
                indices_in_old_combi_dist = [
                    float(idx_new_trajs)
                    for idx_new_trajs in combi_total_distance_next[
                        row, 0:n_traj
                    ].tolist()
                ]
                indices_in_old_combi_dist.append(float(lost_index))
                # Obtain total distances of new combinations by subtracting
                # the respective sum of old squared distances
                if set(indices_in_old_combi_dist) == set(
                    old_combi_total_distance[row_old, 0 : n_traj_sample_old - 1]
                ):
                    # + 1 because the total distance columns must be changed.
                    # - one because its the new matrix?
                    combi_total_distance_next[row, n_traj_sample - 1] = np.sqrt(
                        old_combi_total_distance[row_old, n_traj_sample] ** 2
                        - sum_dist_squared
                    )
                else:
                    pass

        # Dissolving the mapping from old to new combi_total_distance by decreasing the
        # indices that are larger than lost_index by 1.
        combi_total_distance_next[:, 0:n_traj] = np.where(
            combi_total_distance_next[:, 0:n_traj] > lost_index,
            combi_total_distance_next[:, 0:n_traj] - 1,
            combi_total_distance_next[:, 0:n_traj],
        )

    # Select indices of combination that yields highest total distance.
    max_dist_indices_next_row = (
        combi_total_distance_next[:, n_traj].argsort()[-1:][::-1].tolist()
    )
    max_dist_indices_next = combi_total_distance_next[
        max_dist_indices_next_row, 0:n_traj
    ]
    # Convert list of float indices to list of ints.
    max_dist_indices_next = [int(i) for i in max_dist_indices_next.tolist()[0]]

    pair_dist_matrix_next = np.delete(old_pair_dist_matrix, lost_index, axis=0)
    pair_dist_matrix_next = np.delete(pair_dist_matrix_next, lost_index, axis=1)

    lost_index_next = [
        item
        for item in list(np.arange(0, n_traj + 1))
        if item not in max_dist_indices_next
    ][0]

    return combi_total_distance_next, pair_dist_matrix_next, lost_index_next


def final_ge_menendez_2014(
    sample_traj_list: List[np.ndarray], n_traj: int
) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    """
    Implement both "improvements" in [2] vis-a-vis [1].

    Parameters
    ----------
    sample_traj_list
        Set of samples.
    n_traj
        Number of samples to choose from `sample_traj_list`.

    Returns
    -------
    sample_traj_list
        Set of trajectories.
    select_dist_matrix
        Symmetric `distance_matrix` of selection.
    select_indices
        Indices of selected samples.

    See Also
    --------
    next_combi_total_distance_gm14

    Notes
    -----
    This function, is in fact much slower than `intermediate_ge_menendez_2014`
    because it uses more for loops to get the pair distances from the right
    combinations that must be subtracted from the total distances.
    This function selects n_traj trajectories from n_traj_sample trajectories by
    iteratively selecting n_traj_sample - i for i = 1,...,n_traj_sample - n-traj.
    For this purpose, next_combi_total_distance_gm14 computes the total distance
    of each trajectory combination by using the total distance of each combination
    in the previous step and subtracting each pair distance with the dropped trajectory,
    that yielded the lowest total distance combinations in the previous step.

    """
    n_traj_sample = len(sample_traj_list)
    # Step 1: Compute pair distance and total distances for the trajectory
    # combinations.
    pair_dist_matrix = distance_matrix(sample_traj_list)

    # Step 2: Compute total distances for combinations and identify worst trajectory.
    max_dist_indices, combi_total_distance = select_trajectories(
        pair_dist_matrix, len(sample_traj_list) - 1
    )

    # Get lost index from first intitial selection (to get previous
    # combi_total_distance). This index is used to access the pair distance
    # with the lost trajectory in the old pair distance matrix.
    # They are subtracted from the total in the old combi_total distance.
    indices = np.arange(0, len(pair_dist_matrix)).tolist()
    lost_index = [item for item in indices if item not in max_dist_indices][0]

    # Init index tracker and delete first index.
    tracker_keep_indices = np.arange(0, len(pair_dist_matrix)).tolist()
    tracker_keep_indices = np.delete(tracker_keep_indices, lost_index, axis=0)

    #  ... to (n_traj_sample - n_traj - 1), the minus one is because
    # this is already Step 2.
    for _i in range(0, n_traj_sample - n_traj - 1):

        # Use shrink trick for largest loop.
        (
            combi_total_distance,
            pair_dist_matrix,
            lost_index,
        ) = next_combi_total_distance_gm14(
            combi_total_distance, pair_dist_matrix, lost_index
        )

        tracker_keep_indices = np.delete(tracker_keep_indices, lost_index, axis=0)

    select_indices = tracker_keep_indices.tolist()
    select_trajs = [sample_traj_list[idx] for idx in select_indices]

    select_dist_matrix = distance_matrix(select_trajs)

    return select_trajs, select_dist_matrix, select_indices


def select_sample_set_normal(
    samp_list: List[np.ndarray], n_select: int, numeric_zero: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Post-select set of samples based on [0,1] and transform it to stnormal space.

    Parameters
    ----------
    samp_list
        Sub-samples.
    n_select
        Number of sub-samples to select from `samp_list`.
    numeric_zero
        `if normal is True`: Prevents `scipy.normal.ppt` to return `-Inf`
        and `Inf` for 0 and 1.

    Returns
    -------
    samp_list
    steps_list

    Notes
    -----
    Function for post-selection is `intermediate_ge_menendez_2014` because it is the
    fastest.

    See Also
    --------
    intermediate_ge_menendez_2014

    """
    # Post- select `samp_list`.
    samp_list, _, _ = intermediate_ge_menendez_2014(samp_list, n_select)

    steps_list = []

    for i in range(0, n_select):
        samp_list[i] = np.apply_along_axis(
            transform_uniform_stnormal_uncorr, 1, samp_list[i], numeric_zero
        )
        steps_list.append(samp_list[i][-1, :] - samp_list[i][0, :])

    return samp_list, steps_list
