import numpy as np

from numpy.testing import assert_array_equal

from hypermodern_screening.sampling_schemes import morris_trajectory
from hypermodern_screening.sampling_schemes import radial_sample


def test_morris_trajectory_value_grid():
    """
    Tests wether the point grid is composed of the rights values.

    Notes
    -----
    `morris_trajectory` is hard to test because there are many random
    objects involved. A good check is to have a look at a large number
    of trajectories as the conditions that they should meet are easy
    to recognize.

    """
    n_levels = 10
    # Many inputs for high probability to catch all grid points in trajectory.
    n_inputs = 100

    traj, _ = morris_trajectory(n_inputs, n_levels, seed=123)

    # Round the elements in both sets.
    grid_flat_list = [round(item, 6) for sublist in traj.tolist() for item in sublist]
    grid = set(grid_flat_list)

    expected = np.around((np.linspace(0, 9, 10) / (n_levels - 1)), 6)
    expected = set(expected.tolist())

    assert grid == expected


def test_radial_sample():
    """
    Tests wether for each row i (non-pythonic), only the ith elements is
    different from the first row for a sample of radial subsamples.

    """
    n_rads = 10
    n_params = 5

    rad_list, _ = radial_sample(n_rads, n_params)

    for rad in range(0, n_rads - 1):
        for row in range(0, n_params):
            assert_array_equal(
                np.delete(rad_list[rad][row + 1, :], row),
                np.delete(rad_list[rad][0, :], row),
            )
