import numpy as np

from numpy.testing import assert_array_equal

from hypermodern_screening.sampling_schemes import trajectory_sample
from hypermodern_screening.transform_ee import trans_ee_uncorr


def test_trans_ee_uncorr_trajectories():
    """
    Intregation test for `trans_ee_uncorr_trajectories`

    Notes
    -----
    -This test provides strong evidence that the whole transformation
    of the trajectories for the uncorrelated Elementary Effects is correct.
    As written in [1], page 34: The elements in vectors T(p_{i}, i) and
    T(p_{i+1}, i) are the same except of the ith element. This means that -
    without the first row in `trans_pi_i` and the last row in `trans_pi_i` -
    only the diagonal elements of the two trajectories generated by
    `trans_ee_uncorr_trajectories` are different. This is because in T(p_{i}, i)
    to these elements the step is added before the correlation transformation.
    -`ee_corr_trajectories`, however, is hard to test because there, all elements
    differ if there are correlations.

    See Also
    --------
    `trans_ee_uncorr_trajectories`.

    References
    ----------
    [1] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global
    sensitivityanalysis of models with dependent inputs. Reliability Engineering &
    System Safety 100 (162), 28–39.

    """
    mu = np.array([10, 11, 12, 13, 14])

    cov = np.array(
        [
            [10, 0, 0, 2, 0.5],
            [0, 20, 0.4, 0.15, 0],
            [0, 0.4, 30, 0.05, 0],
            [2, 0.15, 0.05, 40, 0],
            [0.5, 0, 0, 0, 50],
        ]
    )

    n_inputs = 5
    n_levels = 10
    n_sample = 10
    sample_traj_list, _ = trajectory_sample(n_sample, n_inputs, n_levels)

    trans_one, trans_zero, _ = trans_ee_uncorr(sample_traj_list, cov, mu, radial=False)

    for traj in range(0, len(trans_zero)):
        for row in range(0, np.size(trans_zero[0], 0) - 1):
            zero = np.delete(trans_zero[traj][row, :], row)
            one = np.delete(trans_one[traj][row + 1, :], row)
            assert_array_equal(zero, one)