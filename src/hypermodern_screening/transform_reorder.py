"""Functions for reordering the sample rows following [1].

The intuition behind the reordering in general is the following: To compute the
uncorrelated Elementary Effects, one moves the sampled elements that have been changed
by `step` to the back of the row. For the correlated EE, one leaves the newly changed
element in front, but moves the elements that were changed in rows above to the end.
These compose the left parts of the numerator in the EE definition. One then subtracts
the same row, except that the changed element is unchanged. The reason for these
reorderings is that the correlation technique works hierarchically, like Dominoes.
The element before is unaffected by the correlation of the elements thereafter.
This implies that the first element is unchanged, as for the correlated EE. Therefore,
the step is involved in correlating the other elements without becoming changed itself.
The opposite is true for the uncorrelated EE.
The above procedure is dervied from the ordering in trajectory samples. It also works
for the radial design.
Other functions order the expectations and covariance matrix accordingly. They are also
used to initialize the correlating loops in the two functions in `transform_ee.py` in
the right order.

References
----------
[1] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global
sensitivityanalysis of models with dependent inputs. Reliability Engineering &
System Safety 100 (162), 28–39.

"""
import numpy as np


def ee_uncorr_reorder_sample(
    sample: np.ndarray, row_plus_one: bool = True
) -> np.ndarray:
    """For each row i (non-pythonic), move the first i elements to the back.

    Parameters
    ----------
    sample
        sample.
    row_plus_one
        Add 1 to row index, i.e. start with second row.

    Returns
    -------
    sample_reordered
        Reordered sample.

    """
    sample_reordered = np.ones([np.size(sample, 0), np.size(sample, 1)]) * np.nan

    for i in range(0, np.size(sample, 0)):

        if row_plus_one is False:

            # In the first row, put the first element to the back.
            # In the second, the first two etc.
            sample_reordered[i, :] = np.roll(sample[i, :], -(i + 1))

        if row_plus_one is True:

            # In the first row, put 0 elements to the back.
            # In the second, put the first element to the back etc.
            sample_reordered[i, :] = np.roll(sample[i, :], -(i))

    return sample_reordered


def reverse_ee_uncorr_reorder_sample(
    sample_reordered: np.ndarray, row_plus_one: bool = True
) -> np.ndarray:
    """Reverse of function `uncorr_reorder_sample`.

    Parameters
    ----------
    sample_reordered : ndarray
        Reordered sample.

    Returns
    -------
    sample : ndarray
        Trjectory in original order.

    """
    sample = (
        np.ones([np.size(sample_reordered, 0), np.size(sample_reordered, 1)]) * np.nan
    )

    for i in range(0, np.size(sample_reordered, 0)):

        if row_plus_one is False:
            sample[i, :] = np.roll(
                sample_reordered[i, :], -(np.size(sample_reordered, 1) - (i + 1))
            )

        if row_plus_one is True:
            sample[i, :] = np.roll(
                sample_reordered[i, :], -(np.size(sample_reordered, 1) - (i))
            )

    return sample


def ee_corr_reorder_sample(sample: np.ndarray) -> np.ndarray:
    """For each row i (non-pythonic), move the first i-1 elements to the back.

    Parameters
    ----------
    sample
        sample.

    Returns
    -------
    sample_reordered
        Reordered sample.

    Notes
    -----
    There is no `row_plus_one=False` option because this is equivalent
    with `uncorr_reorder_sample(sample, row_plus_one=True)`.

    """
    sample_reordered = np.ones([np.size(sample, 0), np.size(sample, 1)]) * np.nan

    for i in range(0, np.size(sample, 0)):

        # In the first row, put the first two elements to the back.
        # In the second row, put the first three element to the back etc.
        sample_reordered[i, :] = np.roll(sample[i, :], -(i - 1))

    return sample_reordered


def reverse_ee_corr_reorder_sample(sample_reordered: np.ndarray) -> np.ndarray:
    """Reverse of function `corr_reorder_sample`.

    Parameters
    ----------
    sample_reordered
        Reordered sample.

    Returns
    -------
    sample
        Trjectory in original order.

    """
    sample = (
        np.ones([np.size(sample_reordered, 0), np.size(sample_reordered, 1)]) * np.nan
    )

    for i in range(0, np.size(sample, 0)):
        sample[i, :] = np.roll(
            sample_reordered[i, :], -(np.size(sample_reordered, 1) - (i - 1))
        )

    return sample


def reorder_mu(mu: np.ndarray) -> np.ndarray:
    """Move the first element of the expectation vector to the end.

    Parameters
    ----------
    mu
        Expectation values of row.

    Returns
    -------
    mu_reordered
        Reordered expectation values of row.

    """
    mu_reordered = np.roll(mu, -1)

    return mu_reordered


def reorder_cov(cov: np.ndarray) -> np.ndarray:
    """Arrange covariance matrix according to the expectation vector.

    (When the first element is moved to the end.)

    Parameters
    ----------
    cov
        Covariance matrix of row.

    Returns
    -------
    cov_reordered
        Reordered covariance matrix of row.

    """
    cov_reordered = np.ones(cov.shape) * np.nan

    # Put untouched square one up and one left
    cov_reordered[0 : len(cov) - 1, 0 : len(cov) - 1] = cov[1 : len(cov), 1 : len(cov)]

    # Put [0,0] to [n,n]
    cov_reordered[len(cov) - 1, len(cov) - 1] = cov[0, 0]

    # Put [0, 1:n] to [n, 0:n-1] and same for the column.
    cov_reordered[len(cov) - 1, 0 : len(cov) - 1] = cov[0, 1 : len(cov)]
    cov_reordered[0 : len(cov) - 1, len(cov) - 1] = cov[0, 1 : len(cov)]

    return cov_reordered


def reverse_reorder_mu(mu_reordered: np.ndarray) -> np.ndarray:
    """Reverse of function `reorder_mu`.

    Parameters
    ----------
    mu_reordered
        Reordered expectation values of row.

    Returns
    -------
    mu
        Expectation values of row in original order.

    """
    mu = np.roll(mu_reordered, +1)

    return mu


def reverse_reorder_cov(cov_reordered: np.ndarray) -> np.ndarray:
    """
    Reverse of function `reorder_cov`.

    Parameters
    ----------
    cov_reordered : ndarray
        Reordered covariance matrix.

    Returns
    -------
    cov : ndarray
        Covarince matrix in original order.

    """
    cov = np.ones(cov_reordered.shape) * np.nan

    cov[1 : len(cov_reordered), 1 : len(cov_reordered)] = cov_reordered[
        0 : len(cov_reordered) - 1, 0 : len(cov_reordered) - 1
    ]
    cov[0, 0] = cov_reordered[len(cov_reordered) - 1, len(cov_reordered) - 1]

    cov[0, 1 : len(cov_reordered)] = cov_reordered[
        len(cov_reordered) - 1, 0 : len(cov_reordered) - 1
    ]
    cov[1 : len(cov_reordered), 0] = cov_reordered[
        0 : len(cov_reordered) - 1, len(cov_reordered) - 1
    ]

    return cov
