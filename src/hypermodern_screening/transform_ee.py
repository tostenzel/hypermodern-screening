"""
Compute the component for the redesigned EE expressions.

Functions to compute the arguments for the function evaluations in the numerator
of the individual uncorrelated and correlated Elementary Effects following [1],
page 33 and 34, and coefficients that scale the step.
These functions can handle samples in both, trajectory and radial, designs.

References
----------
[1] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global
sensitivity analysis of models with dependent inputs. Reliability Engineering &
System Safety 100 (162), 28–39.

"""
from typing import List, Tuple, Union

import numpy as np

from hypermodern_screening.transform_distributions import transform_stnormal_normal_corr
from hypermodern_screening.transform_reorder import (
    ee_corr_reorder_sample,
    ee_uncorr_reorder_sample,
    reorder_cov,
    reorder_mu,
    reverse_ee_corr_reorder_sample,
    reverse_ee_uncorr_reorder_sample,
    reverse_reorder_cov,
    reverse_reorder_mu,
)


def trans_ee_uncorr(
    sample_list: List[np.ndarray], cov: np.ndarray, mu: np.ndarray, radial: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Transform list of samples to two lists of transformed samples.

    (For the computation of the uncorrelated Elementary Effects.)

    Parameters
    ----------
    sample_list
        Set of untransformed samples.
    cov : np.ndarray
        Covariance matrix.
    mu
        Expectation value.
    radial
        Sample is in trajectory or radial design.

    Returns
    -------
    trans_piplusone_i
        samples containing the rows that are the arguments for the LHS function
        evaluation for the uncorrelated Elementary Effect.
    trans_pi_i
        samples containing the rows that are the arguments for the RHS function
        evaluation for the uncorrelated Elementary Effect.
    coeff_step
        Factors in the denumerator of the uncorrelated Elementary Effect. Accounts
        for the decorrelation of the Step.

    Raises
    ------
    AssertionError
        If the dimension of `mu`, `cov` and the elements in `sample_list`
        do not fit together.

    Notes
    -----
    The rows in the two different transformed samples equal to T(p_{i+1}, i)
    and T(p_{i}, i). Understanding the transformations may require to write up the
    first transformation from p_i and p_{i+1} to T_1(p_{i}, i) and T_1(p_{i+1}, i).
    T_1 shifts the first i elements to the end for each row p_{i}.
    This function creates list of transformations of whole samples.
    The rows in the samples for T(p_{i}, i) that are to be subtracted from
    T(p_{i+1}, i), are still positioned one below compared to the samples for
    T(p_{i}, i). Therefore, importantly, one needs to compare each row in a sample from
    `trans_pi_i` with the respective row one below in `trans_piplusone_i`.
    To compute the EEs from radial samples, the arguments of the subtracted function
    are the first row of the sample. Yet, they must be reordered and transformed
    according to their order, too.

    """
    assert len(mu) == len(cov) == np.size(sample_list[0], 1)

    n_sub_sample = len(sample_list)
    n_rows = np.size(sample_list[0], 0)
    zero_idx_diff = []
    one_idx_diff = []

    # Transformation 1.
    for samp in range(0, n_sub_sample):
        z = sample_list[samp]
        one_idx_diff.append(ee_uncorr_reorder_sample(z))

        # The radial design only subtracts the first row - but must be reordered, too.
        if radial is not False:
            z = np.tile(z[0, :], (n_rows, 1))
        else:
            pass

        zero_idx_diff.append(ee_uncorr_reorder_sample(z, row_plus_one=False))

    # Transformation 2 for p_{i+1}.
    # No re-arrangement needed as the first transformation for p_{i+1}
    # is using the original order of mu and cov.

    # ´coeff_step` saves the coefficient from the last element in the Cholesky matrix
    # that transforms the step.
    coeff_step = []
    for samp in range(0, n_sub_sample):

        # Needs to be set up again for each samp - otherwise it'd be one `i`too much.
        mu_one = mu
        cov_one = cov

        c_step = np.ones([n_rows - 1, 1]) * np.nan
        for row in range(0, n_rows):
            (
                one_idx_diff[samp][row, :],
                correlate_step,
            ) = transform_stnormal_normal_corr(
                one_idx_diff[samp][row, :], cov_one, mu_one
            )

            # We do not need the coefficient of the first row.
            if row > 0:
                c_step[row - 1, 0] = correlate_step
            else:
                pass

            mu_one = reorder_mu(mu_one)
            cov_one = reorder_cov(cov_one)
        coeff_step.append(c_step)

    # Transformation 2 for p_i.

    # Need to reorder mu and covariance according to the zero idx difference.
    for samp in range(0, n_sub_sample):
        mu_zero = reorder_mu(mu)
        cov_zero = reorder_cov(cov)
        for row in range(0, n_rows):
            zero_idx_diff[samp][row, :], _ = transform_stnormal_normal_corr(
                zero_idx_diff[samp][row, :], cov_zero, mu_zero
            )
            mu_zero = reorder_mu(mu_zero)
            cov_zero = reorder_cov(cov_zero)

    # Transformation 3: Undo Transformation 1.
    trans_pi_i = []
    trans_piplusone_i = []
    for samp in range(0, n_sub_sample):
        trans_pi_i.append(
            reverse_ee_uncorr_reorder_sample(zero_idx_diff[samp], row_plus_one=False)
        )
        trans_piplusone_i.append(reverse_ee_uncorr_reorder_sample(one_idx_diff[samp]))

    return trans_piplusone_i, trans_pi_i, coeff_step


def trans_ee_corr(
    sample_list: List, cov: np.ndarray, mu: np.ndarray, radial: bool = False
) -> Tuple[List, Union[List, None]]:
    """Transform list of samples to two lists of transformed samples.

    (For the computation of the correlated Elementary Effects.)

    Parameters
    ----------
    sample_list
        Set of untransformed samples.
    cov
        Covariance matrix.
    mu
        Expectation value.
    radial
        Sample is in trajectory or radial design.

    Returns
    -------
    trans_piplusone_iminusone
        samples containing the rows that are the arguments for the LHS function
        evaluation for the correlated Elementary Effect.

    Raises
    ------
    AssertionError
        If the dimension of `mu`, `cov` and the elements in `sample_list`
        do not fit together.

    Notes
    -----
    For the trajectory design, the transformation for the rows on the RHS of the
    correlated Elementary Effects is equal to the one on the LHS of the uncorrelated
    Elementary Effects. Therefore, if `radial is False`, this transformation is skipped
    and left to `trans_ee_uncorr_samples`.
    To compute the EEs from radial samples, the arguments of the subtracted function
    are the first row of the sample. Yet, they must be reordered and transformed
    according to their order, too.

    See Also
    --------
    trans_ee_uncorr_samples

    """
    assert len(mu) == len(cov) == np.size(sample_list[0], 1)

    n_sub_sample = len(sample_list)
    n_rows = np.size(sample_list[0], 0)
    two_idx_diff = []

    # Transformation 1 for p_{i+1}.
    for samp in range(0, n_sub_sample):
        z = sample_list[samp]
        two_idx_diff.append(ee_corr_reorder_sample(z))

    # Transformation 2 for p_{i+1}.
    # Need to reorder mu and cov using the reverse function.
    for samp in range(0, n_sub_sample):
        mu_two = reverse_reorder_mu(mu)
        cov_two = reverse_reorder_cov(cov)
        for row in range(0, n_rows):
            two_idx_diff[samp][row, :], _ = transform_stnormal_normal_corr(
                two_idx_diff[samp][row, :], cov_two, mu_two
            )
            mu_two = reorder_mu(mu_two)
            cov_two = reorder_cov(cov_two)

    # Transformation 3: Undo Transformation 1.
    trans_piplusone_iminusone = []
    for samp in range(0, n_sub_sample):
        trans_piplusone_iminusone.append(
            reverse_ee_corr_reorder_sample(two_idx_diff[samp])
        )

    if radial is False:
        return trans_piplusone_iminusone, None

    else:
        # The radial design cannot reuse p_{i+1} from `trans_ee_uncorr`
        # because EE_corr requires to consider only the first row.
        one_idx_diff = []

        # Transformation 1 for p_{i+1} 2.
        for samp in range(0, n_sub_sample):
            z = sample_list[samp]
            # Only use first row.
            z = np.tile(z[0, :], (n_rows, 1))
            one_idx_diff.append(ee_uncorr_reorder_sample(z))

        # Transformation 2 for p_{i+1}.
        # No re-arrangement needed.
        for samp in range(0, n_sub_sample):
            mu_one = mu
            cov_one = cov
            for row in range(0, n_rows):
                (one_idx_diff[samp][row, :], _) = transform_stnormal_normal_corr(
                    one_idx_diff[samp][row, :], cov_one, mu_one
                )
                mu_one = reorder_mu(mu_one)
                cov_one = reorder_cov(cov_one)

        # Transformation 3: Undo Transformation 1.
        trans_piplusone_i = []
        for samp in range(0, n_sub_sample):
            trans_piplusone_i.append(
                reverse_ee_uncorr_reorder_sample(one_idx_diff[samp])
            )

        return trans_piplusone_iminusone, trans_piplusone_i
