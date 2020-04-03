"""Compute Elementary Effects from transformed samples and derived measures.

Computes the screening measures for correlated inputs that I improved upon
[1] by adjusting the step in the denumeroter to the transformed step in the
nominator in order to not violate the definition of the function derivative.

References
----------
[1] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global
sensitivityanalysis of models with dependent inputs. Reliability Engineering &
System Safety 100 (162), 28–39.

"""
from typing import Callable, List, Tuple

import numpy as np

from hypermodern_screening.transform_ee import trans_ee_corr, trans_ee_uncorr


def screening_measures(
    function: Callable,
    traj_list: List[np.ndarray],
    step_list: List[np.ndarray],
    cov: np.ndarray,
    mu: np.ndarray,
    radial: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute screening measures for a set of paramters.

    Parameters
    ----------
    function
        Function or Model of which its parameters are subject to screening.
    traj_list
        List of transformed trajectories according to [1].
    step_list
        List of steps that each parameter takes in each trajectory.
    cov
        Covariance matrix of the input parameters.
    mu
        Expectation values of the input parameters.
    radial
        Sample is in trajectory or radial design.

    Returns
    -------
    measures_list
       contains:
            ee_uncorr
                Mean uncorrelated Elementary Effect for each parameter.
            ee_corr
                Mean correlated Elementary Effect for each parameter.
            abs_ee_uncorr
                Mean absolute uncorrelated Elementary Effect for each parameter.
            abs_ee_corr
                Mean absolute correlated Elementary Effect for each parameter.
            sd_ee_uncorr
                SD of uncorrelated Elementary Effects for each parameter.
            sd_ee_corr
                SD of correlated Elementary Effects for each parameter.
    obs_list
        contains:
            ee_uncorr_i
                Observations of uncorrelated Elementary Effects.
            ee_corr_i
                Observations of correlated Elementary Effects.

    Notes
    -----
    The samples can be in trajectory or in radial design and the deviates can be
    from an arbitrary (correlated) normal distribution or an uncorrelated
    Uniform[0,1] distribution.
    Unorrelated uniform paramters require different interpretion of `mu`
    as a scaling summand rather than the expectation value.
    It might be necessary to multiply the SDs by `(n_trajs/(n_trajs - 1))`
    for the precise formula. However, this leads to problems for the case
    of only one trajectory - which is used in
    `test_screening_measures_uncorrelated_g_function`.

    """
    n_trajs = len(traj_list)
    n_rows = np.size(traj_list[0], 0)
    n_inputs = np.size(traj_list[0], 1)

    # Compute the transformed trajectory lists/function arguments.

    # `If radial is true` `trans_pi_i_list` can be reused as ``trans_pi_iminus1_list`
    # in subtrahend of EE_uncorr.
    trans_piplusone_i_list, trans_pi_i_list, coeff_step = trans_ee_uncorr(
        traj_list, cov, mu, radial
    )
    # Fix at False b/c first output is unaffected by `radial`.
    trans_piplusone_iminusone_list, _ = trans_ee_corr(traj_list, cov, mu, radial=False)

    # Init function evals.
    fct_evals_pi_i = np.ones([n_rows, n_trajs]) * np.nan
    fct_evals_piplusone_i = np.ones([n_rows, n_trajs]) * np.nan
    fct_evals_piplusone_iminusone = np.ones([n_rows, n_trajs]) * np.nan

    # Compute the function evaluations for each transformed trajectory list.
    for traj in range(0, n_trajs):
        for row in range(0, n_rows):

            # In trajectory design `piplusone_i` is used as minuend and subtrahend.
            if radial is False:
                fct_evals_piplusone_i[row, traj] = function(
                    *trans_piplusone_i_list[traj][row, :]
                )
                # pi_i is not reused for trajs, we do not need last row ofsubtrahend
                if row < n_rows - 1:
                    fct_evals_pi_i[row, traj] = function(*trans_pi_i_list[traj][row, :])

            # For radial design, we do not need first row of subtrahend.
            else:

                # pi_i is reused as pi_i-1 for rads, we need all rows (see line 81-85).
                fct_evals_pi_i[row, traj] = function(*trans_pi_i_list[traj][row, :])
                if row < n_rows - 1:
                    fct_evals_piplusone_i[row + 1, traj] = function(
                        *trans_piplusone_i_list[traj][row + 1, :]
                    )
                else:
                    pass

            # We do not need first row of minuend.
            if row < n_rows - 1:
                fct_evals_piplusone_iminusone[row + 1, traj] = function(
                    *trans_piplusone_iminusone_list[traj][row + 1, :]
                )
            else:
                pass

    # Init individual EEs.
    ee_uncorr_i = np.ones([n_inputs, n_trajs]) * np.nan
    ee_corr_i = np.ones([n_inputs, n_trajs]) * np.nan

    # Compute the individual Elementary Effects for each parameter draw.
    for traj in range(0, n_trajs):
        # uncorr Elementary Effects for each trajectory (for each parameter).
        ee_uncorr_i[:, traj] = (
            fct_evals_piplusone_i[1 : n_inputs + 1, traj]
            - fct_evals_pi_i[0:n_inputs, traj]
        ) / (
            step_list[traj]
            * np.squeeze(coeff_step[traj])
            * np.squeeze(np.sqrt(np.diag(cov)))
        )
        # Above, we additionally need to account for the decorrelation
        # when we account for the scaling by the SD.

    if radial is False:
        for traj in range(0, n_trajs):
            ee_corr_i[:, traj] = (
                fct_evals_piplusone_iminusone[1 : n_inputs + 1, traj]
                - fct_evals_piplusone_i[0:n_inputs, traj]
            ) / (step_list[traj] * np.squeeze(np.sqrt(np.diag(cov))))
            # Above, account for the scaling by the SD.
    else:
        for traj in range(0, n_trajs):

            # Move last row of f-evals of pi_i=p1_i to top to recycle it.
            temp = np.roll(fct_evals_pi_i[0:n_inputs, traj], 1)

            ee_corr_i[:, traj] = (
                fct_evals_piplusone_iminusone[1 : n_inputs + 1, traj] - temp[0:n_inputs]
            ) / (step_list[traj] * np.squeeze(np.sqrt(np.diag(cov))))
            # Above, account for the scaling by the SD.

    # Init measures.
    ee_uncorr = np.ones([n_inputs, 1]) * np.nan
    abs_ee_uncorr = np.ones([n_inputs, 1]) * np.nan
    sd_ee_uncorr = np.ones([n_inputs, 1]) * np.nan

    ee_corr = np.ones([n_inputs, 1]) * np.nan
    abs_ee_corr = np.ones([n_inputs, 1]) * np.nan
    sd_ee_corr = np.ones([n_inputs, 1]) * np.nan

    # Compute the aggregate screening measures.
    ee_uncorr, abs_ee_uncorr, sd_ee_uncorr = compute_measures(ee_uncorr_i)

    ee_corr, abs_ee_corr, sd_ee_corr = compute_measures(ee_corr_i)

    measures_list = [
        ee_uncorr,
        ee_corr,
        abs_ee_uncorr,
        abs_ee_corr,
        sd_ee_uncorr,
        sd_ee_corr,
    ]

    obs_list = [ee_uncorr_i, ee_corr_i]

    return measures_list, obs_list


one = np.array([1])


def compute_measures(
    ee_i: np.ndarray,
    sd_x: np.ndarray = one,
    sd_y: np.ndarray = one,
    sigma_norm: bool = False,
    ub: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute aggregate measures based on (individual) Elementary Effects.

    Paramters
    ---------
    ee_i
        (individual) Elementary Effects of input paramters (cols).
    sd_x
        Parameters' SD.
    sd_y
        QoI's SD.
    sigma_norm
        Indicates wether to compute measures normalized by `sd_x / sd_y`.
    ub
        Indicates wether to compute squared EEs and measures normalized by
        `var_x / var_y`.

    Returns
    -------
    measures_list
       contains:
            ee_mean
                Mean Elementary Effect for each parameter.
            ee_abs_mean
                Mean absolute correlated Elementary Effect for each parameter.
            ee_sd
                SD of correlated Elementary Effects for each parameter.

    Notes
    -----
    `ub` follows http://www.andreasaltelli.eu/file/repository/DGSA_MATCOM_2009.pdf.

    """
    n_inputs = np.size(ee_i, 0)

    if sigma_norm is not False:
        norm = (sd_x / sd_y).reshape(n_inputs, 1)
        ee_i = ee_i * norm
    else:
        pass

    if ub is not False:
        norm = (sd_x ** 2 / sd_y ** 2).reshape(n_inputs, 1)
        ee_i = (ee_i ** 2) * norm
    else:
        pass

    # Init measures.
    ee_mean = np.ones([n_inputs, 1]) * np.nan
    abs_ee_mean = np.ones([n_inputs, 1]) * np.nan
    sd_ee = np.ones([n_inputs, 1]) * np.nan

    # Compute the aggregate screening measures.
    ee_mean[:, 0] = np.mean(ee_i, axis=1)
    abs_ee_mean[:, 0] = np.mean(abs(ee_i), axis=1)
    sd_ee[:, 0] = np.sqrt(np.var(ee_i, axis=1))

    return ee_mean, abs_ee_mean, sd_ee
