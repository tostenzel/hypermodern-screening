"""
Computes the screening measures for correlated inputs that I improved upon
[1] by adjusting the step in the denumeroter to the transformed step in the
nominator in order to not violate the definition of the function derivative.

References
----------
[1] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global
sensitivityanalysis of models with dependent inputs. Reliability Engineering &
System Safety 100 (162), 28–39.

"""
import numpy as np
from hypermodern_screening.transform_ee import trans_ee_corr
from hypermodern_screening.transform_ee import trans_ee_uncorr


def screening_measures(function, traj_list, step_list, cov, mu, radial=False):
    """
    Computes screening measures for a set of paramters.

    Parameters
    ----------
    function : function
        Function or Model of which its parameters are subject to screening.
    traj_list : list of ndarrays
        List of transformed trajectories according to [1].
    step_list : list of ndarrays
        List of steps that each parameter takes in each trajectory.
    cov : ndarray
        Covariance matrix of the input parameters.
    mu : ndarray
        Expectation values of the input parameters.
    radial : bool
        Sample is in trajectory or radial design.

    Returns
    -------
    ee_uncorr : ndarray
        Mean uncorrelated Elementary Effect for each parameter.
    ee_corr : ndarray
        Mean correlated Elementary Effect for each parameter.
    abs_ee_uncorr : ndarray
        Mean absolute uncorrelated Elementary Effect for each parameter.
    abs_ee_corr : ndarray
        Mean absolute correlated Elementary Effect for each parameter.
    sd_ee_uncorr : ndarray
        SD of individual uncorrelated Elementary Effects for each parameter.
    sd_ee_corr : ndarray
        SD of individual correlated Elementary Effects for each parameter.

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
            fct_evals_pi_i[row, traj] = function(*trans_pi_i_list[traj][row, :])
            fct_evals_piplusone_i[row, traj] = function(
                *trans_piplusone_i_list[traj][row, :]
            )
            fct_evals_piplusone_iminusone[row, traj] = function(
                *trans_piplusone_iminusone_list[traj][row, :]
            )

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

        # Need to get the samples of first rows in different orders.
        _, pp_one_row_zero = trans_ee_corr(traj_list, cov, mu, radial=True)

        fct_evals_pp_one_row_zero = np.ones([n_rows, n_trajs]) * np.nan

        for traj in range(0, n_trajs):
            for row in range(0, n_rows):
                fct_evals_pp_one_row_zero[row, traj] = function(
                    *pp_one_row_zero[traj][row, :]
                )

        for traj in range(0, n_trajs):
            ee_corr_i[:, traj] = (
                fct_evals_piplusone_iminusone[1 : n_inputs + 1, traj]
                - fct_evals_pp_one_row_zero[0:n_inputs, traj]
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
    ee_uncorr[:, 0] = np.mean(ee_uncorr_i, axis=1)
    abs_ee_uncorr[:, 0] = np.mean(abs(ee_uncorr_i), axis=1)
    sd_ee_uncorr[:, 0] = np.sqrt(np.var(ee_uncorr_i, axis=1))

    ee_corr[:, 0] = np.mean(ee_corr_i, axis=1)
    abs_ee_corr[:, 0] = np.mean(abs(ee_corr_i), axis=1)
    sd_ee_corr[:, 0] = np.sqrt(np.var(ee_corr_i, axis=1))

    return [ee_uncorr, ee_corr, abs_ee_uncorr, abs_ee_corr, sd_ee_uncorr, sd_ee_corr]
