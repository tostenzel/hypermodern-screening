import numpy as np

from hypermodern_screening.sampling_schemes import radial_sample
from hypermodern_screening.screening_measures import screening_measures

"""Example from Ge/Menendez (2017)"""
def linear_function(a, b, c, *args):
    return a + b + c


def main():

    mu = np.array([0, 0, 0])

    cov = np.array(
        [
	    [1.0, 0.9, 0.4],
	    [0.9, 1.0, 0.01],
	    [0.4, 0.01, 1.0],
        ]
    )
    n_inputs = 3
    n_sample = 10_0

    # Trajectory-specific paramters.
    seed = 2020
    n_levels = 10
    n_inputs = 3


    traj_list, step_list = radial_sample(n_sample, n_inputs, normal=True)

    measures_list = screening_measures(linear_function, traj_list, step_list, cov, mu, radial=True)

    ee_uncorr = measures_list[0]
    ee_corr = measures_list[1]
    abs_ee_uncorr = measures_list[2]
    abs_ee_corr = measures_list[3]
    sd_ee_uncorr = measures_list[4]
    sd_ee_corr = measures_list[5]

    print(ee_uncorr)
