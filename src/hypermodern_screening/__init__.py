"""
Entry point for the hypermodern screening package.

This module imports all functions to make them easily callable for the end user.

"""

from hypermodern_screening.sampling_schemes import *
from hypermodern_screening.screening_measures import *
from hypermodern_screening.select_sample_set import *
from hypermodern_screening.transform_distributions import *
from hypermodern_screening.transform_ee import *
from hypermodern_screening.transform_reorder import *


# Automate package version update.
try:
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
