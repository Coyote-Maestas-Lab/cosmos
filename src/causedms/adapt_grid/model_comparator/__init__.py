"""
Compare models using the parameter samples generated from the posterior.
"""

# pylint: disable=wrong-import-position

import logging

logger = logging.getLogger(__name__)

from .elpd_pairwise import ElpdPairwise

__all__ = ["ElpdPairwise"]
