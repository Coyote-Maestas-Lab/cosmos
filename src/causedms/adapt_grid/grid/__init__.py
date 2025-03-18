"""
Adaptive grid approximation for Bayesian posterior approximation, 
  with adaptive boundary extension and hypercube refinement.
"""

# pylint: disable=wrong-import-position

import logging

GRID_TOL = 1e-12
logger = logging.getLogger(__name__)

from .grid_margin import GridMargin  # isort:skip
from .grid import Grid  # isort:skip
