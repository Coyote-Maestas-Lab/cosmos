"""
`adjustText.adjust_text` uses print statements to track random shifts.
We silence this by redirecting stdout to null.
"""

import os
import sys

from adjustText import adjust_text as _adjust_text


class NullWriter:
    """
    Redirects stdout to null.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def adjust_text(*args, **kwargs):
    """
    `adjustText.adjust_text`, but silencing print statements
    """
    with NullWriter():
        return _adjust_text(*args, **kwargs)
