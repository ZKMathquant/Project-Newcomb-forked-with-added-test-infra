"""Helper functions for infrabayesian module — direct port from coin-learning.ipynb."""
from enum import Enum


class Coin(Enum):
    H = 0
    T = 1


def match(pattern, history):
    """Check if pattern matches a given history.

    A pattern can be a tuple (element-wise match, None as wildcard)
    or a list of tuples (match if any element matches).
    """
    if isinstance(pattern, list):
        for p in pattern:
            if match(p, history):
                return True
        return False

    for obs_pattern, obs_history in zip(pattern, history):
        if obs_pattern is not None and obs_pattern != obs_history:
            return False
    return True


def glue(func1, pattern, func2):
    """Glue operator: func1 on pattern, func2 elsewhere."""
    def glued(history):
        return func1(history) if match(pattern, history) else func2(history)
    return glued
