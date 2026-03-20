import numpy as np
from numpy.typing import NDArray


def solve_2x2_game(P: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    """Solve a 2x2 zero-sum game for the row player (maximizer).

    The row player chooses a mixed strategy x in Delta({0,1}) to maximize
    min_b (x^T P)_b, where P is the 2x2 payoff matrix with P[b,a] being
    the payoff when column plays b and row plays a.

    For IUCB: the "row player" is the agent choosing mixed strategy x,
    and the "column player" is the environment/predictor choosing prediction b.
    The agent wants to maximize the minimum expected payoff over predictor actions:
        max_x min_b sum_a x[a] * P[b, a]

    Arguments:
        P: 2x2 payoff matrix, P[b, a] = payoff for predictor action b, agent action a

    Returns:
        (value, x_opt): game value and optimal row player (agent) mixed strategy
    """
    a, b = P[0, 0], P[0, 1]  # row 0: predictor plays 0
    c, d = P[1, 0], P[1, 1]  # row 1: predictor plays 1

    # Pure strategy values
    # x = [1, 0] (action 0): min(P[0,0], P[1,0]) = min(a, c)
    # x = [0, 1] (action 1): min(P[0,1], P[1,1]) = min(b, d)
    v_pure0 = min(a, c)
    v_pure1 = min(b, d)

    # Check for mixed equilibrium:
    # x[0] * P[0,0] + x[1] * P[0,1] = x[0] * P[1,0] + x[1] * P[1,1]
    # x[0] * (a - c) = x[1] * (d - b) = (1 - x[0]) * (d - b)
    # x[0] = (d - b) / (a - c + d - b)
    denom = a - c + d - b  # = (a - b) - (c - d) = a + d - b - c
    if abs(denom) > 1e-12:
        p0 = (d - b) / denom
        if 0 < p0 < 1:
            # Valid mixed equilibrium
            v_mixed = p0 * a + (1 - p0) * b  # = p0 * c + (1 - p0) * d
            if v_mixed > max(v_pure0, v_pure1):
                return v_mixed, np.array([p0, 1 - p0])

    # Pure strategy is optimal
    if v_pure0 >= v_pure1:
        return v_pure0, np.array([1.0, 0.0])
    else:
        return v_pure1, np.array([0.0, 1.0])
