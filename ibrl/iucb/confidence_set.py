"""Confidence set for IUCB on 2x2 games.

The confidence set C is maintained as the intersection of:
- The initial hypothesis set H = [-1, 1]^4 (valid payoff matrices)
- Slab constraints from completed cycles: {theta : dist(theta, V_k) <= r_k}

All operations are in the 4D payoff space, with distances computed via
the full Z_bar algebra.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .algebra import (
    theta_from_payoff,
    compute_V,
    dist_to_V,
    DIM_Z,
)
from ..utils.game_solving import solve_2x2_game


# The 16 corners of [-1,1]^4
_BOX_CORNERS = np.array([[(-1) ** ((i >> j) & 1) for j in range(4)] for i in range(16)],
                        dtype=np.float64)


class SlabConstraint:
    """A slab constraint from one completed IUCB cycle."""
    __slots__ = ("x_arm", "y_bar", "V_basis", "radius")

    def __init__(self, x_arm, y_bar, V_basis, radius):
        self.x_arm = x_arm
        self.y_bar = y_bar
        self.V_basis = V_basis
        self.radius = radius


class ConfidenceSet:
    """Maintains the IUCB confidence set for 2x2 games."""

    def __init__(self):
        self.slabs: list[SlabConstraint] = []
        self.bounds = [(-1.0, 1.0)] * 4

    def _penalty(self, theta: NDArray[np.float64]) -> float:
        """Compute penalty for slab constraint violations."""
        penalty = 0.0
        for slab in self.slabs:
            d = dist_to_V(theta, slab.V_basis, slab.x_arm)
            if d > slab.radius:
                penalty += 1000.0 * (d - slab.radius) ** 2
        return penalty

    def _is_feasible(self, P_flat: NDArray[np.float64]) -> bool:
        """Check if a point satisfies all slab constraints."""
        theta = theta_from_payoff(P_flat.reshape(2, 2))
        for slab in self.slabs:
            if dist_to_V(theta, slab.V_basis, slab.x_arm) > slab.radius + 1e-6:
                return False
        return True

    def contains(self, P: NDArray[np.float64]) -> bool:
        P_flat = P.ravel()
        for i in range(4):
            if P_flat[i] < -1.0 - 1e-8 or P_flat[i] > 1.0 + 1e-8:
                return False
        return self._is_feasible(P_flat)

    def max_dist_to_V(self, V_basis: NDArray[np.float64],
                      x_arm: NDArray[np.float64]) -> float:
        """Compute max_{theta in C} dist(theta, V) in Z_bar norm.

        Uses corner evaluation + local refinement for speed.
        """
        best_dist = 0.0

        # Evaluate at box corners (fast screening)
        for corner in _BOX_CORNERS:
            P = corner.reshape(2, 2)
            theta = theta_from_payoff(P)
            d = dist_to_V(theta, V_basis, x_arm)
            # Check feasibility
            if self._is_feasible(corner):
                best_dist = max(best_dist, d)

        # If no slabs yet, also try local refinement from best corner
        if len(self.slabs) == 0:
            # Simple: the full box corners give a good approximation
            return best_dist

        # Local optimization from best feasible point
        best_P = None
        best_val = 0.0
        for corner in _BOX_CORNERS:
            if self._is_feasible(corner):
                P = corner.reshape(2, 2)
                theta = theta_from_payoff(P)
                d = dist_to_V(theta, V_basis, x_arm)
                if d > best_val:
                    best_val = d
                    best_P = corner.copy()

        if best_P is not None:
            def neg_dist(P_flat):
                P = np.clip(P_flat, -1, 1).reshape(2, 2)
                theta = theta_from_payoff(P)
                d = dist_to_V(theta, V_basis, x_arm)
                return -d + self._penalty(theta)

            result = minimize(neg_dist, best_P, method='L-BFGS-B',
                            bounds=self.bounds, options={'maxiter': 50})
            if -result.fun > best_dist:
                best_dist = -result.fun + self._penalty(theta_from_payoff(result.x.reshape(2,2)))
                # Only use if feasible
                if self._is_feasible(result.x):
                    best_dist = max(best_dist, dist_to_V(
                        theta_from_payoff(result.x.reshape(2,2)), V_basis, x_arm))

        return best_dist

    def update(self, x_arm: NDArray[np.float64], y_bar: NDArray[np.float64],
               radius: float) -> None:
        V_basis = compute_V(x_arm, y_bar)
        self.slabs.append(SlabConstraint(x_arm, y_bar, V_basis, radius))

    def optimistic_theta(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Find argmax_{P in C} max_x ME_P[r|x] and the corresponding policy."""
        best_value = -np.inf
        best_P = None

        # Screen corners
        for corner in _BOX_CORNERS:
            if not self._is_feasible(corner):
                continue
            P = corner.reshape(2, 2)
            v, _ = solve_2x2_game(P)
            if v > best_value:
                best_value = v
                best_P = corner.copy()

        # Local refinement
        if best_P is not None:
            def neg_value(P_flat):
                P = np.clip(P_flat, -1, 1).reshape(2, 2)
                v, _ = solve_2x2_game(P)
                theta = theta_from_payoff(P)
                return -v + self._penalty(theta)

            result = minimize(neg_value, best_P, method='L-BFGS-B',
                            bounds=self.bounds, options={'maxiter': 100})
            P_candidate = np.clip(result.x, -1, 1).reshape(2, 2)
            if self._is_feasible(result.x):
                v, _ = solve_2x2_game(P_candidate)
                if v > best_value:
                    best_value = v
                    best_P = result.x.copy()

        if best_P is None:
            # Fallback: use center of box
            best_P = np.zeros(4)

        P_opt = np.clip(best_P, -1, 1).reshape(2, 2)
        _, x_opt = solve_2x2_game(P_opt)
        return P_opt, x_opt
