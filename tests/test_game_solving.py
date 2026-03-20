"""Tests for 2x2 game solving utilities."""
import numpy as np
import pytest

from ibrl.utils.game_solving import solve_2x2_game


class TestSolve2x2Game:
    """Test the 2x2 zero-sum game solver."""

    def test_pure_strategy_dominant(self):
        """When one action dominates, the solution is a pure strategy."""
        # Action 0 dominates: P[b,0] > P[b,1] for all b
        P = np.array([[5.0, 1.0], [3.0, 2.0]])
        value, x = solve_2x2_game(P)
        assert value == 3.0
        np.testing.assert_array_almost_equal(x, [1.0, 0.0])

    def test_pure_strategy_action1(self):
        """When action 1 is better in the minimax sense."""
        P = np.array([[1.0, 5.0], [2.0, 3.0]])
        value, x = solve_2x2_game(P)
        assert value == 3.0
        np.testing.assert_array_almost_equal(x, [0.0, 1.0])

    def test_damascus_mixed_equilibrium(self):
        """Damascus: anti-diagonal payoffs require mixed strategy p=0.5."""
        P = np.array([[0.0, 10.0], [10.0, 0.0]])
        value, x = solve_2x2_game(P)
        assert abs(value - 5.0) < 1e-10
        np.testing.assert_array_almost_equal(x, [0.5, 0.5])

    def test_asymmetric_damascus(self):
        """Asymmetric Damascus: death=0, death_aleppo=5, life=10."""
        P = np.array([[0.0, 10.0], [10.0, 5.0]])
        value, x = solve_2x2_game(P)
        # Mixed equilibrium: x[0]*(0) + x[1]*10 = x[0]*10 + x[1]*5
        # 10*x[1] = 10*x[0] + 5*(1-x[0])  =>  10*(1-x[0]) = 5 + 5*x[0]
        # 10 - 10*x[0] = 5 + 5*x[0]  =>  5 = 15*x[0]  =>  x[0] = 1/3
        assert abs(x[0] - 1.0 / 3.0) < 1e-10
        expected_value = 10.0 * (1.0 - 1.0 / 3.0)  # = 20/3
        assert abs(value - expected_value) < 1e-10

    def test_newcomb_two_box_dominates(self):
        """In Newcomb, the minimax strategy is two-boxing.

        P[b,a]: b=predictor, a=agent. Two-boxing (a=1) gives higher
        minimum over predictor actions.
        """
        P = np.array([[10.0, 15.0], [0.0, 5.0]])
        value, x = solve_2x2_game(P)
        assert abs(value - 5.0) < 1e-10
        np.testing.assert_array_almost_equal(x, [0.0, 1.0])

    def test_coordination_pure_equilibrium(self):
        """Coordination game: diagonal payoffs, off-diagonal zeros."""
        P = np.array([[2.0, 0.0], [0.0, 1.0]])
        value, x = solve_2x2_game(P)
        # Minimax: max_x min(2*x[0], 1-x[0]+x[0]*0) = max_x min(2*x[0], x[1])
        # At equilibrium: 2*x[0] = x[1] = 1-x[0], so x[0] = 1/3
        # Value = 2/3
        assert abs(x[0] - 1.0 / 3.0) < 1e-10
        assert abs(value - 2.0 / 3.0) < 1e-10

    def test_equal_payoffs(self):
        """When all payoffs are equal, any strategy is optimal."""
        P = np.array([[3.0, 3.0], [3.0, 3.0]])
        value, x = solve_2x2_game(P)
        assert abs(value - 3.0) < 1e-10
        assert abs(x.sum() - 1.0) < 1e-10

    def test_strategies_are_valid_distributions(self):
        """Strategy should be a valid probability distribution."""
        for _ in range(20):
            P = np.random.randn(2, 2)
            value, x = solve_2x2_game(P)
            assert x.shape == (2,)
            assert np.all(x >= -1e-10)
            assert abs(x.sum() - 1.0) < 1e-10
