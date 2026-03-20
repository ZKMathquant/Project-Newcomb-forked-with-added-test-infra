"""Tests for IUCB algebra (Example 4.3 of Kosoy's thesis, |B1|=|B2|=2)."""
import numpy as np
import pytest

from ibrl.iucb.algebra import (
    theta_from_payoff,
    payoff_from_theta,
    F_eval,
    F_bar_eval,
    F_bar_matrix,
    compute_V,
    dist_to_V,
    z_bar_norm,
    outcome_to_y,
    lower_prevision,
    mu,
    DIM_Z,
    DIM_W,
    DIM_Y,
    DIM_Z_BAR,
)


class TestDimensions:
    def test_dimensions(self):
        assert DIM_Y == 8
        assert DIM_Z == 10
        assert DIM_W == 6
        assert DIM_Z_BAR == 16


class TestPayoffConversion:
    def test_roundtrip(self):
        P = np.array([[0.5, -0.3], [0.2, 0.8]])
        theta = theta_from_payoff(P)
        P2 = payoff_from_theta(theta)
        np.testing.assert_array_almost_equal(P, P2)

    def test_theta_satisfies_constraints(self):
        """theta_b = 1 and theta_{ba,0} + theta_{ba,1} = 1."""
        P = np.array([[0.5, -0.3], [0.2, 0.8]])
        theta = theta_from_payoff(P)
        # z_b = 1
        assert abs(theta[0] - 1.0) < 1e-10
        assert abs(theta[1] - 1.0) < 1e-10
        # z_{ba,0} + z_{ba,1} = 1
        for b in range(2):
            for a in range(2):
                idx0 = 2 + 4 * b + 2 * a
                assert abs(theta[idx0] + theta[idx0 + 1] - 1.0) < 1e-10

    def test_theta_nonnegative_components(self):
        """For P in [-1,1], theta components should be in [0,1]."""
        P = np.array([[1.0, -1.0], [-1.0, 1.0]])
        theta = theta_from_payoff(P)
        assert np.all(theta >= -1e-10)

    def test_extreme_payoffs(self):
        """Test with boundary payoff values."""
        P = np.array([[1.0, 1.0], [1.0, 1.0]])
        theta = theta_from_payoff(P)
        P2 = payoff_from_theta(theta)
        np.testing.assert_array_almost_equal(P, P2)

        P = np.array([[-1.0, -1.0], [-1.0, -1.0]])
        theta = theta_from_payoff(P)
        P2 = payoff_from_theta(theta)
        np.testing.assert_array_almost_equal(P, P2)


class TestBilinearMapF:
    def test_F_zero_at_true_hypothesis(self):
        """F(x, theta*, E[y]) = 0 for the true hypothesis (Assumption 3)."""
        P = np.array([[0.5, -0.3], [0.2, 0.8]])
        theta = theta_from_payoff(P)

        for x0 in [0.0, 0.3, 0.5, 0.7, 1.0]:
            x = np.array([x0, 1 - x0])
            # E[y[b,a,s]] = x[b] * x[a] * theta_{ba,s}
            Ey = np.zeros(DIM_Y)
            for b in range(2):
                for a in range(2):
                    Ey[4 * b + 2 * a + 0] = x[b] * x[a] * (1 - P[b, a]) / 2
                    Ey[4 * b + 2 * a + 1] = x[b] * x[a] * (1 + P[b, a]) / 2

            w = F_eval(x, theta, Ey)
            np.testing.assert_array_almost_equal(w, 0, decimal=10,
                err_msg=f"F != 0 at x={x}")

    def test_F_bilinear_in_z(self):
        """F(x, alpha*z1 + beta*z2, y) = alpha*F(x,z1,y) + beta*F(x,z2,y)."""
        x = np.array([0.6, 0.4])
        y = np.random.randn(DIM_Y)
        z1 = np.random.randn(DIM_Z)
        z2 = np.random.randn(DIM_Z)
        alpha, beta = 0.3, 0.7

        w_combined = F_eval(x, alpha * z1 + beta * z2, y)
        w_separate = alpha * F_eval(x, z1, y) + beta * F_eval(x, z2, y)
        np.testing.assert_array_almost_equal(w_combined, w_separate)

    def test_F_bilinear_in_y(self):
        """F(x, z, alpha*y1 + beta*y2) = alpha*F(x,z,y1) + beta*F(x,z,y2)."""
        x = np.array([0.6, 0.4])
        z = np.random.randn(DIM_Z)
        y1 = np.random.randn(DIM_Y)
        y2 = np.random.randn(DIM_Y)
        alpha, beta = 0.3, 0.7

        w_combined = F_eval(x, z, alpha * y1 + beta * y2)
        w_separate = alpha * F_eval(x, z, y1) + beta * F_eval(x, z, y2)
        np.testing.assert_array_almost_equal(w_combined, w_separate)


class TestFBar:
    def test_F_bar_extends_F(self):
        """F_bar(x, [z; 0], y) = F(x, z, y)."""
        x = np.array([0.6, 0.4])
        z = np.random.randn(DIM_Z)
        y = np.random.randn(DIM_Y)

        z_bar = np.zeros(DIM_Z_BAR)
        z_bar[:DIM_Z] = z
        w_bar = F_bar_eval(x, z_bar, y)
        w_f = F_eval(x, z, y)
        np.testing.assert_array_almost_equal(w_bar, w_f)

    def test_F_bar_w_component(self):
        """F_bar(x, [0; w], y) = mu(y) * w."""
        x = np.array([0.6, 0.4])
        w = np.random.randn(DIM_W)
        y = np.random.randn(DIM_Y)

        z_bar = np.zeros(DIM_Z_BAR)
        z_bar[DIM_Z:] = w
        result = F_bar_eval(x, z_bar, y)
        expected = mu(y) * w
        np.testing.assert_array_almost_equal(result, expected)

    def test_F_bar_matrix_correct(self):
        """F_bar_matrix(x, y) @ z_bar == F_bar_eval(x, z_bar, y)."""
        x = np.array([0.6, 0.4])
        y = np.random.randn(DIM_Y)
        z_bar = np.random.randn(DIM_Z_BAR)

        M = F_bar_matrix(x, y)
        result_matrix = M @ z_bar
        result_eval = F_bar_eval(x, z_bar, y)
        np.testing.assert_array_almost_equal(result_matrix, result_eval)


class TestKernel:
    def test_true_theta_in_kernel(self):
        """The true theta should be in V(x, E[y]) for the correct E[y]."""
        P = np.array([[0.5, -0.3], [0.2, 0.8]])
        theta = theta_from_payoff(P)
        x = np.array([0.6, 0.4])

        # Compute E[y]
        Ey = np.zeros(DIM_Y)
        for b in range(2):
            for a in range(2):
                Ey[4 * b + 2 * a + 0] = x[b] * x[a] * (1 - P[b, a]) / 2
                Ey[4 * b + 2 * a + 1] = x[b] * x[a] * (1 + P[b, a]) / 2

        V_basis = compute_V(x, Ey)

        # theta_bar = [theta; 0] should be in the kernel (or very close)
        theta_bar = np.zeros(DIM_Z_BAR)
        theta_bar[:DIM_Z] = theta
        # Check F_bar(x, theta_bar, Ey) = 0
        M = F_bar_matrix(x, Ey)
        residual = M @ theta_bar
        np.testing.assert_array_almost_equal(residual, 0, decimal=8)

    def test_kernel_dimension(self):
        """The kernel of F_bar_{x,y} should have dimension >= DIM_Z_BAR - DIM_W."""
        x = np.array([0.6, 0.4])
        y = np.random.randn(DIM_Y)
        V_basis = compute_V(x, y)
        # Kernel dimension should be at least DIM_Z_BAR - DIM_W = 10
        assert V_basis.shape[0] >= DIM_Z_BAR - DIM_W


class TestOutcomeEncoding:
    def test_outcome_is_valid_distribution(self):
        """Encoded outcome should be in Delta(B) (nonneg, sums to 1)."""
        y = outcome_to_y(0, 1, 0.5)
        assert np.all(y >= -1e-10)
        assert abs(y.sum() - 1.0) < 1e-10

    def test_outcome_support(self):
        """Only the (b, a) entry should be nonzero."""
        y = outcome_to_y(1, 0, 0.3)
        # Only y[1, 0, 0] and y[1, 0, 1] should be nonzero
        for b in range(2):
            for a in range(2):
                for s in range(2):
                    idx = 4 * b + 2 * a + s
                    if b == 1 and a == 0:
                        assert y[idx] > 0
                    else:
                        assert y[idx] == 0.0

    def test_outcome_reward_encoding(self):
        """Reward r should be encoded as (1-r)/2 and (1+r)/2."""
        r = 0.6
        y = outcome_to_y(0, 0, r)
        assert abs(y[0] - (1 - r) / 2) < 1e-10  # y[0,0,0] = (1-r)/2
        assert abs(y[1] - (1 + r) / 2) < 1e-10  # y[0,0,1] = (1+r)/2

    def test_outcome_extreme_rewards(self):
        """Test with r = -1 and r = +1."""
        y_neg = outcome_to_y(0, 0, -1.0)
        assert abs(y_neg[0] - 1.0) < 1e-10  # (1-(-1))/2 = 1
        assert abs(y_neg[1] - 0.0) < 1e-10  # (1+(-1))/2 = 0

        y_pos = outcome_to_y(0, 0, 1.0)
        assert abs(y_pos[0] - 0.0) < 1e-10
        assert abs(y_pos[1] - 1.0) < 1e-10


class TestNorms:
    def test_z_bar_norm_nonneg(self):
        """The norm should be non-negative."""
        z_bar = np.random.randn(DIM_Z_BAR)
        assert z_bar_norm(z_bar) >= 0

    def test_z_bar_norm_zero_vector(self):
        """The zero vector should have norm zero."""
        assert z_bar_norm(np.zeros(DIM_Z_BAR)) < 1e-10

    def test_z_bar_norm_positive_for_nonzero(self):
        """A nonzero vector should have positive norm."""
        z_bar = np.zeros(DIM_Z_BAR)
        z_bar[3] = 1.0
        assert z_bar_norm(z_bar) > 0

    def test_z_bar_norm_homogeneous(self):
        """||alpha * z|| = |alpha| * ||z||."""
        z_bar = np.random.randn(DIM_Z_BAR)
        alpha = 2.5
        n1 = z_bar_norm(alpha * z_bar)
        n2 = abs(alpha) * z_bar_norm(z_bar)
        assert abs(n1 - n2) < 1e-6 * max(n1, n2, 1e-10)


class TestLowerPrevision:
    def test_lower_prevision_pure_strategy(self):
        """For a pure strategy, lower_prevision = min payoff over predictor."""
        P = np.array([[10.0, 15.0], [0.0, 5.0]])
        # Pure action 0: min(P[0,0], P[1,0]) = min(10, 0) = 0
        assert abs(lower_prevision(P, np.array([1.0, 0.0])) - 0.0) < 1e-10
        # Pure action 1: min(P[0,1], P[1,1]) = min(15, 5) = 5
        assert abs(lower_prevision(P, np.array([0.0, 1.0])) - 5.0) < 1e-10

    def test_lower_prevision_mixed(self):
        """Damascus at p=0.5: min(5, 5) = 5."""
        P = np.array([[0.0, 10.0], [10.0, 0.0]])
        x = np.array([0.5, 0.5])
        assert abs(lower_prevision(P, x) - 5.0) < 1e-10


class TestDistToV:
    def test_zero_distance_for_true_theta(self):
        """The true theta should have distance 0 to V(x, E[y])."""
        P = np.array([[0.5, -0.3], [0.2, 0.8]])
        theta = theta_from_payoff(P)
        x = np.array([0.6, 0.4])

        Ey = np.zeros(DIM_Y)
        for b in range(2):
            for a in range(2):
                Ey[4 * b + 2 * a + 0] = x[b] * x[a] * (1 - P[b, a]) / 2
                Ey[4 * b + 2 * a + 1] = x[b] * x[a] * (1 + P[b, a]) / 2

        V_basis = compute_V(x, Ey)
        d = dist_to_V(theta, V_basis, x)
        assert d < 1e-6, f"Expected distance ~0, got {d}"

    def test_positive_distance_for_wrong_theta(self):
        """A wrong theta should have positive distance to V(x, E[y])."""
        P_true = np.array([[0.5, -0.3], [0.2, 0.8]])
        P_wrong = np.array([[-0.5, 0.3], [-0.2, -0.8]])
        theta_wrong = theta_from_payoff(P_wrong)
        x = np.array([0.6, 0.4])

        Ey = np.zeros(DIM_Y)
        for b in range(2):
            for a in range(2):
                Ey[4 * b + 2 * a + 0] = x[b] * x[a] * (1 - P_true[b, a]) / 2
                Ey[4 * b + 2 * a + 1] = x[b] * x[a] * (1 + P_true[b, a]) / 2

        V_basis = compute_V(x, Ey)
        d = dist_to_V(theta_wrong, V_basis, x)
        assert d > 0.01, f"Expected positive distance, got {d}"
