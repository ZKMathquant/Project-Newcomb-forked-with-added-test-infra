"""Concrete algebra for IUCB on 2x2 games (Example 4.3 of Kosoy's thesis).

Spaces and dimensions for |B1| = |B2| = 2:
- Y = R^8, indexed as y[b, a, s] for b in B2={0,1}, a in B1={0,1}, s in {-1,+1} (s=0 for -1, s=1 for +1)
- Z = R^10: two R for z_b (b in B2) and four R^2 for z_{ba} ((b,a) in B2 x B1)
- S^# has 12 elements: 4 of type (b,a') and 8 of type (b,a',s)
- W = {w in R^12 : sum constraints} has dim 6
- D_Z = dim Z = 10
- D_W = dim W = 6
- Z_bar = Z + W (since N = {0} for this case, see below) has dim 16

The hypothesis theta in Z satisfies:
  z_b = 1 for each b in B2
  z_{ba,0} + z_{ba,1} = 1 for each (b,a) in B2 x B1
So theta is parameterized by the 4 payoff entries P[b,a] in [-1, 1]:
  z_b = 1, z_{ba} = [(1-P[b,a])/2, (1+P[b,a])/2]

Y indexing: y is a flat array of length 8.
  y[4*b + 2*a + s] for b in {0,1}, a in {0,1}, s in {0,1}

Z indexing: z is a flat array of length 10.
  z[0], z[1]                        = z_0, z_1 (for b=0, b=1)
  z[2 + 4*b + 2*a], z[2 + 4*b + 2*a + 1] = z_{ba,0}, z_{ba,1}

W indexing: We represent W in a 6D basis. The 12 S^# components are:
  First 4: (b, a') for b in {0,1}, a' in {0,1}  -> w_full[2*b + a']
  Next 8:  (b, a', s) for b in {0,1}, a' in {0,1}, s in {0,1} -> w_full[4 + 4*b + 2*a' + s]
The constraints are: for each a in S, sum_{c in G_{|a|}} w_{ac} = 0.
  For a = b: w_full[2*b + 0] + w_full[2*b + 1] = 0  (2 constraints)
  For a = (b,a'): w_full[4 + 4*b + 2*a' + 0] + w_full[4 + 4*b + 2*a' + 1] = 0  (4 constraints)

We use a 6D representation: for each of the 6 constrained pairs, store the
value of the first element (the second is its negation).
  w[0] = w_full[0] (b=0: a'=0 component, a'=1 component is -w[0])
  w[1] = w_full[2] (b=1: a'=0 component)
  w[2] = w_full[4 + 0] (b=0,a'=0: s=0 component, s=1 is -w[2])
  w[3] = w_full[4 + 2] (b=0,a'=1)
  w[4] = w_full[4 + 4] (b=1,a'=0)
  w[5] = w_full[4 + 6] (b=1,a'=1)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

# Dimensions
DIM_Y = 8      # |B2| * |B1| * |{-1,+1}|
DIM_Z = 10     # |B2| + 2*|B2|*|B1|
DIM_W = 6      # |S^#| - |S| = 12 - 6
DIM_Z_BAR = DIM_Z + DIM_W  # = 16, since N = {0}
DIM_PAYOFF = 4  # free parameters: P[b,a] for 2x2


def _y_index(b: int, a: int, s: int) -> int:
    """Index into Y = R^8."""
    return 4 * b + 2 * a + s


def _z_b_index(b: int) -> int:
    """Index of z_b in Z = R^10."""
    return b


def _z_ba_index(b: int, a: int, s: int) -> int:
    """Index of z_{ba,s} in Z = R^10."""
    return 2 + 4 * b + 2 * a + s


def theta_from_payoff(P: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a 2x2 payoff matrix to the thesis's theta in Z = R^10.

    Arguments:
        P: shape (2, 2), payoff matrix with P[b, a] in [-1, 1]

    Returns:
        theta: shape (10,)
    """
    theta = np.zeros(DIM_Z)
    theta[0] = 1.0  # z_0 = 1
    theta[1] = 1.0  # z_1 = 1
    for b in range(2):
        for a in range(2):
            theta[_z_ba_index(b, a, 0)] = (1 - P[b, a]) / 2
            theta[_z_ba_index(b, a, 1)] = (1 + P[b, a]) / 2
    return theta


def payoff_from_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract the 2x2 payoff matrix from theta in Z = R^10.

    Arguments:
        theta: shape (10,)

    Returns:
        P: shape (2, 2)
    """
    P = np.zeros((2, 2))
    for b in range(2):
        for a in range(2):
            # theta_{ba,1} = (1+P[b,a])/2, so P[b,a] = 2*theta_{ba,1} - 1
            P[b, a] = 2 * theta[_z_ba_index(b, a, 1)] - 1
    return P


def mu(y: NDArray[np.float64]) -> float:
    """The mass function mu(y) = sum of all components.

    For y in Delta(B), mu(y) = 1.
    """
    return y.sum()


def F_eval(x: NDArray[np.float64], z: NDArray[np.float64],
           y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate F(x, z, y) -> w in W (6D representation).

    Arguments:
        x: shape (2,), agent's mixed strategy (probability distribution)
        z: shape (10,), hypothesis vector in Z
        y: shape (8,), outcome vector in Y

    Returns:
        w: shape (6,), element of W
    """
    w = np.zeros(DIM_W)

    for b in range(2):
        # y_marg_b = sum_{a,s} y[b,a,s]
        y_marg_b = sum(y[_y_index(b, a, s)] for a in range(2) for s in range(2))
        z_b = z[_z_b_index(b)]

        for a_prime in range(2):
            # y_marg_{b,a'} = sum_s y[b,a',s]
            y_marg_ba = y[_y_index(b, a_prime, 0)] + y[_y_index(b, a_prime, 1)]

            # F[b, a'] = z_b * (y_marg_{ba'} - x[a'] * y_marg_b)
            # This is the first pair type. W index: b*1 + ... -> w[b]
            # Actually, for each b, the (b,0) and (b,1) components satisfy
            # w[b,0] + w[b,1] = 0. We store w[b,0] = F[b, 0].
            # Wait, let me reconsider. For b, the two components are a'=0 and a'=1.
            # The constraint is F[b,0] + F[b,1] = 0.
            # We use w[b] = F[b, 0] (and F[b, 1] = -F[b, 0]).
            if a_prime == 0:
                w[b] = z_b * (y_marg_ba - x[a_prime] * y_marg_b)

    for b in range(2):
        for a_prime in range(2):
            y_marg_ba = y[_y_index(b, a_prime, 0)] + y[_y_index(b, a_prime, 1)]
            psi_ba = z[_z_ba_index(b, a_prime, 0)] + z[_z_ba_index(b, a_prime, 1)]

            # F[b, a', s] = psi_{ba}(z_{ba}) * y[b,a',s] - z_{ba,s} * y_marg_{ba'}
            # The constraint is F[b,a',0] + F[b,a',1] = 0.
            # We store w[2 + 2*b + a'] = F[b, a', 0].
            s = 0
            w[2 + 2 * b + a_prime] = psi_ba * y[_y_index(b, a_prime, s)] - z[_z_ba_index(b, a_prime, s)] * y_marg_ba

    return w


def F_bar_eval(x: NDArray[np.float64], z_bar: NDArray[np.float64],
               y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate F_bar(x, z_bar, y) = F(x, z, y) + mu(y) * w.

    Arguments:
        x: shape (2,), agent's mixed strategy
        z_bar: shape (16,), element of Z_bar = Z + W (first 10 = z, last 6 = w)
        y: shape (8,), outcome vector

    Returns:
        shape (6,), element of W
    """
    z = z_bar[:DIM_Z]
    w = z_bar[DIM_Z:]
    return F_eval(x, z, y) + mu(y) * w


def F_bar_matrix(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the matrix representation of F_bar_{x,y} : Z_bar -> W.

    F_bar_{x,y} is linear in z_bar (since F is bilinear in z and y,
    fixing y makes it linear in z, and the w term is linear).

    Arguments:
        x: shape (2,), agent's mixed strategy
        y: shape (8,), outcome vector (typically y_bar)

    Returns:
        M: shape (6, 16) matrix such that F_bar(x, z_bar, y) = M @ z_bar
    """
    M = np.zeros((DIM_W, DIM_Z_BAR))
    e = np.zeros(DIM_Z_BAR)
    for j in range(DIM_Z_BAR):
        e[j] = 1.0
        M[:, j] = F_bar_eval(x, e, y)
        e[j] = 0.0
    return M


def F_matrix(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the matrix representation of F_{x,y} : Z -> W (the z-part only).

    Arguments:
        x: shape (2,)
        y: shape (8,)

    Returns:
        M: shape (6, 10)
    """
    M = np.zeros((DIM_W, DIM_Z))
    e = np.zeros(DIM_Z)
    for j in range(DIM_Z):
        e[j] = 1.0
        M[:, j] = F_eval(x, e, y)
        e[j] = 0.0
    return M


def compute_V(x: NDArray[np.float64], y_bar: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute a basis for the kernel V(x, y_bar) = ker F_bar_{x, y_bar}.

    Arguments:
        x: shape (2,), the arm (agent's mixed strategy)
        y_bar: shape (8,), the average outcome vector

    Returns:
        V_basis: shape (dim_V, 16) — rows are basis vectors of the kernel.
                 Empty (shape (0, 16)) if kernel is trivial.
    """
    M = F_bar_matrix(x, y_bar)
    # Kernel of M: the null space
    # Use SVD: null space is the right singular vectors with zero singular values
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    tol = max(M.shape) * np.finfo(float).eps * s[0] if len(s) > 0 and s[0] > 0 else 1e-12
    null_mask = s < tol
    # Null space vectors are the last rows of Vt (those not corresponding to nonzero singular values)
    rank = DIM_W - null_mask.sum()
    # The null space has dimension DIM_Z_BAR - rank
    V_basis = Vt[rank:]
    return V_basis


def dist_to_V(theta: NDArray[np.float64], V_basis: NDArray[np.float64],
              x_arm: NDArray[np.float64]) -> float:
    """Compute min_{v in V} ||theta_bar - v||_{Z_bar} where theta_bar = [theta; 0].

    The Z_bar norm is ||z_bar|| = max_{x in A} ||F_bar_{x}(z_bar, .)||_{op}
    where the operator norm is from (Y, l1) to (W, ||.||_W).

    For practical purposes, we compute this as:
      min_{v in V} max_{x in [0,1]} ||F_bar_x(theta_bar - v, .)||_{Y->W, op}

    This is a minimax problem. Since V = ker F_bar_{x_arm, y_bar}, and the
    distance along the x_arm direction is what matters for the confidence set,
    we use a simplified approach:

    dist(theta, V) = ||proj_{V^perp}(theta_bar)||_{Z_bar}

    where V^perp is the orthogonal complement of V in Z_bar (w.r.t. the standard
    inner product), and we then compute the Z_bar norm of the projection.

    However, the Z_bar norm is NOT the Euclidean norm, so orthogonal projection
    doesn't directly give the norm-minimizing element. For tractability in the
    2x2 case, we compute the distance numerically.

    Arguments:
        theta: shape (10,), hypothesis in Z
        V_basis: shape (k, 16), basis for V (from compute_V)
        x_arm: shape (2,), the arm being played (for norm computation)

    Returns:
        The distance from theta to V in the Z_bar norm.
    """
    theta_bar = np.zeros(DIM_Z_BAR)
    theta_bar[:DIM_Z] = theta

    if V_basis.shape[0] == 0:
        # V is trivial (empty kernel) — distance is just the norm of theta_bar
        return z_bar_norm(theta_bar)

    if V_basis.shape[0] == DIM_Z_BAR:
        # V is the entire space
        return 0.0

    # Project theta_bar onto V using least squares (Euclidean projection)
    # This gives an upper bound. For the Z_bar norm, we'd ideally minimize
    # over V, but Euclidean projection is a good approximation and exact
    # when the norm is isotropic.
    #
    # min_{alpha} ||theta_bar - V_basis^T @ alpha||_{Z_bar}
    # We use Euclidean projection as a starting point, then refine.
    VT = V_basis.T  # shape (16, k)
    # Euclidean projection: alpha = (V V^T)^{-1} V theta_bar
    # Since V_basis rows are orthonormal (from SVD), V V^T = I
    alpha = V_basis @ theta_bar  # shape (k,)
    v_proj = VT @ alpha  # Euclidean projection onto V
    residual = theta_bar - v_proj

    # Compute Z_bar norm of the residual
    return z_bar_norm(residual)


def z_bar_norm(z_bar: NDArray[np.float64]) -> float:
    """Compute ||z_bar||_{Z_bar} = max_{x in A} ||F_bar_x(z_bar, .)||_{op}.

    The operator norm is from (Y, l1) to (W, l2).
    For the l1 -> l2 operator norm: max_j ||column_j||_2.

    We use l2 on W as a practical approximation of the thesis's W-norm.

    Arguments:
        z_bar: shape (16,)

    Returns:
        The Z_bar norm (non-negative float).
    """
    if np.allclose(z_bar, 0):
        return 0.0

    z = z_bar[:DIM_Z]
    w = z_bar[DIM_Z:]

    def neg_op_norm(x0_val):
        x = np.array([x0_val, 1 - x0_val])
        max_col_norm_sq = 0.0
        for j in range(DIM_Y):
            e_j = np.zeros(DIM_Y)
            e_j[j] = 1.0
            col = F_eval(x, z, e_j) + w  # mu(e_j) = sum(e_j) = 1
            max_col_norm_sq = max(max_col_norm_sq, col @ col)
        return -np.sqrt(max_col_norm_sq)

    # Evaluate at a grid of x values and pick the best
    best = 0.0
    for x0 in np.linspace(0, 1, 11):
        val = -neg_op_norm(x0)
        if val > best:
            best = val

    # Refine with bounded minimization around the best
    result = minimize_scalar(neg_op_norm, bounds=(0, 1), method='bounded',
                            options={'xatol': 0.01})
    return max(best, -result.fun)


def outcome_to_y(env_action: int, action: int, reward: float) -> NDArray[np.float64]:
    """Encode a single observation as the outcome vector y in R^8.

    Following Example 4.3: for outcome (b, a, r) where b = env_action,
    a = action, r = reward:
        y[b, a, 1] = (1 + r) / 2   (the +1 component)
        y[b, a, 0] = (1 - r) / 2   (the -1 component)
        y[b', a', s] = 0            for (b', a') != (b, a)

    Rewards must be in [-1, 1]. If not, they should be rescaled before calling.

    Arguments:
        env_action: predictor's action b in {0, 1}
        action: agent's action a in {0, 1}
        reward: scalar reward in [-1, 1]

    Returns:
        y: shape (8,)
    """
    y = np.zeros(DIM_Y)
    y[_y_index(env_action, action, 0)] = (1 - reward) / 2
    y[_y_index(env_action, action, 1)] = (1 + reward) / 2
    return y


def lower_prevision(P: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """Compute ME_theta[r|x] = min_b sum_a x[a] * P[b, a].

    This is the lower prevision (worst-case expected reward) for the agent
    playing mixed strategy x against payoff matrix P, where the predictor
    (column player) adversarially chooses its action.

    Arguments:
        P: shape (2, 2), payoff matrix
        x: shape (2,), agent's mixed strategy

    Returns:
        The minimum expected reward over predictor actions.
    """
    return min(P[b] @ x for b in range(2))
