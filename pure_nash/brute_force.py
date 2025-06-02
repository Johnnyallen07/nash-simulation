import numpy as np


def has_pure_nash_equilibrium(A, B):
    """
    Check if the bimatrix game (A, B) has a pure Nash equilibrium.

    Inputs:
      - A: np.ndarray, shape (n, m): payoff matrix for Row player.
      - B: np.ndarray, shape (n, m): payoff matrix for Column player.

    Returns:
      - (exists, equilibria_list)
        • exists: bool, True if at least one pure‐NE exists
        • equilibria_list: list of (i, j) pairs that are pure‐NEs
    """
    n, m = A.shape
    equilibria = []

    for i in range(n):
        for j in range(m):
            # 1) Row player's best‐response condition at column j
            if A[i, j] < np.max(A[:, j]):
                continue
            # 2) Column player's best‐response condition at row i
            if B[i, j] < np.max(B[i, :]):
                continue
            # If both hold, (i, j) is a pure NE
            equilibria.append((i, j))

    return len(equilibria) > 0, equilibria


# unique
A = np.array([[3, 0], [4, 1]], dtype=float)
B = np.array([[3, 4], [0, 1]], dtype=float)

# does not exist
A = np.array([[1, -1], [-1, 1]], dtype=float)
B = np.array([[-1, 1], [1, -1]], dtype=float)

# # multiple
A = np.array([[1, 0], [0, 2]], dtype=float)
B = np.array([[2, 0], [0, 1]], dtype=float)

pure_exists, pure_eqs = has_pure_nash_equilibrium(A, B)
print(f"Pure NE exists: {pure_exists}; Pure NE choices (i, j): {pure_eqs}")