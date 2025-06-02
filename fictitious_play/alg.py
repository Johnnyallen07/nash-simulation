import numpy as np
import matplotlib.pyplot as plt

def fictitious_play(payoff_matrix, num_iterations=1000, noise=0.0, seed=None):
    """
    Run fictitious play for a 2-player normal-form game.
    payoff_matrix: shape (n_actions, n_actions), payoffs for player 1.
    num_iterations: int, number of rounds.
    noise: float, probability of random choice instead of best response.
    Returns:
        p1_freqs, p2_freqs: shape (num_iterations, n_actions)
    """
    rng = np.random.default_rng(seed)
    n_actions = payoff_matrix.shape[0]
    p1_history = [rng.integers(n_actions)]
    p2_history = [rng.integers(n_actions)]

    p1_freqs = np.zeros((num_iterations, n_actions))
    p2_freqs = np.zeros((num_iterations, n_actions))

    for t in range(1, num_iterations):
        # Empirical frequencies
        p2_counts = np.bincount(p2_history, minlength=n_actions) / len(p2_history)
        p1_counts = np.bincount(p1_history, minlength=n_actions) / len(p1_history)

        # Best responses
        p1_payoff = payoff_matrix @ p2_counts
        p2_payoff = -payoff_matrix.T @ p1_counts

        # With probability 'noise', pick randomly; otherwise best response
        if rng.random() < noise:
            p1_action = rng.integers(n_actions)
        else:
            # Best-response: choice one of the best randomly
            max_indices = np.flatnonzero(p1_payoff == p1_payoff.max())
            p1_action = rng.choice(max_indices)
        if rng.random() < noise:
            p2_action = rng.integers(n_actions)
        else:
            max_indices = np.flatnonzero(p2_payoff == p2_payoff.max())
            p2_action = rng.choice(max_indices)

        p1_history.append(p1_action)
        p2_history.append(p2_action)

        p1_freqs[t] = np.bincount(p1_history, minlength=n_actions) / len(p1_history)
        p2_freqs[t] = np.bincount(p2_history, minlength=n_actions) / len(p2_history)

    return p1_freqs, p2_freqs


def plot_fictitious_play(
    payoff_matrix,
    num_iterations=1000,
    noise=0.0,
    repeats=10,
    actions=None,
    nash_eq=None,
    show_legend=True
):
    """
    Plot fictitious play empirical frequencies for a general 2-player game.

    Args:
        payoff_matrix: np.ndarray, shape (n_actions, n_actions), player 1's payoff matrix
        num_iterations: int, number of rounds per run
        noise: float, tremble probability
        repeats: int, number of simulation runs to average
        actions: list[str], names for actions; if None, uses 'Action 0', etc.
        nash_eq: list[float] or np.ndarray, Nash eq distribution for reference line (optional)
        show_legend: bool, whether to show the legend
    """
    n_actions = payoff_matrix.shape[0]
    if actions is None:
        actions = [f"Action {i}" for i in range(n_actions)]

    p1_freqs_accum = np.zeros((num_iterations, n_actions))
    p2_freqs_accum = np.zeros((num_iterations, n_actions))

    for r in range(repeats):
        p1_freqs, p2_freqs = fictitious_play(
            payoff_matrix, num_iterations=num_iterations, noise=noise, seed=r+42)
        p1_freqs_accum += p1_freqs
        p2_freqs_accum += p2_freqs

    # Average over runs
    p1_freqs_mean = p1_freqs_accum / repeats
    p2_freqs_mean = p2_freqs_accum / repeats

    plt.figure(figsize=(14, 7))
    for i in range(n_actions):
        plt.plot(p1_freqs_mean[:, i], label=f'P1 {actions[i]}')
        plt.plot(p2_freqs_mean[:, i], '--', label=f'P2 {actions[i]}')
    # Plot Nash equilibrium reference line(s) if given
    if nash_eq is not None:
        for i in range(n_actions):
            plt.axhline(nash_eq[i], color='gray', linestyle=':', linewidth=1)

    plt.xlabel('Iteration')
    plt.ylabel('Empirical Frequency')
    plt.title(f'Fictitious Play (Noise={noise}, Repeats={repeats})')
    if show_legend:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Usage
payoff_matrix = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0]
])
p1_freqs, p2_freqs = fictitious_play(
            payoff_matrix, num_iterations=1000)

print("The probability distribution of player 1 after 1000 iterations: ", p1_freqs[-1])
print("The probability distribution of player 2 after 1000 iterations: ", p2_freqs[-1])
# plot_fictitious_play(payoff_matrix, num_iterations=1000, noise=0, repeats=1)
