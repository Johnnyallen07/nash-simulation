import numpy as np
import matplotlib.pyplot as plt

'''
p{i}_count: the historical probability distribution of player i (current round)
p{i}_freqs: the historical probability distribution of player i (after round)
0, 1, 2 printing in console represents Rock, Paper, Scissors action
'''


# Payoff matrix for player 1 (row player)
# Rows: Rock, Paper, Scissors
# Columns: Rock, Paper, Scissors
payoff_matrix = np.array([
    [0, -1, 1],  # Rock
    [1, 0, -1],  # Paper
    [-1, 1, 0]   # Scissors
])

num_actions = 3
num_iterations = 500

# Initialize play history
p1_history = []
p2_history = []

# Initialize with random moves
p1_history.append(np.random.choice(num_actions))
p2_history.append(np.random.choice(num_actions))

# Store empirical frequencies
p1_freqs = np.zeros((num_iterations, num_actions))
p2_freqs = np.zeros((num_iterations, num_actions))

for t in range(1, num_iterations):
    # Count opponent's actions
    p2_counts = np.bincount(p2_history, minlength=num_actions) / len(p2_history)
    p1_counts = np.bincount(p1_history, minlength=num_actions) / len(p1_history)



    # Player 1: Best respond to empirical distribution of P2
    p1_payoff = payoff_matrix @ p2_counts
    p1_action = np.argmax(p1_payoff)
    p1_history.append(int(p1_action))

    # Player 2: Best respond to empirical distribution of P1
    # Negative because it's zero-sum, and payoff_matrix is from P1's perspective
    p2_payoff = -payoff_matrix.T @ p1_counts
    p2_action = np.argmax(p2_payoff)
    p2_history.append(int(p2_action))

    # Update frequencies
    p1_freqs[t] = np.bincount(p1_history, minlength=num_actions) / len(p1_history)
    p2_freqs[t] = np.bincount(p2_history, minlength=num_actions) / len(p2_history)

    print(f"---------------round {t}----------------")
    print("p1_counts", p1_counts)
    print("p2_counts", p2_counts)

    print("p1_payoff", p1_payoff)
    print("p2_payoff", p2_payoff)

    print("p1_action", p1_action)
    print("p2_action", p2_action)


    print("p1_history", p1_history)
    print("p2_history", p2_history)






    print()

# For labeling
actions = ['Rock', 'Paper', 'Scissors']

# Plot empirical frequencies
plt.figure(figsize=(12, 6))
for i in range(num_actions):
    plt.plot(p1_freqs[:, i], label=f'P1 {actions[i]}')
    plt.plot(p2_freqs[:, i], '--', label=f'P2 {actions[i]}')
plt.axhline(1/3, color='gray', linestyle=':', label='Nash equilibrium (1/3)')
plt.xlabel('Iteration')
plt.ylabel('Empirical Frequency')
plt.title('Fictitious Play in Rock-Paper-Scissors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
