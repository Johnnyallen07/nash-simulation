import numpy as np
import matplotlib.pyplot as plt


# Function to simulate Best Response Dynamics for Rock-Paper-Scissors
def best_response_dynamics(initial_dist, num_iterations=20):
    # Payoff matrix for RPS: rows = strategies of player, columns = opponent distribution
    # Order: Rock, Paper, Scissors
    payoff_matrix = np.array([
        [0, -1, 1],  # Rock vs [Rock, Paper, Scissors]
        [1, 0, -1],  # Paper vs [Rock, Paper, Scissors]
        [-1, 1, 0]  # Scissors vs [Rock, Paper, Scissors]
    ])

    distributions = [np.array(initial_dist)]

    for _ in range(num_iterations):
        current_dist = distributions[-1]
        # Expected payoff for each pure strategy: payoff_matrix * current_dist
        expected_payoffs = payoff_matrix.dot(current_dist)
        # Choose best response (if ties, pick first max)
        best_response = np.zeros(3)
        best_response[np.argmax(expected_payoffs)] = 1
        distributions.append(best_response)

    return np.array(distributions)


random_values = np.random.rand(3)
initial_distribution = random_values / random_values.sum()
distributions = best_response_dynamics(initial_distribution, num_iterations=20)

# Plotting the time series of probabilities
time_steps = np.arange(distributions.shape[0])
plt.figure(figsize=(10, 6))
plt.plot(time_steps, distributions[:, 0], marker='o', label='Rock')
plt.plot(time_steps, distributions[:, 1], marker='s', label='Paper')
plt.plot(time_steps, distributions[:, 2], marker='^', label='Scissors')
plt.xlabel('Time Step')
plt.ylabel('Probability')
plt.title('Best-Response Dynamics in Rock-Paper-Scissors')
plt.legend()
plt.grid(True)
plt.ylim(-0.1, 1.1)
plt.show()
