import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Original multilaterate function
def multilaterate_3D(anchors, distances):
    def objective(position):
        return np.sum((np.linalg.norm(anchors - position, axis=1) - distances)**2)
    
    initial_guess = np.mean(anchors, axis=0)
    result = minimize(objective, initial_guess)
    return result.x

# Objective function for optimization
def objective(position, anchors, distances):
    return np.sum((np.linalg.norm(anchors - position, axis=1) - distances)**2)

# Modified multilaterate function
def multilaterate_3D_top_t(anchors, distances, t=10, threshold=0.5, max_iter=50, num_guesses=5):
    initial_guesses = np.random.rand(num_guesses, 3) * 50
    estimated_positions = []
    
    for guess in initial_guesses:
        result = minimize(objective, guess, args=(anchors, distances), options={'maxiter': max_iter})
        estimated_positions.append(result.x)
    
    estimated_positions = np.array(estimated_positions)
    distances_to_true = np.linalg.norm(estimated_positions - true_position_3D, axis=1)
    sorted_indices = np.argsort(distances_to_true)
    
    for idx in sorted_indices[:t]:
        if distances_to_true[idx] <= threshold:
            return True
    return False

# Run the experiment with top t positions
def run_experiment_top_t(t=3, threshold=0.01):
    results = []
    for error_magnitude in measurement_errors_range:
        measurement_errors_3D = np.random.normal(0, error_magnitude, size=true_distances_3D.shape)
        measured_distances_3D = true_distances_3D + measurement_errors_3D
        is_true_position_in_top_t = multilaterate_3D_top_t(anchors_3D, measured_distances_3D, t, threshold)
        results.append(is_true_position_in_top_t)
    return np.array(results)

# Experiment setup
np.random.seed(0)
anchors_3D = np.random.rand(24, 3) * 50
true_position_3D = np.array([25, 25, 25])
true_distances_3D = np.linalg.norm(anchors_3D - true_position_3D, axis=1)
measurement_errors_range = np.linspace(0, 5, 100)
N_reduced = 100
results_reduced = np.zeros((N_reduced, len(measurement_errors_range)), dtype=bool)

for i in range(N_reduced):
    results_reduced[i] = run_experiment_top_t()

frequencies_reduced = np.mean(results_reduced, axis=0)

# Plotting the frequencies to visualize how often the true position is in the top t predictions

plt.figure(figsize=(10, 6))
plt.plot(measurement_errors_range, frequencies_reduced * 100, color='blue')  # Convert frequencies to percentages
plt.xlabel('Measurement Error Magnitude')
plt.ylabel('Frequency (%)')
plt.title('Frequency of True Position (threshold = 0.01) in Top 3 Predictions vs. Measurement Error Magnitude')
plt.grid(True)
plt.show()