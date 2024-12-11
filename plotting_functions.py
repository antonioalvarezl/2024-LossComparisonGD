import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm  # Import tqdm for progress bar

# Define the get_unique_directory function
def get_unique_directory(base_directory, base_name):
    """
    Generates a unique directory name by appending an index to avoid overwriting.
    """
    i = 0
    while True:
        dir_name = f"{base_name}_{i+1}"
        full_path = os.path.join(base_directory, dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        i += 1
    
def plot_final_distr(x, target_density, model_density, results_dict, filename, output_directory, method='standard'):
    """
    Plot the final estimated distributions against the target distribution.

    Parameters:
    - x: The domain for plotting.
    - target_density: Function to compute the target density.
    - model_density: Function to compute the model density.
    - results_dict: Dictionary containing optimization results for each objective.
    - filename: Name of the output image file.
    - output_directory: Directory where the image will be saved.
    - method: The optimization method used.
    """
    import matplotlib.pyplot as plt
    import os

    # Remove the creation of 'Figures' directory

    plt.figure(figsize=(10, 6))

    # Plot the estimated distributions first
    for objective in ['kl', 'l1', 'l2']:
        optimal_means = results_dict[objective]['optimal_means']
        optimal_variances = results_dict[objective]['optimal_variances']
        optimal_weights = results_dict[objective]['optimal_weights']
        est_distribution = model_density(
            x,
            optimal_means,
            optimal_variances,
            optimal_weights
        )
        label = f'{objective.upper()} Optimization'
        if method == 'alternating':
            label += ' (Alternating Gradient Descent)'
        elif method == 'adam':
            label += ' (Adam)'
        plt.plot(x, est_distribution, label=label, linewidth=2)

    # Plot the target distribution on top
    plt.plot(x, target_density(x), label='Target Distribution', linewidth=3, color='black', linestyle='--', zorder=5)

    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Final Estimated Distributions vs. Target Distribution')
    plt.legend()
    plt.grid(True)
    # Save the figure
    image_path = os.path.join(output_directory, filename)
    plt.savefig(image_path)
    plt.close()  # Close the figure to free up memory

def plot_error_decay(results_dict, filename, output_directory, method='standard'):
    """
    Plot the decay of the error function with respect to iterations for all three objectives,
    both in linear and logarithmic scales in the same figure.

    Parameters:
    - results_dict: Dictionary containing optimization results.
    - filename: Name of the output image file.
    - output_directory: Directory where the image will be saved.
    - method: The optimization method used.
    """
    import matplotlib.pyplot as plt
    import os

    # Remove the creation of 'Figures' directory

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=80)

    # Plot in linear scale
    ax = axes[0]
    for objective in ['kl', 'l1', 'l2']:
        values = results_dict[objective]['objective_values']
        ax.plot(values, label=f'{objective.upper()} Optimization')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    title_linear = 'Error Function Decay (Linear Scale)'
    if method == 'alternating':
        title_linear += ' (Alternating Gradient Descent)'
    elif method == 'adam':
        title_linear += ' (Adam)'
    ax.set_title(title_linear)
    ax.legend()
    ax.grid(True)

    # Plot in logarithmic scale
    ax = axes[1]
    for objective in ['kl', 'l1', 'l2']:
        values = results_dict[objective]['objective_values']
        ax.plot(values, label=f'{objective.upper()} Optimization')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    title_log = 'Error Function Decay (Logarithmic Scale)'
    if method == 'alternating':
        title_log += ' (Alternating Gradient Descent)'
    elif method == 'adam':
        title_log += ' (Adam)'
    ax.set_title(title_log)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    image_path = os.path.join(output_directory, filename)
    plt.savefig(image_path)
    plt.close(fig)
    
def create_combined_gif(results_dict, x, target_density, model_density, max_iterations, filename, output_directory, frame_step=5, method='standard', chosen_params=None):
    """
    Create a GIF showing the evolution of the estimated distributions for different objectives.

    Parameters:
    - results_dict: Dictionary containing optimization results.
    - x: Domain for plotting.
    - target_density: Function to compute the target density.
    - model_density: Function to compute the model density.
    - max_iterations: Maximum number of iterations.
    - filename: Name of the output GIF file.
    - output_directory: Directory where the GIF will be saved.
    - frame_step: Step size for frames in the GIF.
    - method: Optimization method used.
    - chosen_params: List of parameters to optimize.
    """
    from matplotlib.animation import PillowWriter

    gif_path = os.path.join(output_directory, filename)

    fig, axes = plt.subplots(3, 1, figsize=(10, 18), dpi=80)
    plt.tight_layout(pad=4.0)

    frames = range(0, max_iterations, frame_step)

    print("Creating combined GIF...")
    pbar = tqdm(total=len(frames))

    def update(frame):
        for ax, objective in zip(axes, ['kl', 'l1', 'l2']):
            ax.clear()
            params_history = results_dict[objective]['params_history']
            idx = min(frame, len(params_history)-1)
            means, variances, weights = params_history[idx]
            estimated_distribution = model_density(x, means, variances, weights)
            target_distribution = target_density(x)
            
            # Plot distributions
            ax.plot(x, estimated_distribution, label=f"Iteration {idx}", alpha=0.7)
            ax.plot(x, target_distribution, label="Target Distribution", color="black", linewidth=2)
            
            # Mark each Gaussian mean on the x-axis
            ax.scatter(means, np.zeros_like(means), marker='s', s=100, label='Model Means')
            
            ax.set_xlabel("x")
            ax.set_ylabel("Probability Density")
            title = f"{objective.upper()} Optimization"
            if method == 'alternating':
                title += " (Alternating GD)"
            elif method == 'adam':
                title += " (Adam)"
            elif method == 'stagewise':
                param_index = frame // (max_iterations // len(chosen_params))
                param_name = chosen_params[min(param_index, len(chosen_params)-1)]
                title += f" (Stagewise): Optimizing {param_name.capitalize()}"
            ax.set_title(title)
            ax.legend()
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(0, max(target_distribution)*1.2)
        pbar.update(1)

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)

    writer = PillowWriter(fps=10)
    ani.save(gif_path, writer=writer)
    pbar.close()
    plt.close(fig)

def create_error_evolution_gif(results_dict, x, target_density, model_density, max_iterations, filename, output_directory, frame_step=5, method='standard'):
    """
    Create a GIF showing the evolution of the difference between the target function and the model
    for all three objectives, with fixed y-axis scales.

    Parameters:
    - results_dict: Dictionary containing optimization results.
    - x: Domain for plotting.
    - target_density: Function to compute the target density.
    - model_density: Function to compute the model density.
    - max_iterations: Maximum number of iterations.
    - filename: Name of the output GIF file.
    - output_directory: Directory where the GIF will be saved.
    - frame_step: Step size for frames in the GIF.
    - method: Optimization method used.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import os
    from tqdm import tqdm

    # Remove the creation of 'Gifs' directory

    # Update the filename to include the full path
    gif_path = os.path.join(output_directory, filename)

    fig, axes = plt.subplots(3, 1, figsize=(10, 18), dpi=80)
    plt.tight_layout(pad=4.0)

    target_distribution = target_density(x)

    frames = range(0, max_iterations, frame_step)

    # Create a progress bar
    print("Creating error evolution GIF...")
    pbar = tqdm(total=len(frames))

    # Precompute global y-axis limits for each subplot
    error_values = {'kl': [], 'l1': [], 'l2': []}

    # Collect error values across all frames for each objective
    for frame in frames:
        for objective in ['kl', 'l1', 'l2']:
            params_history = results_dict[objective]['params_history']
            idx = min(frame, len(params_history)-1)
            means, variances, weights = params_history[idx]
            estimated_distribution = model_density(x, means, variances, weights)
            error = target_distribution - estimated_distribution
            error_values[objective].extend(error)

    # Compute global min and max for y-axis limits
    y_limits = {}
    for objective in ['kl', 'l1', 'l2']:
        errors = np.array(error_values[objective])
        ymin, ymax = np.min(errors)*1.2, np.max(errors)*1.2
        y_limits[objective] = (ymin, ymax)

    def update(frame):
        for ax, objective in zip(axes, ['kl', 'l1', 'l2']):
            ax.clear()
            params_history = results_dict[objective]['params_history']
            idx = min(frame, len(params_history)-1)
            means, variances, weights = params_history[idx]
            estimated_distribution = model_density(x, means, variances, weights)
            error = target_distribution - estimated_distribution
            ax.plot(x, error, label=f"Iteration {idx}", alpha=0.7)
            ax.set_xlabel("x")
            ax.set_ylabel("Error")
            title = f"{objective.upper()} Optimization Error Evolution"
            if method == 'alternating':
                title += " (Alternating Gradient Descent)"
            elif method == 'adam':
                title += " (Adam)"
            ax.set_title(title)
            ax.legend()
            ax.set_xlim(-10, 10)
            ymin, ymax = y_limits[objective]
            ax.set_ylim(ymin, ymax)
        pbar.update(1)  # Update progress bar

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)

    # Save the animation with PillowWriter
    writer = PillowWriter(fps=10)
    ani.save(gif_path, writer=writer)
    pbar.close()
    plt.close(fig)
