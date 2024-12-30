import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
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
def get_user_input(prompt):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ['yes', 'no']:
            return user_input == 'yes'
        print("Invalid input. Please enter 'yes' or 'no'.")
    
def get_domain(dim, bounds, grid_points, x=None):
    """Generate grid points for density estimation."""
    if dim == 1:
        x = np.linspace(bounds[0][0], bounds[1][0], grid_points)
        dx = x[1] - x[0]
    elif dim >= 2:
        linspaces = [np.linspace(lb, ub, grid_points) for lb, ub in zip(bounds[0], bounds[1])]
        x = np.meshgrid(*linspaces, indexing='ij')
        dx = np.prod([(ub - lb) / (grid_points - 1) for lb, ub in zip(bounds[0], bounds[1])])
    else:
        raise ValueError("Dimension must be >= 1.")
    return x, dx
    
def plot_final_distr(results_dict, x, target_density, model_density, filename, output_directory, method='standard', dpi=100, plot3d=False):
    """
    Plot the final estimated distributions against the target distribution, one subplot per objective.
    """
    objectives = ['kl', 'l1', 'l2']
    image_path = os.path.join(output_directory, filename)
    single_variable = np.ndim(x) == 1

    def generate_3d_plot():
        X, Y = x
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        Z_target = target_density(grid_points).reshape(X.shape)
        
        fig = plt.figure(figsize=(20, 7), dpi=dpi)

        for i, obj in enumerate(objectives):
            params = results_dict[obj]
            Z_model = model_density(grid_points, params['optimal_means'], 
                                    params['optimal_variances'], 
                                    params['optimal_weights']).reshape(X.shape)

            ax = fig.add_subplot(1, 3, i + 1, projection='3d')
            ax.plot_surface(X, Y, Z_target, cmap='winter', alpha=0.9)
            ax.plot_surface(X, Y, Z_model, cmap='gray', alpha=0.5)
            ax.set_title(f"{obj.upper()} Distribution ({method.capitalize() if method != 'standard' else ''})")
            ax.set(xlabel='X', ylabel='Y', zlabel='Density')
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())
        plt.tight_layout()
        return fig

    def generate_2d_contour_plot():
        X, Y = x
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        Z_target = target_density(grid_points).reshape(X.shape)

        fig, axes = plt.subplots(1, len(objectives), figsize=(20, 7), dpi=dpi)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for ax, obj in zip(axes, objectives):
            params = results_dict[obj]
            Z_model = model_density(grid_points, params['optimal_means'], params['optimal_variances'], params['optimal_weights']).reshape(X.shape)

            ax.contourf(X, Y, Z_target, 20, cmap='winter', alpha=0.7)
            ax.contour(X, Y, Z_model, levels=30, cmap='viridis', linestyles='solid', alpha=0.6)

            optimal_means = np.array(params['optimal_means'])
            ax.scatter(optimal_means[:, 0], optimal_means[:, 1], color='black', s=100, edgecolor='white', label='Optimal Means')
            ax.set_title(f"Final {obj.upper()} Distribution {f'({method.capitalize()})' if method != 'standard' else ''}")
            ax.set(xlabel='X', ylabel='Y')
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())
        return fig

    def generate_1d_plot():
        fig = plt.figure(figsize=(10, 6))

        for obj in objectives:
            params = results_dict[obj]
            est_dist = model_density(x, params['optimal_means'], params['optimal_variances'], params['optimal_weights'])
            label = f'{obj.upper()} Optimization'
            if method == 'alternating':
                label += ' (Alternating Gradient Descent)'
            elif method == 'adam':
                label += ' (Adam)'

            plt.plot(x, est_dist, label=label, linewidth=2)
        plt.plot(x, target_density(x), label='Target Distribution', linewidth=3, color='black', linestyle='--', zorder=5)
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.title('Final Estimated Distributions vs. Target Distribution')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.xlim(x.min(), x.max())
        return fig

    # Select the appropriate plot generation function
    if single_variable:
        fig = generate_1d_plot()
    elif plot3d:
        fig = generate_3d_plot()
    else:
        fig = generate_2d_contour_plot()

    plt.tight_layout()
    plt.savefig(image_path, dpi=dpi)
    plt.close(fig)


def plot_error_decay(results_dict, filename, output_directory, method='standard', dpi=100):
    """
    Plot the decay of the error function with respect to iterations for all three objectives,
    both in linear and logarithmic scales in the same figure.

    Parameters:
    - results_dict: Dictionary containing optimization results.
    - filename: Name of the output image file.
    - output_directory: Directory where the image will be saved.
    - method: The optimization method used.
    """

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)

    # Plot in linear scale
    ax = axes[0]
    colors = {'kl': 'blue', 'l1': 'green', 'l2': 'red'}
    for objective in ['kl', 'l1']:
        values = results_dict[objective]['objective_values']
        ax.plot(values, label=f'{objective.upper()} Optimization', color=colors[objective])
    for objective in ['kl', 'l1', 'l2']:
        l2_values = results_dict[objective]['l2_values']
        ax.plot(l2_values, label=f'{objective.upper()} Optimization (L2 Norm)', linestyle='--', color=colors[objective], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    title_linear = 'Error Function Decay (Linear Scale)'
    if method == 'alternating':
        title_linear += ' (Alternating Gradient Descent)'
    elif method == 'adam':
        title_linear += ' (Adam)'
    ax.set_title(title_linear)
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # More ticks on x-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))  # More ticks on y-axis

    # Plot in logarithmic scale
    ax = axes[1]
    colors = {'kl': 'blue', 'l1': 'green', 'l2': 'red'}
    for objective in ['kl', 'l1']:
        values = results_dict[objective]['objective_values']
        ax.plot(values, label=f'{objective.upper()} Optimization', color=colors[objective])
    for objective in ['kl', 'l1', 'l2']:
        l2_values = results_dict[objective]['l2_values']
        ax.plot(l2_values, label=f'{objective.upper()} Optimization (L2 Norm)', linestyle='--', color=colors[objective], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    title_log = 'Error Function Decay (Logarithmic Scale)'
    if method == 'alternating':
        title_log += ' (Alternating Gradient Descent)'
    elif method == 'adam':
        title_log += ' (Adam)'
    ax.set_title(title_log)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="--")
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    image_path = os.path.join(output_directory, filename)
    plt.savefig(image_path)
    plt.close(fig)

def gif_generator(results_dict, x, target_density, model_density, max_iterations, output_directory, frame_step=5, method="standard", dpi=100, error=False, dist_filename="Distribution.gif", error_filename="Error.gif"):

    os.makedirs(output_directory, exist_ok=True)

    # Determine dimensionality
    dim = 2 if isinstance(x, (tuple, list)) and len(x) == 2 else 1
    frames = range(0, max_iterations, frame_step)
    if not frames:
        print("No frames to animate. Check 'max_iterations' or 'frame_step'.")
        return

    # Precompute target distribution (1D or 2D)
    if dim == 2:
        X, Y = x
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        Z_target = target_density(grid_points).reshape(X.shape)
    else:
        x_values = x
        Z_target = target_density(x_values)

    objectives = ["kl", "l1", "l2"]

    # ------------------------
    # 1) Distribution GIF
    # ------------------------
    dist_gif_path = os.path.join(output_directory, dist_filename)
    fig_dist, axes_dist = plt.subplots(1, len(objectives), figsize=(20, 7), dpi=dpi)
    if not isinstance(axes_dist, np.ndarray):
        axes_dist = [axes_dist]

    # Basic axis setup
    for ax in axes_dist:
        ax.set_xlabel("X" if dim == 2 else "x")
        ax.set_ylabel("Y" if dim == 2 else "Density")

    # Dictionary to store plot elements for each objective
    plot_elems_dist = {obj: {} for obj in objectives}

    # Initialize each axis with the target distribution
    for ax, obj in zip(axes_dist, objectives):
        if dim == 2:
            # Plot the target distribution as background contour
            ax.contourf(X, Y, Z_target, levels=20, cmap="winter", alpha=0.7)
            scatter = ax.scatter([], [], marker="s", s=30, color="navy", edgecolors="black")
            plot_elems_dist[obj]["scatter"] = scatter
            plot_elems_dist[obj]["contour"] = None
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())
        else:
            # Plot target distribution (1D)
            ax.plot(x_values, Z_target, label="Target", color="black")
            line, = ax.plot([], [], label=f"Model", color="blue")
            scatter = ax.scatter([], [], marker="s", s=50, color="navy", edgecolors="black")
            ax.legend(loc='upper right')
            ax.set_xlim(x_values.min(), x_values.max())
            margin = 0.1 * (Z_target.max() - Z_target.min())
            ax.set_ylim(Z_target.min() - margin, Z_target.max() + margin)
            plot_elems_dist[obj]["line"] = line
            plot_elems_dist[obj]["scatter"] = scatter

    def update_distribution(frame):
        for ax, obj in zip(axes_dist, objectives):
            idx = min(frame, len(results_dict[obj]["params_history"]) - 1)
            means, variances, weights = results_dict[obj]["params_history"][idx]

            if dim == 2:
                # Remove old contour for this objective
                if plot_elems_dist[obj]["contour"] is not None:
                    for c in plot_elems_dist[obj]["contour"].collections:
                        c.remove()

                Z_model = model_density(grid_points, means, variances, weights).reshape(X.shape)
                contour = ax.contour(X, Y, Z_model, levels=25, cmap="viridis", linestyles="solid", alpha=0.6)
                plot_elems_dist[obj]["contour"] = contour
                # Update scatter (2D means)
                scatter_sizes = 100 * np.array(weights) / np.sum(weights)  
                plot_elems_dist[obj]["scatter"].set_offsets(np.array(means)[:, :2])
                plot_elems_dist[obj]['scatter'].set_sizes(scatter_sizes)
            else:
                # Update 1D line
                est_dist = model_density(x_values, means, variances, weights)
                plot_elems_dist[obj]["line"].set_data(x_values, est_dist)
                # Scatter for the means (assume means is 1D array)
                scatter_sizes = 50 * np.array(weights) / np.sum(weights) 
                plot_elems_dist[obj]['scatter'].set_offsets(np.c_[means, np.zeros_like(means)])
                plot_elems_dist[obj]['scatter'].set_sizes(scatter_sizes)
            title = f"{obj.upper()} Optimization - Iteration {frame}"
            if method != "standard":
                title += f" ({method.capitalize()})"
            ax.set_title(title, fontsize=12)

    print("Creating distribution GIF...")
    pbar_dist = tqdm(total=len(frames))

    def distribution_progress(frame):
        update_distribution(frame)
        pbar_dist.update(1)

    ani_dist = FuncAnimation(fig_dist, distribution_progress, frames=frames, repeat=False)
    ani_dist.save(dist_gif_path, writer="pillow")
    pbar_dist.close()
    plt.close(fig_dist)
    print(f"Distribution GIF saved at: {dist_gif_path}")

    # ------------------------
    # 2) Error GIF (optional)
    # ------------------------
    if error:
        error_path = os.path.join(output_directory, error_filename)
        fig_err, axes_err = plt.subplots(1, len(objectives), figsize=(20, 7), dpi=dpi)
        if not isinstance(axes_err, np.ndarray):
            axes_err = [axes_err]

        # Store contour/colorbar or line references to remove them each frame
        plot_elems_err = {
            obj: {"contour": None, "colorbar": None, "lines": []} for obj in objectives
        }

        # Precompute global min/max for errors to keep a consistent color scale (or y-limits)
        y_limits = {}
        for obj in objectives:
            all_errors = []
            for frame_idx in frames:
                idx = min(frame_idx, len(results_dict[obj]["params_history"]) - 1)
                means, variances, weights = results_dict[obj]["params_history"][idx]
                if dim == 2:
                    Z_model = model_density(grid_points, means, variances, weights).reshape(X.shape)
                    error = Z_target - Z_model
                    all_errors.append(error.ravel())
                else:
                    Z_model = model_density(x_values, means, variances, weights)
                    error = Z_target - Z_model
                    all_errors.append(error)
            all_errors = np.concatenate(all_errors)
            y_min, y_max = all_errors.min(), all_errors.max()
            # Add some margin
            y_limits[obj] = (1.2 * y_min, 1.2 * y_max)

        def update_error(frame):
            for ax, obj in zip(axes_err, objectives):
                # Remove old contours, colorbars, or lines
                if plot_elems_err[obj]["contour"] is not None:
                    for c in plot_elems_err[obj]["contour"].collections:
                        c.remove()
                    plot_elems_err[obj]["contour"] = None

                if plot_elems_err[obj]["colorbar"] is not None:
                    plot_elems_err[obj]["colorbar"].remove()
                    plot_elems_err[obj]["colorbar"] = None

                for ln in plot_elems_err[obj]["lines"]:
                    ln.remove()
                plot_elems_err[obj]["lines"].clear()

                idx = min(frame, len(results_dict[obj]["params_history"]) - 1)
                means, variances, weights = results_dict[obj]["params_history"][idx]

                if dim == 2:
                    Z_model = model_density(grid_points, means, variances, weights).reshape(X.shape)
                    error = abs(Z_target - Z_model)
                    # Create a new contour for the error
                    contour_err = ax.contourf(X, Y, error, levels=20, cmap="Wistia")
                    # Add a colorbar and store it
                    cb_err = fig_err.colorbar(contour_err, ax=ax, shrink=0.8)
                    plot_elems_err[obj]["contour"] = contour_err
                    plot_elems_err[obj]["colorbar"] = cb_err

                    ax.set_xlim(X.min(), X.max())
                    ax.set_ylim(Y.min(), Y.max())
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")

                else:
                    Z_model = model_density(x_values, means, variances, weights)
                    error = Z_target - Z_model
                    line_err, = ax.plot(x_values, error, label=f"Iteration {idx}", color="red")
                    plot_elems_err[obj]["lines"].append(line_err)
                    ax.set_xlim(x_values.min(), x_values.max())
                    ymin, ymax = y_limits[obj]
                    ax.set_ylim(ymin, ymax)
                    ax.set_xlabel("x")
                    ax.set_ylabel("Error")
                    ax.legend(loc='upper right')

                title = f"{obj.upper()} Error Evolution"
                if method != "standard":
                    title += f" ({method.capitalize()})"
                ax.set_title(title, fontsize=12)

        print("Creating error GIF...")
        pbar_err = tqdm(total=len(frames))

        def error_progress(frame):
            update_error(frame)
            pbar_err.update(1)

        ani_err = FuncAnimation(fig_err, error_progress, frames=frames, repeat=False)
        ani_err.save(error_path, writer=PillowWriter(fps=10))
        pbar_err.close()
        plt.close(fig_err)
        print(f"Error GIF saved at: {error_path}")

    print("GIF generation completed.")