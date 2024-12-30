import json 
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
from gauss_mix_optimization import gaussian_mixture, gradient_descent
from plotting_functions import plot_final_distr, plot_error_decay, gif_generator, get_unique_directory, get_user_input, get_domain

# -------------------- Generate target parameters --------------------
dim = int(input("Enter the number of dimensions: "))

# Target parameters: means, variances, weights
target_N = 40
target_seed = 1
np.random.seed(target_seed)
target_means = np.random.uniform(-10, 10, (target_N, dim))
target_var = 0.1
var_clip = target_var * 0.1

target_variances = np.full((target_N, dim), target_var)
#target_variances = np.ones((target_N, dim))
if np.any(target_variances <= 0):
    raise ValueError("Variances must be greater than 0.")
target_weights = np.ones(target_N)
weight_clip = 0.01

# -------------------- Generate initial model parameters --------------------
model_N = 5
initial_seed = 10
np.random.seed(initial_seed)
initial_means = np.random.uniform(-5, 5, (model_N, dim))
initial_variances = np.ones((model_N, dim))
if np.any(initial_variances <= 0):
    raise ValueError("Variances must be greater than 0.")
initial_weights = np.ones(model_N)

# -------------------- Define density functions --------------------
def target_density(x):
    return gaussian_mixture(x, target_means, target_variances, target_weights)

def model_density(x, means, variances, weights):
    return gaussian_mixture(x, means, variances, weights)

# -------------------- Optimization parameters --------------------
tol = 1e-3
max_iterations = 15000
verbose = True
signed_weights = False
print(f"Optimization settings: tol={tol}, max_iterations={max_iterations}, verbose={verbose}, signed_weights={signed_weights}")

# -------------------- Plotting options --------------------
dpi = 250
frame_step=int(0.01*max_iterations)
error = False
plot3d = False
grid_points = 200 #Number of points per dimension
# Calculate bounds based on means and variances
bound_factor = 10
all_means = np.vstack([target_means, initial_means])
all_variances = np.vstack([target_variances, initial_variances])
bounds = np.array([
    np.min(all_means - bound_factor * np.sqrt(all_variances), axis=0),
    np.max(all_means + bound_factor * np.sqrt(all_variances), axis=0)
])
x, dx = get_domain(dim, bounds, grid_points)

# -------------------- User input for optimization options --------------------
print("Select the parameters to optimize (enter 'yes' or 'no'):")
optimize_means = get_user_input("Optimize means? (yes/no): ")
optimize_variances = get_user_input("Optimize variances? (yes/no): ")
optimize_weights = get_user_input("Optimize weights? (yes/no): ")

optimized_params_str = ''.join(
    p for p, o in zip(['M', 'V', 'W'], [optimize_means, optimize_variances, optimize_weights]) if o
) or 'None'

print("\nSelect the optimization method:")
print("1. Standard Gradient Descent")
print("2. Alternating Gradient Descent")
print("3. Adam Optimizer")
print("4. Stage-wise Optimization")
method_input = input("Enter the number corresponding to your choice (1/2/3/4): ").strip()
method_mapping = {'1': 'standard', '2': 'alternating', '3': 'adam', '4': 'stagewise'}
method = method_mapping.get(method_input, 'standard')
print(f"Optimization method selected: {method}")

if method == 'stagewise':
    print("\nYou selected stage-wise optimization. Choose the method to use in each stage:")
    print("1. Standard Gradient Descent")
    print("2. Adam Optimizer")
    stagewise_method_input = input("Enter 1 for standard or 2 for adam: ").strip()
    stagewise_method_mapping = {'1': 'standard', '2': 'adam'}
    stagewise_method = stagewise_method_mapping.get(stagewise_method_input, 'standard')
    print(f"Stage-wise method selected: {stagewise_method}")
else:
    stagewise_method = None

# -------------------- Learning rate selection --------------------
if method == 'alternating':
    print("\nEnter separate learning rates for each parameter type:")
    lr_means = float(input("Learning rate for means: ").strip())
    lr_variances = float(input("Learning rate for variances: ").strip())
    lr_weights = float(input("Learning rate for weights: ").strip())
    learning_rate = [lr_means, lr_variances, lr_weights]
elif method == 'stagewise':
    learning_rate = float(input("\nEnter the learning rate for the stagewise optimization: ").strip())
else:
    learning_rate = float(input("\nEnter the learning rate: ").strip())

if method == 'alternating':
    lr_str = '_'.join(str(lr) for lr in learning_rate)
else:
    lr_str = str(learning_rate)

filename = f"{optimized_params_str}_{method}_lr{lr_str}_target{target_N}_model{model_N}_{dim}D"

# -------------------- Create output directory --------------------
base_output_dir = "Results"
os.makedirs(base_output_dir, exist_ok=True)
output_directory = get_unique_directory(base_output_dir, filename)


# -------------------- Optimization Process --------------------
objectives = ['kl', 'l1', 'l2']  # Parameterized list of objectives
results_dict = {}
parameters_to_optimize = [('means', optimize_means), ('variances', optimize_variances), ('weights', optimize_weights)]
chosen_params = [p for p, o in parameters_to_optimize if o]
num_params_chosen = len(chosen_params) if chosen_params else 1

for objective in objectives:
    print(f"\nOptimizing for objective: {objective.upper()}")

    if method != 'stagewise':
        optimal_means, optimal_variances, optimal_weights, objective_values, l2_values, params_history = gradient_descent(
            dx, target_density, model_density, initial_means, initial_variances, initial_weights, 
            optimize_means, optimize_variances, optimize_weights, signed_weights, 
            objective, learning_rate, var_clip, weight_clip, tol, max_iterations, method, verbose, x
        )
    else:
        stage_iterations = max_iterations // num_params_chosen
        current_means = initial_means.copy()
        current_variances = initial_variances.copy()
        current_weights = initial_weights.copy()
        all_objective_values = []
        all_l2_values = []
        all_params_history = []

        for param_type in chosen_params:
            opt_means_stage = (param_type == 'means')
            opt_vars_stage = (param_type == 'variances')
            opt_wgts_stage = (param_type == 'weights')

            m, v, w, objs, l2vs, phist = gradient_descent(
                dx, target_density, model_density, current_means, current_variances, current_weights, 
                opt_means_stage, opt_vars_stage, opt_wgts_stage, objective, learning_rate, var_clip, weight_clip, tol, stage_iterations, 
                stagewise_method, verbose, x,
            )

            current_means, current_variances, current_weights = m, v, w
            all_objective_values.extend(objs)
            all_l2_values.extend(l2vs)
            all_params_history.extend(phist)

            if objs and objs[-1] <= tol:
                break

        optimal_means, optimal_variances, optimal_weights = current_means, current_variances, current_weights
        objective_values, l2_values, params_history = all_objective_values, all_l2_values, all_params_history

    results_dict[objective] = {'optimal_means': optimal_means.tolist(), 'optimal_variances': optimal_variances.tolist(), 'optimal_weights': optimal_weights.tolist(), 'objective_values': objective_values, 'l2_values': l2_values, 'params_history': [tuple(params) for params in params_history]}

# -------------------- Saving Parameters to InitialValues.json --------------------
parameters = {'target_N': target_N, 'model_N': model_N, 'dim': dim, 'random_seeds': {'target': target_seed, 'initial_model': initial_seed}, 'target_parameters': {'means': target_means.tolist(), 'variances': target_variances.tolist(), 'weights': target_weights.tolist()}, 'initial_model_parameters': {'means': initial_means.tolist(), 'variances': initial_variances.tolist(), 'weights': initial_weights.tolist()}, 'optimization_settings': {'optimize_means': optimize_means, 'optimize_variances': optimize_variances, 'optimize_weights': optimize_weights, 'method': method}, 'learning_rate': learning_rate, 'algorithm_parameters': {'tol': tol, 'max_iterations': max_iterations, 'verbose': verbose}}
initial_values_path = os.path.join(output_directory, 'InitialValues.json')
with open(initial_values_path, 'w') as json_file:
    json.dump(parameters, json_file, indent=4)

# -------------------- Plotting --------------------
print('Plotting')
if dim > 2:
    print("High-dimensional data detected. Only error decay is plotted")
    plot_error_decay(results_dict, f'Obj_{filename}.png', output_directory, method, dpi=dpi)
else:
    if dim ==2  and plot3d:
        plot_final_distr(results_dict, x, target_density, model_density, f'FinalDistr_{filename}.png', output_directory, method, dpi=dpi, plot3d=plot3d)
    else:
        plot_final_distr(results_dict, x, target_density, model_density, f'FinalDistr_{filename}.png', output_directory, method, dpi=dpi)
    plot_error_decay(results_dict, f'Obj_{filename}.png', output_directory, method, dpi=dpi)
    gif_generator(results_dict, x, target_density, model_density, max_iterations, output_directory, frame_step=frame_step, method=method, dpi=dpi, error=error)
