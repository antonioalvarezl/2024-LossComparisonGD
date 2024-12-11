import json 
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
from gauss_mix_optimization import gaussian_mixture, gradient_descent
from plotting_functions import (
    plot_final_distr, create_combined_gif, plot_error_decay,
    create_error_evolution_gif, get_unique_directory
)

# -------------------- Generate target parameters --------------------
target_N = 4
target_seed = 1
np.random.seed(target_seed)
target_means = np.random.uniform(-5, 5, target_N)
#target_variances = np.ones(target_N)
target_variances = np.full(target_N, 0.05)
target_weights = np.ones(target_N)

# -------------------- Generate initial model parameters --------------------
model_N = 20
initial_seed = 1
np.random.seed(initial_seed)
initial_means = np.random.uniform(-6, 6, model_N)
initial_variances = np.ones(model_N)
initial_weights = np.ones(model_N)

def target_density(x):
    return gaussian_mixture(x, target_means, target_variances, target_weights)

def model_density(x, means, variances, weights):
    return gaussian_mixture(x, means, variances, weights)

# -------------------- Optimization parameters --------------------
tol = 1e-3
max_iterations = 40000
verbose = True
signed_weights = True

# -------------------- User input for optimization options --------------------
print("Select the parameters to optimize (enter 'yes' or 'no'):")
optimize_means = input("Optimize means? (yes/no): ").strip().lower() == 'yes'
optimize_variances = input("Optimize variances? (yes/no): ").strip().lower() == 'yes'
optimize_weights = input("Optimize weights? (yes/no): ").strip().lower() == 'yes'

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

# If stagewise is chosen, ask the user which method to use inside each stage (standard or adam)
if method == 'stagewise':
    print("\nYou selected stage-wise optimization. Choose the method to use in each stage:")
    print("1. Standard Gradient Descent")
    print("2. Adam Optimizer")
    stagewise_method_input = input("Enter 1 for standard or 2 for adam: ").strip()
    stagewise_method_mapping = {'1': 'standard', '2': 'adam'}
    stagewise_method = stagewise_method_mapping.get(stagewise_method_input, 'standard')
    # 'alternating' not included as per user's requirement
    # We will store stagewise_method and use it later when running each stage
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
    # Single learning rate for stagewise (standard or adam)
    learning_rate = float(input("\nEnter the learning rate for the stagewise optimization: ").strip())
else:
    learning_rate = float(input("\nEnter the learning rate: ").strip())

# -------------------- Construct filename --------------------
if method == 'alternating':
    lr_str = '_'.join(str(lr) for lr in learning_rate)
else:
    lr_str = str(learning_rate)

filename = f"{optimized_params_str}_{method}_lr{lr_str}_target{target_N}_model{model_N}"

# -------------------- Create output directory --------------------
base_output_dir = "Results"
os.makedirs(base_output_dir, exist_ok=True)
output_directory = get_unique_directory(base_output_dir, filename)
print(f"All outputs will be saved in: {output_directory}")

# -------------------- Define domain for plotting and optimization --------------------
bounds = np.concatenate([target_means - 3 * np.sqrt(target_variances), target_means + 3 * np.sqrt(target_variances), initial_means - 3 * np.sqrt(initial_variances), initial_means + 3 * np.sqrt(initial_variances)])
lower_bound, upper_bound = np.min(bounds), np.max(bounds)
x = np.linspace(lower_bound, upper_bound, 200)

# -------------------- Optimization Process --------------------
results_dict = {}
parameters_to_optimize = [('means', optimize_means), ('variances', optimize_variances), ('weights', optimize_weights)]
chosen_params = [p for p, o in parameters_to_optimize if o]
num_params_chosen = len(chosen_params) if chosen_params else 1

for objective in ['kl', 'l1', 'l2']:
    print(f"\nOptimizing for objective: {objective.upper()}")

    if method != 'stagewise':
        # Directly run gradient_descent once for the chosen parameters
        optimal_means, optimal_variances, optimal_weights, objective_values, params_history = gradient_descent(target_density=target_density, model_density=model_density, initial_means=initial_means, initial_variances=initial_variances, initial_weights=initial_weights, optimize_means=optimize_means, optimize_variances=optimize_variances, optimize_weights=optimize_weights, signed_weights=signed_weights, objective=objective, learning_rate=learning_rate, tol=tol, max_iterations=max_iterations, method=method, verbose=verbose, x=x)
    else:
        # Stagewise optimization
        # Divide the total iterations among the chosen parameters
        stage_iterations = max_iterations // num_params_chosen

        current_means = initial_means.copy()
        current_variances = initial_variances.copy()
        current_weights = initial_weights.copy()

        all_objective_values = []
        all_params_history = []

        # Optimize each chosen parameter type in sequence
        for param_type in chosen_params:
            opt_means_stage = (param_type == 'means')
            opt_vars_stage = (param_type == 'variances')
            opt_wgts_stage = (param_type == 'weights')

            # Run one stage
            m, v, w, objs, phist = gradient_descent(target_density, model_density, current_means, current_variances, current_weights, opt_means_stage, opt_vars_stage, opt_wgts_stage, objective, learning_rate, tol, stage_iterations, stagewise_method, verbose, x)

            current_means, current_variances, current_weights = m, v, w
            all_objective_values.extend(objs)
            all_params_history.extend(phist)

            # If tolerance reached, stop early
            if objs and objs[-1] <= tol:
                break

        # After completing all stages or stopping early
        optimal_means, optimal_variances, optimal_weights = current_means, current_variances, current_weights
        objective_values, params_history = all_objective_values, all_params_history


    results_dict[objective] = {
        'optimal_means': optimal_means.tolist(),
        'optimal_variances': optimal_variances.tolist(),
        'optimal_weights': optimal_weights.tolist(),
        'objective_values': objective_values,
        'params_history': [tuple(params) for params in params_history]
    }

# -------------------- Plotting --------------------
plot_final_distr(x, target_density, model_density, results_dict, f'FinalDistr_{filename}.png', output_directory, method)
plot_error_decay(results_dict, f'Obj_{filename}.png', output_directory, method)
create_combined_gif(results_dict, x, target_density, model_density, max_iterations, f'Evol_{filename}.gif', output_directory, frame_step=int(0.005*max_iterations), method=method, chosen_params=chosen_params)
create_error_evolution_gif(results_dict, x, target_density, model_density, max_iterations, f'Error_{filename}.gif', output_directory, frame_step=int(0.005*max_iterations), method=method)


# -------------------- Saving Parameters to InitialValues.json --------------------

# Prepare the parameters dictionary
parameters = {'target_N': target_N, 'model_N': model_N, 'random_seeds': {'target': target_seed, 'initial_model': initial_seed}, 'target_parameters': {'means': target_means.tolist(), 'variances': target_variances.tolist(), 'weights': target_weights.tolist()}, 'initial_model_parameters': {'means': initial_means.tolist(), 'variances': initial_variances.tolist(), 'weights': initial_weights.tolist()}, 'optimization_settings': {'optimize_means': optimize_means, 'optimize_variances': optimize_variances, 'optimize_weights': optimize_weights, 'method': method}, 'learning_rate': learning_rate, 'algorithm_parameters': {'tol': tol, 'max_iterations': max_iterations, 'verbose': verbose}}

# Save parameters to a JSON file
initial_values_path = os.path.join(output_directory, 'InitialValues.json')
with open(initial_values_path, 'w') as json_file:
    json.dump(parameters, json_file, indent=4)
