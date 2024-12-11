import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

def kl_divergence(p, q, dx, eps=1e-12):
    """Calculate the KL divergence between two distributions."""
    return np.sum(np.where(p != 0, p * np.log((p + eps) / (q + eps)), 0)) * dx

def l1_norm(p, q, dx):
    """Calculate the L1 norm (integrated absolute difference) between two distributions."""
    return np.sum(np.abs(p - q)) * dx

def l2_norm(p, q, dx):
    """Calculate the L2 norm (squared Euclidean distance) between two distributions."""
    return np.sqrt(np.sum((p - q) ** 2) * dx)

def gaussian(x, mean, variance):
    """Compute Gaussian distribution with normalization."""
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / variance)

def gaussian_mixture(x, means, variances, weights):
    """Compute a mixture of Gaussians without normalizing weights."""
    weights = np.asarray(weights)
    mixture = sum(w * gaussian(x, m, v) for w, m, v in zip(weights, means, variances))
    return mixture / np.sum(weights)

def grad_q_wrt_mu(x, means, variances, weights, i):
    """Compute gradient of q w.r.t mean μ_i."""
    w = weights[i]
    g_i = w * gaussian(x, means[i], variances[i])
    return (g_i * (x - means[i]) / variances[i]) / np.sum(weights)

def grad_q_wrt_variance(x, means, variances, weights, i):
    """Compute gradient of q w.r.t variance σ_i^2."""
    w = weights[i]
    g_i = w * gaussian(x, means[i], variances[i])
    return (g_i * (((x - means[i]) ** 2) / (2 * variances[i] ** 2) - 1 / (2 * variances[i]))) / np.sum(weights)

def grad_q_wrt_weight(x, means, variances, weights, i, model):
    """Compute gradient of q w.r.t weight w_i."""
    total_w = np.sum(weights)
    g_i = gaussian(x, means[i], variances[i])
    return (g_i - model) / total_w

def adam_optimizer(params, grads, m, v, t, learning_rate=0.001, 
                  beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs an Adam optimization step. Adam update algorithm is an optimization method used for training machine learning models, particularly neural networks.  Adam combines the benefits of two other popular optimization algorithms: AdaGrad and RMSProp.
    1. AdaGrad adapts the learning rate to parameters, performing larger updates for infrequent parameters and smaller updates for frequent ones. However, its continuously accumulating squared gradients can lead to an overly aggressive and monotonically decreasing learning rate.
    2. RMSProp modifies AdaGrad by using a moving average of squared gradients to adapt the learning rate, which resolves the radical diminishing learning rates of AdaGrad.

    Adam takes this a step further by:
    - Calculating an exponentially moving average of the gradients (m) to smooth out the gradient descent path, addressing the issue of noisy gradients.
    - Computing an exponentially moving average of the squared gradients (v), which scales the learning rate inversely proportional to the square root of the second moments of the gradients. This helps in adaptive learning rate adjustments.
    - Implementing bias corrections to the first (m_hat) and second (v_hat) moment estimates to account for their initialization at the origin, leading to more accurate updates at the beginning of the training.

    This results in an optimization algorithm that can handle sparse gradients on noisy problems, which is efficient for large datasets and high-dimensional parameter spaces.

    Parameters:
    - params (list of np.ndarray): Parameters to be updated (e.g., means, variances, weights).
    - grads (list of np.ndarray): Gradients corresponding to the parameters.
    - m (list of np.ndarray): First moment estimates for each parameter.
    - v (list of np.ndarray): Second moment estimates for each parameter.
    - t (int): Current timestep (iteration number).
    - learning_rate (float): Learning rate for the update.
    - beta1 (float): Exponential decay rate for the first moment estimates.
    - beta2 (float): Exponential decay rate for the second moment estimates.
    - epsilon (float): Small constant for numerical stability.
    """
    
    for i in range(len(params)):
        # Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        # Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
        # Compute bias-corrected first moment estimate
        m_hat = m[i] / (1 - beta1 ** t)
        # Compute bias-corrected second moment estimate
        v_hat = v[i] / (1 - beta2 ** t)
        # Update parameters
        params[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

def gradient_descent(target_density, model_density, initial_means, initial_variances, initial_weights,
                     optimize_means=True, optimize_variances=False, optimize_weights=False, signed_weights=False,
                     objective='kl', learning_rate=0.01,
                     tol=1e-4, max_iterations=1000, method='standard', verbose=False, x=None,
                     beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Optimize a Gaussian mixture model to approximate a target density.
    """
    if x is None or len(x) < 2:
        raise ValueError("A valid domain x must be provided for the optimization.")
    
    means = initial_means.copy()
    variances = initial_variances.copy()
    weights = initial_weights.copy()
    N = len(means)

    objective_values = []
    params_history = []
    dx = x[1] - x[0]
    target = target_density(x)

    print(f"Starting optimization using {objective.upper()} with {method} gradient descent...")
    pbar = tqdm(total=max_iterations)

    optimization_sequence = []
    if optimize_means:
        optimization_sequence.append('means')
    if optimize_variances:
        optimization_sequence.append('variances')
    if optimize_weights:
        optimization_sequence.append('weights')
    
    if method == 'alternating':
        learning_rate = np.array(learning_rate)
        lr_means, lr_variances, lr_weights = learning_rate
    elif method in ['standard', 'adam']:
        lr_means = lr_variances = lr_weights = learning_rate
    else:
        raise ValueError("Invalid method specified. Choose 'standard', 'alternating', 'adam', or 'stagewise'.")

    if method == 'adam':
        params = [means, variances, weights]
        m = [np.zeros_like(param) for param in params]
        v = [np.zeros_like(param) for param in params]

    for iteration in range(1, max_iterations + 1):
        if signed_weights == False:
            weights = np.maximum(weights, 1e-8)
        model = model_density(x, means, variances, weights)

        if objective == 'kl':
            obj_value = kl_divergence(target, model, dx)
        elif objective == 'l1':
            obj_value = l1_norm(target, model, dx)
        elif objective == 'l2':
            obj_value = l2_norm(target, model, dx)
        else:
            raise ValueError("Invalid objective function specified.")

        objective_values.append(obj_value)
        params_history.append((means.copy(), variances.copy(), weights.copy()))

        if obj_value <= tol:
            pbar.update(max_iterations - iteration + 1)
            break

        if objective == 'kl':
            factor = -target / np.where(model > 1e-8, model, 1e-8)
        elif objective == 'l1':
            factor = np.sign(model - target)
        elif objective == 'l2':
            factor = 2 * (model - target)

        grad_means = np.zeros_like(means)
        grad_variances = np.zeros_like(variances)
        grad_weights = np.zeros_like(weights)

        for i in range(N):
            if optimize_means:
                dq_dmu_i = grad_q_wrt_mu(x, means, variances, weights, i)
                grad_means[i] = np.sum(factor * dq_dmu_i) * dx

            if optimize_variances:
                dq_dvar_i = grad_q_wrt_variance(x, means, variances, weights, i)
                grad_variances[i] = np.sum(factor * dq_dvar_i) * dx

            if optimize_weights:
                dq_dwi = grad_q_wrt_weight(x, means, variances, weights, i, model)
                grad_weights[i] = np.sum(factor * dq_dwi) * dx

        if verbose and iteration % 500 == 0:
            print(f"Iteration {iteration}: Obj={obj_value:.6f}, "
                  f"||grad_means||={np.linalg.norm(grad_means):.6f}, "
                  f"||grad_variances||={np.linalg.norm(grad_variances):.6f}, "
                  f"||grad_weights||={np.linalg.norm(grad_weights):.6f}")

        if method == 'standard':
            if optimize_means:
                means -= lr_means * grad_means
            if optimize_variances:
                variances = np.clip(variances - lr_variances * grad_variances, 0.1, 10.0)
            if optimize_weights:
                if signed_weights == False:
                    weights = np.maximum(weights - lr_weights * grad_weights, 1e-8)
                else:
                    weights -= lr_weights * grad_weights

        elif method == 'alternating':
            for param_to_update in optimization_sequence:
                if param_to_update == 'means' and optimize_means:
                    means -= lr_means * grad_means
                elif param_to_update == 'variances' and optimize_variances:
                    variances = np.clip(variances - lr_variances * grad_variances, 0.1, 10.0)
                elif param_to_update == 'weights' and optimize_weights:
                    if signed_weights == False:
                        weights = np.maximum(weights - lr_weights * grad_weights, 1e-8)
                    else:
                        weights -= lr_weights * grad_weights

        elif method == 'adam':
            grads = [grad_means, grad_variances, grad_weights]
            adam_optimizer(params, grads, m, v, iteration, lr_means, beta1, beta2, epsilon)
            means, variances, weights = params
            if optimize_variances:
                variances = np.clip(variances, 0.1, 10.0)
            if optimize_weights:
                if signed_weights == False:
                    weights = np.maximum(weights, 1e-8)

        else:
            raise ValueError("Invalid method specified.")

        pbar.update(1)
        if obj_value <= tol:
            print(f"Convergence achieved at iteration {iteration} with objective value {obj_value:.6f}")
            pbar.update(max_iterations - iteration + 1)
            break
    pbar.close()
    return means, variances, weights, objective_values, params_history