import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from scipy.stats import multivariate_normal

def kl_divergence(p, q, dx, eps=1e-12):
    return np.sum(np.where(p != 0, p * np.log((p + eps) / (q + eps)), 0)) * dx
def l1_norm(p, q, dx):
    return np.sum(np.abs(p - q)) * dx
def l2_norm(p, q, dx):
    return np.sqrt(np.sum((p - q) ** 2) * dx)

# -------------------- Define density functions --------------------
def gaussian(x, mean, covariance):
    """Compute Gaussian distribution with normalization using scipy.stats."""
    return multivariate_normal.pdf(x, mean=mean, cov=covariance)

def gaussian_mixture(x, means, covariances, weights):
    """Compute a mixture of Gaussians using scipy.stats."""
    weights, means, covariances = map(np.asarray, (weights, means, covariances))
    mixture = sum(w * multivariate_normal.pdf(x, mean=m, cov=c) for w, m, c in zip(weights, means, covariances))
    return mixture / np.sum(weights)

# -------------------- Gradients --------------------
def grad_q_wrt_mu(x, means, variances, weights, i):
    """Compute gradient w.r.t mean i."""
    g_i = weights[i] * gaussian(x, means[i], variances[i])
    diff = (x - means[i])/variances[i]
    if means.shape[1] == 1:
        return g_i* diff/ np.sum(weights)
    else:
        return g_i[:, np.newaxis] * diff/ np.sum(weights)

def grad_q_wrt_variance(x, means, variances, weights, i):
    """Compute gradient w.r.t variance i."""
    g_i = weights[i] * gaussian(x, means[i], variances[i])
    diff_squared = (x - means[i]) ** 2
    if means.shape[1] == 1:
        return (g_i * (diff_squared / (2 * variances[i] ** 2) - 1 / (2 * variances[i]))) / np.sum(weights)
    else:
        return (g_i[:, np.newaxis] * (diff_squared / (2 * variances[i] ** 2) - 1 / (2 * variances[i])))/ np.sum(weights)

def grad_q_wrt_weight(x, means, variances, weights, i, model):
    """Compute gradient w.r.t weight w_i."""
    g_i = gaussian(x, means[i], variances[i])
    total_w = np.sum(weights)
    return (g_i - model) / total_w

def gradient_descent(dx, target_density, model_density, initial_means, initial_variances, initial_weights,
                     optimize_means=True, optimize_variances=False, optimize_weights=False, signed_weights=False,
                     objective='kl', learning_rate=0.01, var_clip=0.1, weight_clip=0.1,
                     tol=1e-4, max_iterations=1000, method='standard', verbose=False, x=None,
                     beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Optimize a Gaussian mixture model to approximate a target density in a multivariate setting.
    """
    if x is None:
        raise ValueError("A valid domain x must be provided for the optimization.")

    # Flatten the meshgrid for computations
    x_flat = np.array([grid.ravel() for grid in x]).T if isinstance(x, list) else x

    means, variances, weights = initial_means.copy(), initial_variances.copy(), initial_weights.copy()
    N = len(means)

    objective_values = []
    l2_values = []
    params_history = []

    target = target_density(x_flat)

    print(f"Starting optimization using {objective.upper()} with {method} gradient descent...")
    pbar = tqdm(total=max_iterations)

    optimization_sequence = []
    if optimize_means:
        optimization_sequence.append('means')
    if optimize_variances:
        optimization_sequence.append('variances')
    if optimize_weights:
        optimization_sequence.append('weights')

    if method == 'adam':
        params = [means, variances, weights]
        m = [np.zeros_like(param) for param in params]
        v = [np.zeros_like(param) for param in params]

    for iteration in range(1, max_iterations + 1):
        if not signed_weights:
            weights = np.maximum(weights, 1e-8)

        model = model_density(x_flat, means, variances, weights)
        
        if objective == 'kl':
            obj_value = kl_divergence(target, model, dx)
            l2_values.append(l2_norm(target, model, dx))
            factor = -target / np.clip(model, 1e-8, None)
        elif objective == 'l1':
            obj_value = l1_norm(target, model, dx)
            l2_values.append(l2_norm(target, model, dx))
            factor = np.sign(model - target)
        elif objective == 'l2':
            obj_value = l2_norm(target, model, dx)
            l2_values.append(obj_value)
            factor = 2 * (model - target)
        else:
            raise ValueError("Invalid objective function specified.")

        objective_values.append(obj_value)
        params_history.append((means.copy(), variances.copy(), weights.copy()))

        if obj_value <= tol:
            print(f"Convergence achieved at iteration {iteration} with objective value {obj_value:.6f}")
            pbar.update(max_iterations - iteration + 1)
            break

        grad_means = np.zeros_like(means)
        grad_variances = np.zeros_like(variances)
        grad_weights = np.zeros_like(weights)

        for i in range(N):
            
            if optimize_means:
                dq_dmu_i = grad_q_wrt_mu(x_flat, means, variances, weights, i)
                if means.shape[1] == 1:
                    grad_means[i] = np.sum(factor * dq_dmu_i) * dx
                else:
                    grad_means[i] = np.sum(factor[:, np.newaxis] * dq_dmu_i, axis=0) * dx
                
            if optimize_variances:
                dq_dvar_i = grad_q_wrt_variance(x_flat, means, variances, weights, i)
                if means.shape[1] == 1:
                    grad_variances[i] = np.sum(factor * dq_dvar_i) * dx
                else:
                    grad_variances[i] = np.sum(factor[:, np.newaxis] * dq_dvar_i, axis=0) * dx

            if optimize_weights:
                dq_dwi = grad_q_wrt_weight(x_flat, means, variances, weights, i, model)
                grad_weights[i] = np.sum(factor * dq_dwi) * dx
                
        if verbose and iteration % 500 == 0:
            print(f"Iteration {iteration}: Obj={obj_value:.6f}, "
                  f"L2 error: {l2_values[-1]:.6f}, "
                  f"||grad_means||={np.linalg.norm(grad_means):.6f}, "
                  f"||grad_variances||={np.linalg.norm(grad_variances):.6f}, "
                  f"||grad_weights||={np.linalg.norm(grad_weights):.6f}")

        # Parameter updates
        if method == 'standard':
            if optimize_means:
                means -= learning_rate * grad_means
            if optimize_variances:
                variances = np.clip(variances - learning_rate * grad_variances, var_clip, 10.0)
            if optimize_weights:
                if not signed_weights:
                    weights = np.clip(weights - learning_rate * grad_weights, weight_clip, 1.0)
                else:
                    weights -= learning_rate * grad_weights

        elif method == 'alternating':
            for param_to_update in optimization_sequence:
                if param_to_update == 'means' and optimize_means:
                    means -= learning_rate[0] * grad_means
                elif param_to_update == 'variances' and optimize_variances:
                    variances = np.clip(variances - learning_rate[1] * grad_variances, var_clip, 10.0)
                elif param_to_update == 'weights' and optimize_weights:
                    if not signed_weights:
                        weights = np.clip(weights - learning_rate[2] * grad_weights, weight_clip, 1.0)
                    else:
                        weights -= learning_rate[2] * grad_weights

        elif method == 'adam':
            grads = [grad_means, grad_variances, grad_weights]
            adam_optimizer([means, variances, weights], grads, m, v, iteration, learning_rate, beta1, beta2, epsilon)
            if optimize_variances:
                variances = np.clip(variances, var_clip, 10.0)
            if optimize_weights and not signed_weights:
                weights = np.clip(weights, weight_clip, 1.0)
        else:
            raise ValueError("Invalid method specified.")

        pbar.update(1)

    pbar.close()
    return means, variances, weights, objective_values, l2_values, params_history

def adam_optimizer(params, grads, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
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