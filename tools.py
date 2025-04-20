import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple, List, Dict, Any, Optional

class ConstrainedSGD:
    """
    Implementation of Stochastic Gradient Descent with Deterministic Constraints
    based on the methodology described in the paper.
    """
    
    def __init__(
        self,
        initial_positions: np.ndarray,
        objective_gradient_sampler: Callable[[np.ndarray, Any], np.ndarray],
        penalty_gradient: Callable[[np.ndarray], np.ndarray],
        penalty_function: Callable[[np.ndarray], float],
        learning_rate_scheduler: Callable[[int, float, float, int], float],
        initial_learning_rate: float = 0.1,
        final_learning_rate: float = 0.001,
        initial_constraint_multiplier: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.0,  # Default to 0 based on paper findings
        total_iterations: int = 1000,
        samples_per_iteration: int = 10,
        epsilon: float = 1e-8,
        verbose: bool = False
    ):
        """
        Initialize the constrained SGD optimizer.
        
        Args:
            initial_positions: Starting point for optimization (array of design variables)
            objective_gradient_sampler: Function that returns stochastic gradient estimates 
                                        given current positions and optional sampling parameters
            penalty_gradient: Function that computes the gradient of the penalty function
            penalty_function: Function that returns the penalty value (constraint violation)
            learning_rate_scheduler: Function that computes learning rate at each iteration
            initial_learning_rate: Starting learning rate
            final_learning_rate: Target final learning rate
            initial_constraint_multiplier: Starting weight for the penalty function
            beta1: Momentum parameter (1st moment)
            beta2: Second moment parameter
            total_iterations: Total number of optimization iterations
            samples_per_iteration: Number of stochastic samples per iteration
            epsilon: Small constant to prevent division by zero
            verbose: Whether to print progress information
        """
        self.positions = initial_positions.copy()
        self.objective_gradient_sampler = objective_gradient_sampler
        self.penalty_gradient = penalty_gradient
        self.penalty_function = penalty_function
        self.learning_rate_scheduler = learning_rate_scheduler
        
        # Parameters
        self.eta0 = initial_learning_rate
        self.etaT = final_learning_rate
        self.alpha0 = initial_constraint_multiplier
        self.beta1 = beta1
        self.beta2 = beta2
        self.T = total_iterations
        self.K = samples_per_iteration
        self.epsilon = epsilon
        self.verbose = verbose
        
        # Initialize moment estimates
        self.m = np.zeros_like(initial_positions)
        self.v = np.zeros_like(initial_positions)
        
        # Iteration counter
        self.i = 0
        
        # Computation history
        self.history = {
            'positions': [self.positions.copy()],
            'penalty': [self.penalty_function(self.positions)],
            'learning_rates': [],
            'alphas': []
        }
    
    def _calculate_learning_rate(self) -> float:
        """Calculate the current learning rate based on the schedule"""
        return self.learning_rate_scheduler(self.i, self.eta0, self.etaT, self.T)
    
    def _calculate_constraint_multiplier(self, eta_i: float) -> float:
        """Calculate the constraint penalty multiplier for current iteration"""
        return self.alpha0 * (self.eta0 / eta_i)
    
    def step(self, sampling_params: Any = None) -> np.ndarray:
        """
        Perform one optimization step
        
        Args:
            sampling_params: Optional parameters to pass to the objective gradient sampler
                           (could be random seeds, data batches, etc.)
        
        Returns:
            Current positions after the step
        """
        # Calculate current learning rate
        eta_i = self._calculate_learning_rate()
        
        # Calculate current constraint multiplier
        alpha_i = self._calculate_constraint_multiplier(eta_i)
        
        # Compute the penalty gradient
        penalty_grad = self.penalty_gradient(self.positions)
        
        # Sample the objective gradient
        objective_grad = self.objective_gradient_sampler(self.positions, sampling_params)
        
        # Combine gradients
        combined_grad = objective_grad + alpha_i * penalty_grad
        
        # Update first moment estimate (momentum)
        self.m = self.beta1 * self.m + (1 - self.beta1) * combined_grad
        
        # Update second moment estimate if using it
        if self.beta2 > 0:
            self.v = self.beta2 * self.v + (1 - self.beta2) * (combined_grad**2)
            # Bias correction
            m_hat = self.m / (1 - self.beta1**(self.i+1))
            v_hat = self.v / (1 - self.beta2**(self.i+1))
            
            # Update positions
            self.positions -= eta_i * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            # Bias correction for momentum only
            m_hat = self.m / (1 - self.beta1**(self.i+1))
            
            # Normalize by current gradient magnitude (as described in the paper)
            grad_norm = np.linalg.norm(combined_grad) + self.epsilon
            self.positions -= eta_i * m_hat / grad_norm
        
        # Update iteration counter
        self.i += 1
        
        # Record history
        self.history['positions'].append(self.positions.copy())
        self.history['penalty'].append(self.penalty_function(self.positions))
        self.history['learning_rates'].append(eta_i)
        self.history['alphas'].append(alpha_i)
        
        if self.verbose and self.i % 100 == 0:
            penalty = self.history['penalty'][-1]
            print(f"Iteration {self.i}: Penalty = {penalty:.6f}, LR = {eta_i:.6f}, Alpha = {alpha_i:.2f}")
        
        return self.positions
    
    def optimize(self, sampling_params_generator: Optional[Callable[[int], Any]] = None) -> np.ndarray:
        """
        Run the full optimization process for the specified number of iterations
        
        Args:
            sampling_params_generator: Optional function that generates sampling parameters
                                      for each iteration based on the iteration number
        
        Returns:
            Optimized positions
        """
        if self.verbose:
            print(f"Starting optimization with {self.T} iterations")
            print(f"Initial penalty: {self.history['penalty'][0]:.6f}")
        
        for i in range(self.T):
            params = None
            if sampling_params_generator is not None:
                params = sampling_params_generator(i)
            
            self.step(params)
        
        if self.verbose:
            print(f"Optimization complete")
            print(f"Final penalty: {self.history['penalty'][-1]:.6f}")
        
        return self.positions


# Utility functions for common scheduling approaches

def exponential_decay_schedule(iteration: int, initial_value: float, 
                               final_value: float, total_iterations: int) -> float:
    """
    Exponential decay schedule for learning rate
    
    Args:
        iteration: Current iteration
        initial_value: Starting value
        final_value: Target final value
        total_iterations: Total number of iterations
        
    Returns:
        Scheduled value for the current iteration
    """
    if iteration >= total_iterations:
        return final_value
    
    decay_rate = np.log(final_value / initial_value) / (total_iterations - 1)
    return initial_value * np.exp(decay_rate * iteration)


def linear_schedule(iteration: int, initial_value: float, 
                   final_value: float, total_iterations: int) -> float:
    """
    Linear schedule between initial and final values
    """
    if iteration >= total_iterations:
        return final_value
    
    fraction = iteration / (total_iterations - 1)
    return initial_value + fraction * (final_value - initial_value)


# Helper functions for common constraint types

def distance_constraint(positions: np.ndarray, min_distance: float) -> float:
    """
    Minimum distance constraint between points (e.g., for wind turbine spacing)
    Assumes positions is a 2D array with shape (n_points, dimensions)
    
    Returns the penalty value (sum of squared violations)
    """
    n_points = positions.shape[0]
    penalty = 0.0
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.linalg.norm(positions[i] - positions[j])
            violation = max(0, min_distance - dist)
            penalty += violation**2
    
    return penalty

# --- Rosenbrock function and gradient ---
def rosenbrock(pos):
    """Rosenbrock function value""" 
    a, b = 1, 100
    x, y = pos
    return (a - x)**2 + b * (y - x**2)**2

def distance_constraint_gradient(positions: np.ndarray, min_distance: float) -> np.ndarray:
    """
    Gradient of the minimum distance constraint
    
    Returns gradient array of same shape as positions
    """
    n_points = positions.shape[0]
    dimensions = positions.shape[1]
    gradient = np.zeros_like(positions)
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)
            
            if dist < min_distance:
                # Compute gradient direction
                if dist > 1e-10:  # Avoid division by zero
                    direction = diff / dist
                else:
                    # Random direction if points are exactly at same location
                    direction = np.random.randn(dimensions)
                    direction = direction / np.linalg.norm(direction)
                
                violation = min_distance - dist
                grad_factor = 2 * violation
                
                # Update gradient for both points (opposite directions)
                gradient[i] -= grad_factor * direction
                gradient[j] += grad_factor * direction
    
    return gradient


def boundary_constraint(positions: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Boundary constraint to keep points within specified bounds
    bounds is a tuple of (lower_bounds, upper_bounds)
    
    Returns the penalty value (sum of squared violations)
    """
    lower_bounds, upper_bounds = bounds
    lower_violation = np.maximum(0, lower_bounds - positions)
    upper_violation = np.maximum(0, positions - upper_bounds)
    
    return np.sum(lower_violation**2) + np.sum(upper_violation**2)


def boundary_constraint_gradient(positions: np.ndarray, 
                                bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Gradient of the boundary constraint
    """
    lower_bounds, upper_bounds = bounds
    gradient = np.zeros_like(positions)
    
    # Lower bound gradient
    lower_mask = positions < lower_bounds
    gradient[lower_mask] = -2 * (lower_bounds - positions)[lower_mask]
    
    # Upper bound gradient
    upper_mask = positions > upper_bounds
    gradient[upper_mask] = 2 * (positions - upper_bounds)[upper_mask]
    
    return gradient

def scipy_constraint(pos):
    """Returns >= 0 if constraint is satisfied (outside or on circle)."""
    return np.sum((pos - np.ones(2) )**2) - 0.1**2

def get_solution():
    constraints = ({'type': 'ineq', 'fun': scipy_constraint})
    opt_result = minimize(rosenbrock, [0.8, 0.8], method='SLSQP',
                          constraints=constraints, options={'disp': True})
    return opt_result

# Example of circular boundary constraint for wind farm
def circular_boundary_constraint(positions: np.ndarray, center: np.ndarray, radius: float) -> float:
    """
    Circular boundary constraint (e.g., for circular wind farm)
    """
    # Reshape positions if needed to handle both 1D and 2D arrays
    pos_reshaped = positions.reshape(-1, 2)
    n_points = pos_reshaped.shape[0]
    
    penalty = 0.0
    for i in range(n_points):
        dist_to_center = np.linalg.norm(pos_reshaped[i] - center)
        violation = max(0, dist_to_center - radius)
        penalty += violation**2
    
    return penalty


def circular_boundary_gradient(positions: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Gradient of circular boundary constraint
    """
    # Reshape positions if needed
    original_shape = positions.shape
    pos_reshaped = positions.reshape(-1, 2)
    n_points = pos_reshaped.shape[0]
    
    gradient = np.zeros_like(pos_reshaped)
    
    for i in range(n_points):
        diff = pos_reshaped[i] - center
        dist = np.linalg.norm(diff)
        
        if dist > radius:
            if dist > 1e-10:  # Avoid division by zero
                direction = diff / dist
            else:
                direction = np.array([1.0, 0.0])  # Default direction
                
            violation = dist - radius
            gradient[i] += 2 * violation * direction
    
    return gradient.reshape(original_shape)



def custom_run_optimization(x_init, N, l0, beta1, beta2, penalty_type='linear', 
                           penalty_lambda=100.0, use_scheduling=True, eps=1e-8):
    """Custom version that allows enabling/disabling scheduling"""
    x_sgd = np.array(x_init, dtype=float)
    m = np.zeros_like(x_sgd)
    v = np.zeros_like(x_sgd)
    
    # History logging
    obj_hist = np.zeros(N)
    con_hist = np.zeros(N)
    x_hist = np.zeros((N, x_sgd.size))
    
    # Learning rate setup
    if use_scheduling:
        # When scheduling is enabled, use decaying learning rate
        # We'll use a simple form: eta_i = eta_0 / (1 + 0.1*i)
        # This achieves a product form without needing the delta calculation
        learning_rate = l0
        decay_factor = 0.1  # Simpler than calculating delta
    else:
        # When scheduling is disabled, use constant learning rate
        learning_rate = l0
        decay_factor = 0  # No decay
    
    print(f"Running with scheduling: {use_scheduling}, beta1={beta1}, beta2={beta2}, penalty_type={penalty_type}")
    
    for i in range(N):
        # Update current state in history
        obj_hist[i] = f(x_sgd)
        con_hist[i] = max(0, constraint_violation(x_sgd))
        x_hist[i] = x_sgd.copy()
        
        # Calculate combined gradient (objective + penalty)
        if use_scheduling:
            # When scheduling is enabled, we scale the constraint multiplier inversely with learning rate
            alpha_i = 1.0 / (1.0 + i * decay_factor)  # Increasing constraint importance
        else:
            # When scheduling is disabled, use constant constraint importance
            alpha_i = 1.0
        
        # Calculate gradients
        # For simplicity, we'll directly compute gradients here
        # In a real implementation, you'd use the obj_J function from tools.py
        
        # Objective gradient (Rosenbrock)
        a = 1
        b = 100
        dx1 = -2 * a + 4 * b * x_sgd[0] ** 3 - 4 * b * x_sgd[0] * x_sgd[1] + 2 * x_sgd[0]
        dx2 = 2 * b * (x_sgd[1] - x_sgd[0] ** 2)
        obj_grad = np.array([dx1, dx2])
        
        # Constraint gradient
        if constraint_violation(x_sgd) > 0:
            # Only apply constraint gradient if violating constraint
            # grad_v = 2 * (constraint_center - x_sgd)
            dist_vector = constraint_center - x_sgd
            
            if penalty_type == 'linear':
                pen_grad = 2 * penalty_lambda * dist_vector
            else:  # quadratic
                violation = constraint_violation(x_sgd)
                pen_grad = 2 * penalty_lambda * violation * dist_vector
        else:
            pen_grad = np.zeros_like(x_sgd)
        
        # Combined gradient
        j_sgd = obj_grad + alpha_i * pen_grad
        
        # Adam update
        m = beta1 * m + (1 - beta1) * j_sgd
        v = beta2 * v + (1 - beta2) * (j_sgd**2)
        
        # Bias correction
        m_hat = m / (1 - beta1**(i+1))
        v_hat = v / (1 - beta2**(i+1)) if beta2 > 0 else 1.0
        
        # Update position
        if beta2 > 0:
            x_sgd = x_sgd - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        else:
            # For beta2=0, use gradient norm as denominator
            grad_norm = np.linalg.norm(j_sgd) + eps
            x_sgd = x_sgd - learning_rate * m_hat / grad_norm
        
        # Update learning rate for next iteration if using scheduling
        if use_scheduling:
            learning_rate = l0 / (1 + decay_factor * (i+1))
    
    return {
        'obj_hist': obj_hist,
        'con_hist': con_hist,
        'x_hist': x_hist,
        'beta1': beta1,
        'beta2': beta2,
        'penalty_type': penalty_type,
        'use_scheduling': use_scheduling
    }



