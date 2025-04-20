import numpy as np
import matplotlib.pyplot as plt
from tools import ConstrainedSGD, exponential_decay_schedule, get_solution

# --- Improved delta calculation for product schedule ---
def calculate_delta(initial_rate, final_rate, total_iterations, max_iter=100, tol=1e-6):
    """
    Calculate delta parameter for the product schedule:
    η_i = η_0 × ∏(1/(1+δ×k)) for k=0...i-1
    
    Uses binary search to find delta that achieves final_rate at total_iterations.
    """
    if initial_rate <= final_rate:
        return 0.0  # No decay needed if final rate >= initial rate
    
    # Function to compute final rate given delta
    def compute_final_rate(delta):
        rate = initial_rate
        for i in range(total_iterations):
            rate *= 1.0 / (1.0 + delta * i)
        return rate
    
    # Binary search for delta
    left = 0.0
    right = 1.0
    
    # First find an upper bound that decays enough
    while compute_final_rate(right) > final_rate and right < 1e6:
        right *= 2.0
    
    # Binary search in [left, right]
    for _ in range(max_iter):
        mid = (left + right) / 2.0
        rate = compute_final_rate(mid)
        
        if abs(rate - final_rate) < tol:
            return mid
        elif rate > final_rate:
            left = mid
        else:
            right = mid
    
    # Return best estimate
    return (left + right) / 2.0

# --- Our custom learning rate scheduler ---
def product_schedule(iteration, initial_value, final_value, total_iterations, delta=None):
    """
    Product form learning rate scheduler:
    η_i = η_0 × ∏(1/(1+δ×k)) for k=0...i-1
    
    If delta is not provided, it will be calculated to reach final_value at total_iterations.
    """
    if delta is None:
        delta = calculate_delta(initial_value, final_value, total_iterations)
    
    if iteration == 0:
        return initial_value
    
    rate = initial_value
    for i in range(iteration):
        rate *= 1.0 / (1.0 + delta * i)
    
    return rate

# --- No scheduling version (constant rate) ---
def constant_schedule(iteration, initial_value, final_value, total_iterations):
    """Constant learning rate (no scheduling)"""
    return initial_value

# --- Rosenbrock function and gradient ---
def rosenbrock(pos):
    """Rosenbrock function value"""
    a, b = 1, 100
    x, y = pos
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_gradient_sampler(positions, sample_params=None):
    """Rosenbrock gradient with optional noise"""
    noise_level = 0.1 if sample_params is None else sample_params.get('noise_level', 0.1)
    
    # Calculate gradient
    x, y = positions
    a, b = 1, 100
    dx = -2 * a + 4 * b * x**3 - 4 * b * x * y + 2 * x
    dy = 2 * b * (y - x**2)
    grad = np.array([dx, dy])
    
    # Add noise
    if noise_level > 0:
        grad += noise_level * np.random.randn(2)
    
    return grad

# --- Constraint functions ---
constraint_center = np.array([1.0, 1.0])
constraint_radius = 0.1

def constraint_violation(pos):
    """
    Calculate constraint violation (positive when inside forbidden circle)
    Constraint: Points must be OUTSIDE the circle with radius 0.1 at (1,1)
    """
    dist_sq = np.sum((pos - constraint_center)**2)
    radius_sq = constraint_radius**2
    # Constraint is violated if distance^2 < radius^2
    return radius_sq - dist_sq

def get_penalty_funcs(penalty_type, penalty_lambda=100.0):
    """Create penalty function and gradient based on type"""
    if penalty_type == "linear":
        def penalty_func(pos):
            violation = constraint_violation(pos)
            return penalty_lambda * max(0, violation)
        
        def penalty_grad(pos):
            violation = constraint_violation(pos)
            if violation > 0:  # If constraint is violated (inside circle)
                # Gradient direction is outward from center: 2 * (c - x)
                return 2 * penalty_lambda * (constraint_center - pos)
            return np.zeros_like(pos)
    else:  # quadratic
        def penalty_func(pos):
            violation = constraint_violation(pos)
            return (penalty_lambda/2) * max(0, violation)**2
        
        def penalty_grad(pos):
            violation = constraint_violation(pos)
            if violation > 0:  # If constraint is violated (inside circle)
                # Gradient is lambda * violation * 2 * (c - x)
                return 2 * penalty_lambda * violation * (constraint_center - pos)
            return np.zeros_like(pos)
    
    return penalty_func, penalty_grad

# --- Main experiment ---
def main():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Experiment parameters
    N_iterations = 5000
    initial_learning_rate = 0.01
    final_learning_rate = 0.001
    initial_pos = np.array([0.8, 0.5])
    penalty_lambda = 100.0
    
    # Calculate delta for our schedule
    delta = calculate_delta(initial_learning_rate, final_learning_rate, N_iterations)
    print(f"Calculated delta: {delta:.6f}")
    
    # Verify the schedule
    rates = [product_schedule(i, initial_learning_rate, final_learning_rate, 
                             N_iterations, delta) for i in range(0, N_iterations, 100)]
    print(f"Learning rates at iterations [0, 100, 200, ...]: {rates}")
    
    # Define experiments
    experiments = {
        r"Linear penalty with $\beta_1, \beta_2=(0.9, 0.999)$": {
            #"scheduler": exponential_decay_schedule,
            "scheduler": lambda i, i0, iT, T: product_schedule(i, i0, iT, T, delta),
            "beta1": 0.9, "beta2": 0.999, 
            "penalty_type": "linear"
        },
        #"Quadratic penalty with scheduling": {
       # r"Quadratic penalty with $\beta_1, \beta_2=(0.9, 0.999)$": {
       #     #"scheduler": exponential_decay_schedule,
       #     "scheduler": lambda i, i0, iT, T: product_schedule(i, i0, iT, T, delta),
       #     "beta1": 0.9, "beta2": 0.999, 
       #     "penalty_type": "quadratic"
       # },
        r"Linear penalty with $\beta_1, \beta_2=(0.1, 0.2)$": {
            "scheduler": lambda i, i0, iT, T: product_schedule(i, i0, iT, T, delta),
            "beta1": 0.1, "beta2": 0.2, 
            #"beta1": 0.9, "beta2": 0.999, 
            "penalty_type": "linear"
        },
       # #"Quadratic penalty without scheduling": {
       # r"Quadratic penalty with $\beta_1, \beta_2=(0.1, 0.2)$": {
       #     "scheduler": lambda i, i0, iT, T: product_schedule(i, i0, iT, T, delta),
       #     "beta1": 0.9, "beta2": 0.999, 
       #     "penalty_type": "quadratic"
       # }
    }
    
    # Run experiments
    results = {}
    for label, config in experiments.items():
        print(f"\n--- Running: {label} ---")
        
        # Get penalty functions
        penalty_func, penalty_grad = get_penalty_funcs(config["penalty_type"], penalty_lambda)
        
        # Create optimizer
        optimizer = ConstrainedSGD(
            initial_positions=initial_pos,
            objective_gradient_sampler=rosenbrock_gradient_sampler,
            penalty_gradient=penalty_grad,
            penalty_function=penalty_func,
            learning_rate_scheduler=config["scheduler"],
            initial_learning_rate=initial_learning_rate,
            final_learning_rate=final_learning_rate,
            initial_constraint_multiplier=1.0,
            beta1=config["beta1"],
            beta2=config["beta2"],
            total_iterations=N_iterations,
            samples_per_iteration=1,
            verbose=True
        )
        
        # Run optimization
        final_pos = optimizer.optimize()
        
        # Store results
        results[label] = {
            'final_pos': final_pos,
            'x_hist': np.array(optimizer.history['positions']),
            'learning_rates': optimizer.history['learning_rates'],
            'alphas': optimizer.history['alphas'],
            'beta1': config["beta1"],
            'beta2': config["beta2"],
            'penalty_type': config["penalty_type"],
            'use_scheduling': config["scheduler"] != constant_schedule
        }
        
        # Calculate objective history
        positions = np.array(optimizer.history['positions'])
        results[label]['obj_hist'] = np.array([rosenbrock(pos) for pos in positions])
        
        # Calculate constraint violation history
        results[label]['con_hist'] = np.array([max(0, constraint_violation(pos)) 
                                             for pos in positions])
    
    # Create plots
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.5])
    
    # Plot 1: Objective Function Value
    ax1 = fig.add_subplot(gs[0])
    for label, res in results.items():
        obj_hist = res['obj_hist']
        ax1.plot(np.arange(len(obj_hist)), obj_hist, 
                label=f"{label} (Final: {obj_hist[-1]:.3e})")
    ax1.set_ylabel('Objective Value f(x)')
    ax1.set_yscale('log')
    ax1.set_title('Objective Function Convergence')
    ax1.legend(fontsize='small')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    sol = get_solution()
    ax1.axhline(sol.fun, ls='--', c='k')
    
    # Plot 2: Constraint Violation
    ax2 = fig.add_subplot(gs[1])
    for label, res in results.items():
        con_hist = res['con_hist']
        ax2.plot(np.arange(len(con_hist)), con_hist, 
                label=f"{label} (Final: {con_hist[-1]:.3e})")
    ax2.set_ylabel('Constraint Violation')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-9)
    ax2.set_title('Constraint Violation Convergence')
    #ax2.legend(fontsize='small')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Plot 3: Learning Rate Schedule
    ax3 = fig.add_subplot(gs[2])
    for label, res in results.items():
        if 'learning_rates' in res:
            ax3.plot(np.arange(len(res['learning_rates'])), res['learning_rates'], 
                    label=f"{label}")
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.set_title('Learning Rate Schedule')
    #ax3.legend(fontsize='small')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Plot 4: Optimization Paths
    ax4 = fig.add_subplot(gs[3])
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Add Rosenbrock contours
    x = np.linspace(0.0, 1.5, 100)
    y = np.linspace(0.0, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rosenbrock(np.array([X[i, j], Y[i, j]]))
    contour_levels = np.logspace(0, 3, 10)
    cs = ax4.contour(X, Y, Z, levels=contour_levels, colors='lightgray', alpha=0.6)
    
    # Add constraint circle
    constraint_circle = plt.Circle(constraint_center, constraint_radius, 
                                  edgecolor='r', facecolor='r', alpha=0.2,
                                  linestyle='--', linewidth=1.5, 
                                  label='Forbidden Region')
    ax4.add_patch(constraint_circle)
    
    # Plot paths
    for i, (label, res) in enumerate(results.items()):
        path = res['x_hist']
        # Plot every 20th point to reduce clutter
        ax4.plot(path[::20, 0], path[::20, 1], color=colors[i], alpha=0.7, 
                 label=f"{label}")
        ax4.scatter(path[0, 0], path[0, 1], marker='o', s=80, color=colors[i], 
                    edgecolor='black', zorder=5)
        ax4.scatter(path[-1, 0], path[-1, 1], marker='*', s=150, color=colors[i], 
                    edgecolor='black', zorder=5)
    
    ax4.set_xlabel('x[0]')
    ax4.set_ylabel('x[1]')
    ax4.set_xlim(0.0, 1.5)
    ax4.set_ylim(0.0, 1.5)
    ax4.set_title('Optimization Paths')
    ax4.set_aspect('equal')
    ax4.grid(True, linestyle='--', linewidth=0.5)
    ax4.legend(fontsize='small', loc='upper left')
    
    #plt.tight_layout()
    plt.suptitle('Comparing Linear vs Quadratic Penalties With and Without Scheduling', 
                 fontsize=16, y=0.98)
    plt.savefig('beta_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n=== Results Summary ===")
    print("{:<35} {:<15} {:<15}".format(
        "Configuration", "Final Objective", "Final Violation"))
    print("-" * 65)
    
    for label, res in results.items():
        final_obj = res['obj_hist'][-1]
        final_viol = res['con_hist'][-1]
        print("{:<35} {:<15.6e} {:<15.6e}".format(
            label, final_obj, final_viol))

if __name__ == "__main__":
    main()
