import numpy as np

def runge_kutta_4th_order(f, y0, t0, tf, h):
    """
    Solve a system of first-order ODEs using the fourth-order Runge-Kutta method.
    
    Args:
        f: Function that defines the system of ODEs (dy/dt = f(t, y))
        y0: Initial values
        t0: Initial time
        tf: Final time
        h: Step size
    
    Returns:
        tuple: (t_values, y_values) where t_values is the array of time points and
               y_values is the array of solution values
    """
    # Calculate the number of steps
    n_steps = int((tf - t0) / h)
    
    # Initialize arrays to store the results
    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros((n_steps + 1, len(y0)))
    
    # Set initial values
    t_values[0] = t0
    y_values[0] = y0
    
    # Runge-Kutta iterations
    for i in range(n_steps):
        t = t_values[i]
        y = y_values[i]
        
        # Compute the four RK increments
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)
        
        # Update for the next step
        t_values[i+1] = t + h
        y_values[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t_values, y_values

def ode_system(t, y):
    """
    Define the system of first-order ODEs equivalent to y'' - 3y' + 2y = 0.
    
    For second-order ODE, we introduce two variables:
    y[0] = y (the original function)
    y[1] = y' (the first derivative)
    
    Then the system becomes:
    y'[0] = y[1]
    y'[1] = 3*y[1] - 2*y[0]
    
    Args:
        t: Time (independent variable)
        y: Vector of dependent variables [y, y']
        
    Returns:
        numpy.ndarray: Vector of derivatives [y', y'']
    """
    # Return [y', y'']
    return np.array([y[1], 3*y[1] - 2*y[0]])

def main():
    # Initial conditions
    y0 = np.array([-1.0, 0.0])  # y(0) = -1, y'(0) = 0
    
    # Time interval and step size
    t0 = 0.0
    tf = 1.0
    h = 0.1
    
    # Solve using RK4
    t_values, y_values = runge_kutta_4th_order(ode_system, y0, t0, tf, h)
    
    # Print results
    print(f"{'t':^10}{'y':^15}")
    print("-" * 25)
    for i in range(len(t_values)):
        print(f"{t_values[i]:10.1f}{y_values[i, 0]:15.10f}")
    
    # Calculate the exact solution for verification
    # For this ODE, the general solution is y = C1*e^x + C2*e^2x
    # With the given initial conditions, y = -e^x + e^2x
    exact_values = -np.exp(t_values) + np.exp(2*t_values)
    
    # Check accuracy
    print("\nComparison with exact solution:")
    print(f"{'t':^10}{'Numerical y':^20}{'Exact y':^20}{'Error':^15}")
    print("-" * 65)
    for i in range(len(t_values)):
        error = abs(y_values[i, 0] - exact_values[i])
        print(f"{t_values[i]:10.1f}{y_values[i, 0]:20.10f}{exact_values[i]:20.10f}{error:15.10e}")

if __name__ == "__main__":
    main()