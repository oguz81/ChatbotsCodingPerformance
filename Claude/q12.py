import numpy as np

def ode_system(t, y):
    """
    Define the system of first-order ODEs equivalent to y'' - 3y' + 2y = 0.
    
    For this second-order ODE, we use:
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
    return np.array([y[1], 3*y[1] - 2*y[0]])

def runge_kutta_4th_order(f, y, t, h):
    """
    Perform a single step of the fourth-order Runge-Kutta method.
    
    Args:
        f: Function that defines the system of ODEs
        y: Current values of dependent variables
        t: Current time
        h: Step size
        
    Returns:
        numpy.ndarray: New values of dependent variables after step
    """
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5*h, y + 0.5*k1)
    k3 = h * f(t + 0.5*h, y + 0.5*k2)
    k4 = h * f(t + h, y + k3)
    
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def adams_bashforth_4(f, y_prev, t_prev, h):
    """
    Adams-Bashforth 4th-order predictor.
    
    Args:
        f: Function that defines the system of ODEs
        y_prev: List of previous values [y_n, y_n-1, y_n-2, y_n-3]
        t_prev: List of previous times [t_n, t_n-1, t_n-2, t_n-3]
        h: Step size
        
    Returns:
        numpy.ndarray: Predicted value of y at the next step
    """
    f_n = f(t_prev[0], y_prev[0])
    f_n1 = f(t_prev[1], y_prev[1])
    f_n2 = f(t_prev[2], y_prev[2])
    f_n3 = f(t_prev[3], y_prev[3])
    
    # Adams-Bashforth 4th-order formula
    return y_prev[0] + h * (55*f_n - 59*f_n1 + 37*f_n2 - 9*f_n3) / 24

def adams_moulton_3(f, y_pred, t_next, y_prev, t_prev, h):
    """
    Adams-Moulton 3rd-order corrector.
    
    Args:
        f: Function that defines the system of ODEs
        y_pred: Predicted value of y at the next step
        t_next: Time at the next step
        y_prev: List of previous values [y_n, y_n-1]
        t_prev: List of previous times [t_n, t_n-1]
        h: Step size
        
    Returns:
        numpy.ndarray: Corrected value of y at the next step
    """
    f_next = f(t_next, y_pred)
    f_n = f(t_prev[0], y_prev[0])
    f_n1 = f(t_prev[1], y_prev[1])
    
    # Adams-Moulton 3rd-order formula
    return y_prev[0] + h * (9*f_next + 19*f_n - 5*f_n1) / 24

def solve_ode_abm(f, y0, t0, tf, h):
    """
    Solve a system of ODEs using the Adams-Bashforth-Moulton method.
    
    Args:
        f: Function that defines the system of ODEs
        y0: Initial values
        t0: Initial time
        tf: Final time
        h: Step size
        
    Returns:
        tuple: (t_values, y_values) - arrays of time points and solution values
    """
    # Calculate the number of steps
    n_steps = int((tf - t0) / h)
    
    # Initialize arrays to store the results
    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros((n_steps + 1, len(y0)))
    
    # Set initial values
    t_values[0] = t0
    y_values[0] = y0
    
    # Use RK4 for the first 4 steps to bootstrap ABM
    for i in range(min(4, n_steps)):
        y_values[i+1] = runge_kutta_4th_order(f, y_values[i], t_values[i], h)
        t_values[i+1] = t_values[i] + h
    
    # Use ABM for the remaining steps
    for i in range(4, n_steps):
        # Adams-Bashforth predictor
        y_prev = [y_values[i], y_values[i-1], y_values[i-2], y_values[i-3]]
        t_prev = [t_values[i], t_values[i-1], t_values[i-2], t_values[i-3]]
        y_pred = adams_bashforth_4(f, y_prev, t_prev, h)
        
        # Adams-Moulton corrector
        t_next = t_values[i] + h
        y_corr = adams_moulton_3(f, y_pred, t_next, [y_values[i], y_values[i-1]], 
                                 [t_values[i], t_values[i-1]], h)
        
        # Update for the next step
        t_values[i+1] = t_next
        y_values[i+1] = y_corr
    
    return t_values, y_values

def main():
    # Initial conditions
    y0 = np.array([-1.0, 0.0])  # y(0) = -1, y'(0) = 0
    
    # Time interval and step size
    t0 = 0.0
    tf = 1.0
    h = 0.1
    
    # Solve using Adams-Bashforth-Moulton
    t_values, y_values = solve_ode_abm(ode_system, y0, t0, tf, h)
    
    # Print results
    print("Solving y'' - 3y' + 2y = 0 with y(0) = -1, y'(0) = 0")
    print("Using Adams-Bashforth-Moulton Method with h =", h)
    print(f"\n{'t':^10}{'y':^15}{'y\'':^15}")
    print("-" * 40)
    for i in range(len(t_values)):
        print(f"{t_values[i]:10.1f}{y_values[i, 0]:15.10f}{y_values[i, 1]:15.10f}")
    
    # Calculate the exact solution for verification
    # For this ODE, the general solution is y = C1*e^x + C2*e^2x
    # With the given initial conditions, y = -e^x + e^2x
    exact_values = -np.exp(t_values) + np.exp(2*t_values)
    exact_deriv = -np.exp(t_values) + 2*np.exp(2*t_values)
    
    # Check accuracy
    print("\nComparison with exact solution:")
    print(f"{'t':^10}{'Numerical y':^20}{'Exact y':^20}{'Error':^15}")
    print("-" * 65)
    for i in range(len(t_values)):
        error = abs(y_values[i, 0] - exact_values[i])
        print(f"{t_values[i]:10.1f}{y_values[i, 0]:20.10f}{exact_values[i]:20.10f}{error:15.10e}")

if __name__ == "__main__":
    main()