import numpy as np

def solve_ode_milne(f, t_span, y0, h):
    """
    Solves a system of first-order ordinary differential equations using the
    Milne predictor-corrector method.

    Args:
        f (function): A function that takes time t and state y (a list or array)
                      and returns the derivatives dy/dt (a list or array).
        t_span (tuple): A tuple (t_start, t_end) defining the time interval.
        y0 (list or array): The initial conditions for the state variables.
        h (float): The step size.

    Returns:
        tuple: A tuple containing two lists:
               - t_values: A list of time points.
               - y_values: A list of corresponding state values (each element
                           is a list or array representing the state at that time).
    """
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + h, h)
    n_steps = len(t_values)
    n_eq = len(y0)
    y_values = np.zeros((n_steps, n_eq))
    y_values[0] = y0

    # Use a suitable starting method (e.g., RK4) to get the first few steps
    def rk4_step(t, y, h):
        k1 = np.array(f(t, y))
        k2 = np.array(f(t + h/2, y + h/2 * k1))
        k3 = np.array(f(t + h/2, y + h/2 * k2))
        k4 = np.array(f(t + h, y + h * k3))
        return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    if n_steps >= 4:
        for i in range(1, 4):
            y_values[i] = rk4_step(t_values[i-1], y_values[i-1], h)

        # Milne predictor-corrector method
        for i in range(3, n_steps - 1):
            # Predictor (Milne predictor)
            f_n_minus_3 = np.array(f(t_values[i-3], y_values[i-3]))
            f_n_minus_2 = np.array(f(t_values[i-2], y_values[i-2]))
            f_n_minus_1 = np.array(f(t_values[i-1], y_values[i-1]))

            y_predict = y_values[i-3] + (4*h/3) * (2*f_n_minus_2 - f_n_minus_1 + 2*f_n)
            # Note: We use f_n here, which requires evaluating the derivative at the current step.
            # A pure Milne predictor would use previous values. This is a common modification.
            f_predict_n_plus_1 = np.array(f(t_values[i+1], y_predict))

            # Corrector (Milne corrector)
            f_n_minus_1 = np.array(f(t_values[i-1], y_values[i-1]))
            f_n = np.array(f(t_values[i], y_values[i]))

            y_values[i+1] = y_values[i-1] + (h/3) * (f_n_minus_1 + 4*f_n + f_predict_n_plus_1)

    elif n_steps > 1:
        # If fewer than 4 steps, just use RK4 for the entire interval
        for i in range(1, n_steps):
            y_values[i] = rk4_step(t_values[i-1], y_values[i-1], h)

    return t_values, y_values

def second_order_ode_to_system(t, z):
    """
    Converts the second-order ODE y'' - 3y' + 2y = 0 into a system of two
    first-order ODEs.

    Let z[0] = y and z[1] = y'.
    Then z'[0] = y' = z[1]
    And z'[1] = y'' = 3y' - 2y = 3*z[1] - 2*z[0]

    Args:
        t (float): The current time.
        z (list or array): A list or array [y, y'].

    Returns:
        list: A list [y', y''] representing the derivatives.
    """
    y = z[0]
    yp = z[1]
    ypp = 3*yp - 2*y
    return [yp, ypp]

if __name__ == "__main__":
    # Define the ODE system
    f = second_order_ode_to_system

    # Define the time interval, initial conditions, and step size
    t_span = (0, 1)
    y0 = np.array([-1.0, 0.0])  # [y(0), y'(0)]
    h = 0.1

    # Solve the ODE using the Milne predictor-corrector method
    t_values, solution = solve_ode_milne(f, t_span, y0, h)

    # Print the calculated y values
    print("Calculated y values:")
    for i in range(len(t_values)):
        print(f"t = {t_values[i]:.1f}, y = {solution[i][0]:.6f}")