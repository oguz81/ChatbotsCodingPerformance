def solve_ode_rk4(f, t_span, y0, h):
    """
    Solves a system of first-order ordinary differential equations using the
    fourth-order Runge-Kutta method.

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
    t = t_start
    y = list(y0)  # Ensure y is mutable
    t_values = [t]
    y_values = [list(y)]

    while t < t_end:
        if t + h > t_end:
            h = t_end - t  # Adjust step size for the last interval

        k1 = [h * val for val in f(t, y)]
        k2 = [h * val for val in f(t + h/2, [y[i] + k1[i]/2 for i in range(len(y))])]
        k3 = [h * val for val in f(t + h/2, [y[i] + k2[i]/2 for i in range(len(y))])]
        k4 = [h * val for val in f(t + h, [y[i] + k3[i] for i in range(len(y))])]

        for i in range(len(y)):
            y[i] = y[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6

        t = t + h
        t_values.append(t)
        y_values.append(list(y))

    return t_values, y_values

def second_order_ode_to_system(t, z):
    """
    Converts the second-order ODE y'' - 3y' + 2y = 0 into a system of two
    first-order ODEs.

    Let z[0] = y and z[1] = y'.
    Then z'[0] = y' = z[1]
    And z'[1] = y'' = 3y' - 2y = 3*z[1] - 2*z[0]

    Args:
        t (float): The current time (not explicitly used in this autonomous ODE).
        z (list): A list [y, y'].

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
    y0 = [-1, 0]  # [y(0), y'(0)]
    h = 0.1

    # Solve the ODE using the fourth-order Runge-Kutta method
    t_values, solution = solve_ode_rk4(f, t_span, y0, h)

    # Print the calculated y values
    print("Calculated y values:")
    for i in range(len(t_values)):
        print(f"t = {t_values[i]:.1f}, y = {solution[i][0]:.6f}")