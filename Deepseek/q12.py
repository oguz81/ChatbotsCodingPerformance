def adams_bashforth_moulton():
    """
    Solves the ODE y'' - 3y' + 2y = 0 using Adams-Bashforth-Moulton method
    with initial conditions y(0) = -1, y'(0) = 0 in the interval [0,1] with h=0.1
    """
    # Convert the second-order ODE to a system of first-order ODEs
    def f(t, y, dy):
        return 3*dy - 2*y  # y'' = 3y' - 2y
    
    # Initial conditions
    t0 = 0.0
    y0 = -1.0
    dy0 = 0.0
    h = 0.1
    t_end = 1.0
    
    # First we need starting values from Runge-Kutta (4 steps for 4th-order ABM)
    # We'll use RK4 to get the first 4 points
    def rk4_step(t, y, dy, h):
        # RK4 coefficients for y
        k1_y = dy
        k1_dy = f(t, y, dy)
        
        k2_y = dy + 0.5*h*k1_dy
        k2_dy = f(t + 0.5*h, y + 0.5*h*k1_y, dy + 0.5*h*k1_dy)
        
        k3_y = dy + 0.5*h*k2_dy
        k3_dy = f(t + 0.5*h, y + 0.5*h*k2_y, dy + 0.5*h*k2_dy)
        
        k4_y = dy + h*k3_dy
        k4_dy = f(t + h, y + h*k3_y, dy + h*k3_dy)
        
        # Update y and dy
        y_new = y + (h/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
        dy_new = dy + (h/6)*(k1_dy + 2*k2_dy + 2*k3_dy + k4_dy)
        
        return y_new, dy_new
    
    # Initialize arrays to store values
    t_values = [t0]
    y_values = [y0]
    dy_values = [dy0]
    f_values = [f(t0, y0, dy0)]
    
    # Use RK4 to get first 4 points (including initial condition)
    for _ in range(3):
        t = t_values[-1]
        y, dy = y_values[-1], dy_values[-1]
        y_new, dy_new = rk4_step(t, y, dy, h)
        
        t_values.append(t + h)
        y_values.append(y_new)
        dy_values.append(dy_new)
        f_values.append(f(t + h, y_new, dy_new))
    
    # Adams-Bashforth-Moulton predictor-corrector method
    print("t\t\ty(t)")
    for i in range(len(t_values)):
        print(f"{t_values[i]:.1f}\t\t{y_values[i]:.6f}")
    
    while t_values[-1] < t_end:
        # Current index
        n = len(t_values) - 1
        
        # Adams-Bashforth predictor (4th order)
        t_p = t_values[-1] + h
        y_p = y_values[-1] + h*(55*dy_values[-1] - 59*dy_values[-2] + 37*dy_values[-3] - 9*dy_values[-4])/24
        dy_p = dy_values[-1] + h*(55*f_values[-1] - 59*f_values[-2] + 37*f_values[-3] - 9*f_values[-4])/24
        f_p = f(t_p, y_p, dy_p)
        
        # Adams-Moulton corrector (4th order)
        y_c = y_values[-1] + h*(9*dy_p + 19*dy_values[-1] - 5*dy_values[-2] + dy_values[-3])/24
        dy_c = dy_values[-1] + h*(9*f_p + 19*f_values[-1] - 5*f_values[-2] + f_values[-3])/24
        f_c = f(t_p, y_c, dy_c)
        
        # Update values (using corrected values)
        t_values.append(t_p)
        y_values.append(y_c)
        dy_values.append(dy_c)
        f_values.append(f_c)
        
        # Print the result
        print(f"{t_p:.1f}\t\t{y_c:.6f}")
        
        # Keep only the last 4 values for next step (for 4th-order method)
        if len(t_values) > 4:
            t_values = t_values[-4:]
            y_values = y_values[-4:]
            dy_values = dy_values[-4:]
            f_values = f_values[-4:]

# Run the solver
if __name__ == "__main__":
    adams_bashforth_moulton()