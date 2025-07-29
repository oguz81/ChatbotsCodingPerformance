def runge_kutta_4th_order():
    """
    Solves the ODE y'' - 3y' + 2y = 0 using 4th order Runge-Kutta method
    with initial conditions y(0) = -1, y'(0) = 0 in the interval [0,1] with h=0.1
    """
    # Define the ODE system (converted to first-order system)
    def f(t, y, dy):
        d2y = 3*dy - 2*y  # From y'' = 3y' - 2y
        return d2y
    
    # Initial conditions
    t0 = 0.0
    y0 = -1.0
    dy0 = 0.0
    h = 0.1
    t_end = 1.0
    
    # Initialize lists to store results
    t_values = [t0]
    y_values = [y0]
    dy_values = [dy0]
    
    # RK4 implementation
    t = t0
    y = y0
    dy = dy0
    
    print("t\t\ty(t)")
    print(f"{t:.1f}\t\t{y:.6f}")
    
    while t < t_end:
        # RK4 coefficients for y' (first derivative)
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
        
        # Update t
        t += h
        
        # Store values
        t_values.append(t)
        y_values.append(y_new)
        dy_values.append(dy_new)
        
        # Print current y value
        print(f"{t:.1f}\t\t{y_new:.6f}")
        
        # Update for next iteration
        y = y_new
        dy = dy_new

# Run the solver
if __name__ == "__main__":
    runge_kutta_4th_order()