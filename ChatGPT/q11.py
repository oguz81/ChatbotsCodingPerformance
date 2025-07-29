import numpy as np

def rk4_system(f, x0, y0, h, steps):
    """
    Solves a system of ODEs using 4th order Runge-Kutta method.
    f: function returning derivatives as np.array([dy1/dx, dy2/dx])
    x0: initial x
    y0: initial np.array([y1, y2])
    h: step size
    steps: number of steps
    """
    x = x0
    y = y0
    results = [(x, y[0])]
    
    for _ in range(steps):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x = x + h
        results.append((x, y[0]))
        
    return results

def system(x, y):
    """
    Defines the system:
    dy1/dx = y2
    dy2/dx = 3*y2 - 2*y1
    """
    dy1dx = y[1]
    dy2dx = 3*y[1] - 2*y[0]
    return np.array([dy1dx, dy2dx])

# Initial conditions
x0 = 0
y0 = np.array([-1, 0])  # y(0) = -1, y'(0) = 0
h = 0.1
steps = int(1 / h)

# Solve
results = rk4_system(system, x0, y0, h, steps)

# Print results
print("x\t\ty")
for x, y1 in results:
    print(f"{x:.1f}\t\t{y1:.6f}")
