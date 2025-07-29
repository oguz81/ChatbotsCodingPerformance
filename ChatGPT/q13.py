import numpy as np

def system(x, y):
    dy1dx = y[1]
    dy2dx = 3 * y[1] - 2 * y[0]
    return np.array([dy1dx, dy2dx])

def rk4_step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, y + k2/2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Parameters
h = 0.1
x_values = np.arange(0, 1 + h, h)
steps = len(x_values)

# Initialize solution and derivative arrays
ys = np.zeros((steps, 2))  # y1, y2
fs = np.zeros((steps, 2))  # derivatives

# Initial conditions
ys[0] = [-1, 0]

# Use RK4 for first 3 steps
for i in range(3):
    fs[i] = system(x_values[i], ys[i])
    ys[i+1] = rk4_step(system, x_values[i], ys[i], h)

fs[3] = system(x_values[3], ys[3])

# Milne Predictor-Corrector
for i in range(3, steps - 1):
    # Predictor (Milne's 4th-order predictor)
    yp = ys[i-3] + 4 * h / 3 * (2 * fs[i-2] - fs[i-1] + 2 * fs[i])
    
    # Evaluate f at predicted point
    fp = system(x_values[i+1], yp)
    
    # Corrector (Milne's corrector)
    ys[i+1] = ys[i-1] + h / 3 * (fs[i-1] + 4 * fs[i] + fp)
    
    # Update f for next step
    fs[i+1] = system(x_values[i+1], ys[i+1])

# Print the results
print("x\t\ty")
for i in range(steps):
    print(f"{x_values[i]:.1f}\t\t{ys[i,0]:.6f}")
