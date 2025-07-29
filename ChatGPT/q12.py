import numpy as np

def system(x, y):
    dy1dx = y[1]
    dy2dx = 3 * y[1] - 2 * y[0]
    return np.array([dy1dx, dy2dx])

def rk4_step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h / 2, y + k1 / 2)
    k3 = h * f(x + h / 2, y + k2 / 2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Parameters
h = 0.1
x_values = np.arange(0, 1 + h, h)
steps = len(x_values)

# Initialize arrays
ys = np.zeros((steps, 2))  # columns: y1, y2
ys[0] = [-1, 0]  # y(0) = -1, y'(0) = 0

# Store derivatives for ABM
fs = np.zeros((steps, 2))

# First 3 steps using RK4
for i in range(3):
    fs[i] = system(x_values[i], ys[i])
    ys[i+1] = rk4_step(system, x_values[i], ys[i], h)

fs[3] = system(x_values[3], ys[3])

# Adams-Bashforth-Moulton 4th order
for i in range(3, steps - 1):
    # Predictor (Adams-Bashforth 4)
    yp = ys[i] + h / 24 * (55 * fs[i] - 59 * fs[i - 1] + 37 * fs[i - 2] - 9 * fs[i - 3])

    # Evaluate f at predicted point
    fp = system(x_values[i+1], yp)

    # Corrector (Adams-Moulton 4)
    ys[i+1] = ys[i] + h / 24 * (9 * fp + 19 * fs[i] - 5 * fs[i - 1] + fs[i - 2])

    # Update f for next step
    fs[i+1] = system(x_values[i+1], ys[i+1])

# Print the results
print("x\t\ty")
for i in range(steps):
    print(f"{x_values[i]:.1f}\t\t{ys[i,0]:.6f}")
