def runge_kutta_4th_order(f, g, x0, y0, z0, h, xn):
    """
    Solve the system:
        y' = z
        z' = f(x, y, z) = 3z - 2y

    Parameters:
        f: function for z' = f(x, y, z)
        g: function for y' = g(x, y, z) (usually just returns z)
        x0: initial x value
        y0: initial y value
        z0: initial y' value
        h: step size
        xn: last x value

    Returns:
        (xs, ys): lists of x and y values
    """
    xs = [x0]
    ys = [y0]
    zs = [z0]

    n_steps = int((xn - x0) / h)
    x, y, z = x0, y0, z0

    for _ in range(n_steps):
        k1y = h * g(x, y, z)
        k1z = h * f(x, y, z)

        k2y = h * g(x + h / 2, y + k1y / 2, z + k1z / 2)
        k2z = h * f(x + h / 2, y + k1y / 2, z + k1z / 2)

        k3y = h * g(x + h / 2, y + k2y / 2, z + k2z / 2)
        k3z = h * f(x + h / 2, y + k2y / 2, z + k2z / 2)

        k4y = h * g(x + h, y + k3y, z + k3z)
        k4z = h * f(x + h, y + k3y, z + k3z)

        y += (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        z += (k1z + 2 * k2z + 2 * k3z + k4z) / 6
        x += h

        xs.append(round(x, 10))  # prevent floating point issues
        ys.append(y)
        zs.append(z)

    return xs, ys

# Define the system based on y'' - 3y' + 2y = 0
# Let y' = z, so y'' = z'
# Then: z' = 3z - 2y
def f(x, y, z):
    return 3 * z - 2 * y

def g(x, y, z):
    return z

if __name__ == "__main__":
    # Initial conditions
    x0 = 0
    y0 = -1
    z0 = 0
    h = 0.1
    xn = 1.0

    xs, ys = runge_kutta_4th_order(f, g, x0, y0, z0, h, xn)

    print("x\t\ty")
    for x, y in zip(xs, ys):
        print(f"{x:.1f}\t{y:.6f}")