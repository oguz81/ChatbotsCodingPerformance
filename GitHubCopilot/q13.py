def runge_kutta_4(f, g, x0, y0, z0, h, steps):
    """Use RK4 to get the first 3 points for Milne's method (gives 4 points total)."""
    xs = [x0]
    ys = [y0]
    zs = [z0]
    x, y, z = x0, y0, z0
    for _ in range(steps):
        k1y = h * g(x, y, z)
        k1z = h * f(x, y, z)

        k2y = h * g(x + h/2, y + k1y/2, z + k1z/2)
        k2z = h * f(x + h/2, y + k1y/2, z + k1z/2)

        k3y = h * g(x + h/2, y + k2y/2, z + k2z/2)
        k3z = h * f(x + h/2, y + k2y/2, z + k2z/2)

        k4y = h * g(x + h, y + k3y, z + k3z)
        k4z = h * f(x + h, y + k3y, z + k3z)

        y += (k1y + 2*k2y + 2*k3y + k4y)/6
        z += (k1z + 2*k2z + 2*k3z + k4z)/6
        x += h

        xs.append(round(x, 10))
        ys.append(y)
        zs.append(z)
    return xs, ys, zs

def milne_method(f, g, x0, y0, z0, h, xn):
    # Use RK4 to get y0, y1, y2, y3
    n_steps = int((xn - x0) / h)
    xs, ys, zs = runge_kutta_4(f, g, x0, y0, z0, h, 3)

    fz = [f(xs[i], ys[i], zs[i]) for i in range(4)]  # z' values
    fy = [g(xs[i], ys[i], zs[i]) for i in range(4)]  # y' values

    for n in range(3, n_steps):
        # Predictor for y: y_{n+1}^p = y_{n-3} + (4h/3) * (2f_{n-2} - f_{n-1} + 2f_n)
        y_pred = ys[n-3] + (4*h/3)*(2*fy[n-2] - fy[n-1] + 2*fy[n])
        z_pred = zs[n-3] + (4*h/3)*(2*fz[n-2] - fz[n-1] + 2*fz[n])

        # Evaluate derivatives at predicted point
        x_next = xs[n] + h
        fy_pred = g(x_next, y_pred, z_pred)
        fz_pred = f(x_next, y_pred, z_pred)

        # Corrector for y: y_{n+1} = y_{n-1} + (h/3)*(fy_{n-1} + 4*fy_n + fy_pred)
        y_corr = ys[n-1] + (h/3)*(fy[n-1] + 4*fy[n] + fy_pred)
        z_corr = zs[n-1] + (h/3)*(fz[n-1] + 4*fz[n] + fz_pred)

        xs.append(round(x_next, 10))
        ys.append(y_corr)
        zs.append(z_corr)
        fy.append(fy_pred)
        fz.append(fz_pred)
    return xs, ys

# Differential equation: y'' - 3y' + 2y = 0
# Let y' = z; so y'' = z'
# => z' = 3z - 2y
def f(x, y, z):
    return 3*z - 2*y
def g(x, y, z):
    return z

if __name__ == "__main__":
    x0 = 0
    y0 = -1
    z0 = 0
    h = 0.1
    xn = 1.0

    xs, ys = milne_method(f, g, x0, y0, z0, h, xn)

    print("x\t\ty")
    for x, y in zip(xs, ys):
        print(f"{x:.1f}\t{y:.6f}")