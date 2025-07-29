def runge_kutta_4(f, g, x0, y0, z0, h, steps):
    """Use RK4 to get the first 3 points for Adams-Bashforth-Moulton method."""
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

def adams_bashforth_moulton(f, g, x0, y0, z0, h, xn):
    # We'll use the 4-step method, so we need 3 additional starting points from RK4
    n_steps = int((xn - x0) / h)
    ab4_coeffs = [55, -59, 37, -9]
    am4_coeffs = [9, 19, -5, 1]

    # Get first 3 steps (so 4 points) from RK4
    xs, ys, zs = runge_kutta_4(f, g, x0, y0, z0, h, 3)

    # Store derivatives for y' and z' at each point
    fz = [f(xs[i], ys[i], zs[i]) for i in range(4)]  # z' = f(x, y, z)
    fy = [g(xs[i], ys[i], zs[i]) for i in range(4)]  # y' = g(x, y, z) = z

    for n in range(3, n_steps):
        # Predictor (Adams-Bashforth 4th order)
        yp = ys[-1] + h/24 * sum(c*fy[-i] for i, c in enumerate(ab4_coeffs))
        zp = zs[-1] + h/24 * sum(c*fz[-i] for i, c in enumerate(ab4_coeffs))
        xp = xs[-1] + h

        # Evaluate derivatives at predicted point
        fyp = g(xp, yp, zp)
        fzp = f(xp, yp, zp)

        # Corrector (Adams-Moulton 4th order)
        ycorr = ys[-1] + h/24 * (am4_coeffs[0]*fyp + am4_coeffs[1]*fy[-1] + 
                                 am4_coeffs[2]*fy[-2] + am4_coeffs[3]*fy[-3])
        zcorr = zs[-1] + h/24 * (am4_coeffs[0]*fzp + am4_coeffs[1]*fz[-1] +
                                 am4_coeffs[2]*fz[-2] + am4_coeffs[3]*fz[-3])

        xs.append(round(xp, 10))
        ys.append(ycorr)
        zs.append(zcorr)
        # Update derivative lists (keep last 4)
        fy.append(fyp)
        fy = fy[-4:]
        fz.append(fzp)
        fz = fz[-4:]
    return xs, ys

# For y'' - 3y' + 2y = 0,
# let y' = z, so y'' = z'
# z' = 3z - 2y
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

    xs, ys = adams_bashforth_moulton(f, g, x0, y0, z0, h, xn)

    print("x\t\ty")
    for x, y in zip(xs, ys):
        print(f"{x:.1f}\t{y:.6f}")