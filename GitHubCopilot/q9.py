def calculate_eigenvalues(matrix):
    """
    Calculate the eigenvalues of a 3x3 matrix by solving the characteristic polynomial.

    Args:
        matrix (list): A 3x3 matrix.

    Returns:
        list: Eigenvalues of the matrix.
    """
    # Extracting elements from the matrix
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    # Characteristic polynomial coefficients: λ^3 - (trace)*λ^2 + (determinant of minors)*λ - determinant = 0
    trace = a + e + i
    determinant = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    minor_sum = (
        a * (e + i) + e * (a + i) + i * (a + e)
        - (b * d + c * g + f * h)
    )

    # Coefficients of the cubic polynomial λ^3 - trace*λ^2 + minor_sum*λ - determinant = 0
    coeffs = [1, -trace, minor_sum, -determinant]

    # Finding the roots of the cubic equation (eigenvalues)
    eigenvalues = solve_cubic(coeffs)
    return eigenvalues


def solve_cubic(coeffs):
    """
    Solve a cubic equation ax^3 + bx^2 + cx + d = 0 using Cardano's method.

    Args:
        coeffs (list): Coefficients [a, b, c, d] of the cubic equation.

    Returns:
        list: Real roots of the cubic equation.
    """
    a, b, c, d = coeffs
    # Normalize coefficients
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

    # Discriminant
    discriminant = (q / 2)**2 + (p / 3)**3

    roots = []
    if discriminant > 0:
        # One real root
        u = (-q / 2 + discriminant**0.5)**(1 / 3)
        v = (-q / 2 - discriminant**0.5)**(1 / 3)
        root1 = u + v - b / (3 * a)
        roots.append(root1)
    elif discriminant == 0:
        # Triple or double root
        u = (-q / 2)**(1 / 3)
        root1 = 2 * u - b / (3 * a)
        root2 = -u - b / (3 * a)
        roots.extend([root1, root2])
    else:
        # Three real roots
        r = (-p / 3)**0.5
        theta = math.acos(-q / (2 * r**3))
        for k in range(3):
            root = 2 * r * math.cos((theta + 2 * k * math.pi) / 3) - b / (3 * a)
            roots.append(root)

    return roots


def calculate_eigenvectors(matrix, eigenvalues):
    """
    Calculate the eigenvectors of a 3x3 matrix based on its eigenvalues.

    Args:
        matrix (list): A 3x3 matrix.
        eigenvalues (list): Eigenvalues of the matrix.

    Returns:
        list: Eigenvectors of the matrix.
    """
    eigenvectors = []
    for eigenvalue in eigenvalues:
        # Solve (A - λI)x = 0 for eigenvector x
        A_minus_lambda_I = [
            [matrix[i][j] - (eigenvalue if i == j else 0) for j in range(3)]
            for i in range(3)
        ]
        # Find a non-trivial solution for the system
        eigenvector = solve_homogeneous(A_minus_lambda_I)
        eigenvectors.append(eigenvector)

    return eigenvectors


def solve_homogeneous(matrix):
    """
    Solve a homogeneous system of linear equations Ax = 0.

    Args:
        matrix (list): Coefficient matrix A.

    Returns:
        list: A non-trivial solution vector x.
    """
    # Use Gaussian elimination to find a solution
    matrix = [row[:] for row in matrix]  # Copy the matrix
    n = len(matrix)
    solution = [0] * n

    for i in range(n):
        # Pivot
        for j in range(i + 1, n):
            if matrix[j][i] != 0:
                factor = matrix[j][i] / matrix[i][i]
                for k in range(i, n):
                    matrix[j][k] -= factor * matrix[i][k]

    # Back-substitution
    for i in range(n - 1, -1, -1):
        solution[i] = 1  # Arbitrary non-trivial value
        for j in range(i + 1, n):
            solution[i] -= matrix[i][j] * solution[j]
        solution[i] /= matrix[i][i]

    return solution


# Example usage:
if __name__ == "__main__":
    matrix = [
        [4, -2, 1],
        [1, 1, -1],
        [3, -2, 0]
    ]

    eigenvalues = calculate_eigenvalues(matrix)
    eigenvectors = calculate_eigenvectors(matrix, eigenvalues)

    print("Eigenvalues:")
    print(eigenvalues)

    print("\nEigenvectors:")
    for vec in eigenvectors:
        print(vec)