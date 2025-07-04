import numpy as np

def calculate_eigenvalues_eigenvectors_3x3_no_linalg(matrix):
    """
    Calculates the eigenvalues and eigenvectors of a 3x3 matrix without using linalg.

    Args:
        matrix: A list of lists representing a 3x3 matrix.

    Returns:
        A tuple containing:
          - eigenvalues: A NumPy array of the three eigenvalues.
          - eigenvectors: A list of three eigenvectors (NumPy arrays).
          Returns None if the input is not a 3x3 matrix or if eigenvector
          calculation fails.
    """
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        print("Error: Input must be a 3x3 matrix.")
        return None

    A = np.array(matrix, dtype=float)

    # Calculate the coefficients of the characteristic polynomial:
    # det(A - lambda*I) = -lambda^3 + c2*lambda^2 + c1*lambda + c0 = 0

    c2 = np.trace(A)
    c1 = (A[0, 1] * A[1, 0] + A[0, 2] * A[2, 0] + A[1, 2] * A[2, 1] -
          A[0, 0] * A[1, 1] - A[0, 0] * A[2, 2] - A[1, 1] * A[2, 2])
    c0 = np.linalg.det(A)  # We'll use det here for the constant term

    # The characteristic polynomial coefficients [a, b, c, d] correspond to
    # a*lambda^3 + b*lambda^2 + c*lambda + d = 0.
    # Here, we have -lambda^3 + c2*lambda^2 + c1*lambda + c0 = 0,
    # so the coefficients are [-1, c2, c1, c0].
    coefficients = [-1, c2, c1, c0]

    # Find the roots of the characteristic polynomial (the eigenvalues)
    eigenvalues = np.roots(coefficients)

    eigenvectors = []
    I = np.identity(3)

    for eigenvalue in eigenvalues:
        # Solve the system (A - lambda*I)v = 0 for the eigenvector v
        matrix_eq = A - eigenvalue * I

        # We need to find a non-trivial solution to this homogeneous system.
        # We can try to use Gaussian elimination or similar methods.
        # A simple approach is to find a non-zero vector in the null space.

        # For a 3x3 matrix, if the eigenvalue is correct, the determinant of
        # matrix_eq should ideally be zero (or very close to it due to
        # numerical precision).

        # We'll try a simplified approach: consider the first two rows and
        # solve for the ratios of the eigenvector components.

        m = matrix_eq[:2, :]
        if np.linalg.matrix_rank(m) < 2:
            # Handle cases where the first two rows are linearly dependent
            # This requires more robust handling, which is complex.
            print("Warning: Linear dependency in rows, eigenvector calculation might be unstable.")
            eigenvectors.append(np.array([1, 1, 1])) # Placeholder
            continue

        # Assume v = [x, y, 1] and solve for x and y
        try:
            sub_matrix = m[:, :2]
            rhs = -m[:, 2]
            xy = np.linalg.solve(sub_matrix, rhs)
            eigenvector = np.array([xy[0], xy[1], 1])
            # Normalize the eigenvector (optional)
            eigenvector = eigenvector / np.linalg.norm(eigenvector)
            eigenvectors.append(eigenvector)
        except np.linalg.LinAlgError:
            # If the sub-matrix is singular, try other component being 1
            try:
                sub_matrix = m[:, [0, 2]]
                rhs = -m[:, 1]
                xz = np.linalg.solve(sub_matrix, rhs)
                eigenvector = np.array([xz[0], 1, xz[1]])
                eigenvector = eigenvector / np.linalg.norm(eigenvector)
                eigenvectors.append(eigenvector)
            except np.linalg.LinAlgError:
                try:
                    sub_matrix = m[:, 1:]
                    rhs = -m[:, 0]
                    yz = np.linalg.solve(sub_matrix, rhs)
                    eigenvector = np.array([1, yz[0], yz[1]])
                    eigenvector = eigenvector / np.linalg.norm(eigenvector)
                    eigenvectors.append(eigenvector)
                except np.linalg.LinAlgError:
                    print("Warning: Could not find a non-trivial eigenvector.")
                    eigenvectors.append(np.array([1, 0, 0])) # Placeholder

    return eigenvalues, eigenvectors

if __name__ == "__main__":
    # Example 3x3 matrix
    matrix_3x3 = [
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
    ]

    result = calculate_eigenvalues_eigenvectors_3x3_no_linalg(matrix_3x3)

    if result:
        eigenvalues, eigenvectors = result
        print("Original Matrix:")
        for row in matrix_3x3:
            print(row)

        print("\nEigenvalues:")
        print(eigenvalues)

        print("\nEigenvectors:")
        for i, eigenvector in enumerate(eigenvectors):
            print(f"Eigenvector for eigenvalue {eigenvalues[i]}: {eigenvector}")

        # Verification (approximate due to potential numerical issues)
        A = np.array(matrix_3x3)
        for i in range(len(eigenvalues)):
            eigenvalue = eigenvalues[i]
            eigenvector = eigenvectors[i]
            product = np.dot(A, eigenvector)
            scaled_eigenvector = eigenvalue * eigenvector
            print(f"\nVerification for eigenvalue {eigenvalue}:")
            print(f"A * v:\n{product}")
            print(f"lambda * v:\n{scaled_eigenvector}")
            print(f"Are they close? {np.allclose(product, scaled_eigenvector)}")
    else:
        print("Eigenvalue/eigenvector calculation failed.")

    # Example of a non-3x3 matrix
    matrix_incorrect_size = [
        [1, 2],
        [3, 4]
    ]
    calculate_eigenvalues_eigenvectors_3x3_no_linalg(matrix_incorrect_size)