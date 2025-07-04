import numpy as np
from sympy import symbols, eye, Matrix, solve

def calculate_eigenvalues_7x7(matrix):
    """
    Calculate the eigenvalues of a 7x7 matrix by solving the characteristic polynomial.

    Args:
        matrix (list or numpy.ndarray): A 7x7 matrix.

    Returns:
        list: Eigenvalues of the matrix (real and complex).
    """
    # Verify the matrix size
    if len(matrix) != 7 or any(len(row) != 7 for row in matrix):
        raise ValueError("Input matrix must be 7x7.")

    # Convert the matrix to a sympy Matrix
    matrix = Matrix(matrix)

    # Define the eigenvalue symbol (lambda)
    lam = symbols('lambda')

    # Compute the characteristic polynomial det(A - λI)
    char_poly = (matrix - lam * eye(7)).det()

    # Solve for eigenvalues (roots of the characteristic polynomial)
    eigenvalues = solve(char_poly, lam)
    return eigenvalues


def calculate_eigenvectors_7x7(matrix, eigenvalues):
    """
    Calculate the eigenvectors of a 7x7 matrix based on its eigenvalues.

    Args:
        matrix (list or numpy.ndarray): A 7x7 matrix.
        eigenvalues (list): Eigenvalues of the matrix.

    Returns:
        list: Eigenvectors of the matrix (one for each eigenvalue).
    """
    # Convert the matrix to a sympy Matrix
    matrix = Matrix(matrix)

    eigenvectors = []
    for eigenvalue in eigenvalues:
        # Form the matrix (A - λI)
        A_minus_lambda_I = matrix - eigenvalue * eye(7)

        # Solve the homogeneous equation (A - λI)x = 0 for eigenvector x
        null_space = A_minus_lambda_I.nullspace()

        # Append the eigenvector to the list (take the first basis vector)
        eigenvectors.append(null_space[0] if null_space else None)

    return eigenvectors


# Example usage
if __name__ == "__main__":
    # Example 7x7 matrix
    matrix = [
        [4, -2, 3, 1, 0, 2, 1],
        [1, 5, -1, 2, 3, 0, 4],
        [3, -1, 6, 0, -2, 1, 5],
        [2, 0, 1, 7, 3, -1, -2],
        [1, 3, -2, 4, 8, 0, 2],
        [-1, 2, 0, -3, 2, 9, 1],
        [0, 1, 3, -1, 2, 1, 10]
    ]

    try:
        # Calculate eigenvalues
        eigenvalues = calculate_eigenvalues_7x7(matrix)
        print("Eigenvalues:")
        print(eigenvalues)

        # Calculate eigenvectors
        eigenvectors = calculate_eigenvectors_7x7(matrix, eigenvalues)
        print("\nEigenvectors:")
        for vec in eigenvectors:
            print(vec)
    except Exception as e:
        print(f"Error: {e}")