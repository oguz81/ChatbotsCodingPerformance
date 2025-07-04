import numpy as np

def determinant_recursive(matrix):
    """
    Calculates the determinant of a matrix using Laplace expansion (recursive).
    Works for any square matrix but is inefficient for large matrices.
    """
    n = matrix.shape[0]
    if n == 1:
        return matrix[0, 0]
    if n == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    det = 0
    for col in range(n):
        minor = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)
        cofactor = ((-1) ** col) * matrix[0, col] * determinant_recursive(minor)
        det += cofactor
    return det

def determinant_7x7(matrix):
    if matrix.shape != (7, 7):
        raise ValueError("Matrix must be 7x7.")
    return determinant_recursive(matrix)

def determinant_11x11(matrix):
    if matrix.shape != (11, 11):
        raise ValueError("Matrix must be 11x11.")
    return determinant_recursive(matrix)

# Example usage:
if __name__ == "__main__":
    # Random 7x7 matrix
    matrix7x7 = np.random.randint(1, 10, size=(7, 7))
    print("7x7 Matrix:\n", matrix7x7)
    det7x7 = determinant_7x7(matrix7x7)
    print("Determinant of 7x7 matrix:", det7x7)

    # Random 11x11 matrix
    matrix11x11 = np.random.randint(1, 10, size=(11, 11))
    print("\n11x11 Matrix:\n", matrix11x11)
    det11x11 = determinant_11x11(matrix11x11)
    print("Determinant of 11x11 matrix:", det11x11)
