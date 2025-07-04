import numpy as np

def is_orthogonal(A, tol=1e-10):
    """
    Checks if a square matrix A is orthogonal (A^T * A = I).
    Only for real-valued matrices.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    n = A.shape[0]
    # Compute A^T * A
    product = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sum_val = 0
            for k in range(n):
                sum_val += A[k][i] * A[k][j]
            product[i][j] = sum_val

    # Check if product is close to identity matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                if abs(product[i][j] - 1) > tol:
                    return False
            else:
                if abs(product[i][j]) > tol:
                    return False
    return True

def is_unitary(A, tol=1e-10):
    """
    Checks if a square matrix A is unitary (A^H * A = I).
    Works with complex-valued matrices.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    n = A.shape[0]
    # Compute A^H * A
    product = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            sum_val = 0
            for k in range(n):
                sum_val += np.conj(A[k][i]) * A[k][j]
            product[i][j] = sum_val

    # Check if product is close to identity matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                if abs(product[i][j] - 1) > tol:
                    return False
            else:
                if abs(product[i][j]) > tol:
                    return False
    return True

# Example usage:
if __name__ == "__main__":
    # Orthogonal matrix example (rotation matrix)
    orthogonal_matrix = np.array([[0, 1],
                                  [-1, 0]])

    # Unitary matrix example (complex rotation)
    unitary_matrix = np.array([[1/np.sqrt(2), 1j/np.sqrt(2)],
                               [-1j/np.sqrt(2), 1/np.sqrt(2)]])

    print("Orthogonal Matrix Test:", is_orthogonal(orthogonal_matrix))
    print("Unitary Matrix Test:", is_unitary(orthogonal_matrix))

    print("\nOrthogonal Matrix Test:", is_orthogonal(unitary_matrix.real))
    print("Unitary Matrix Test:", is_unitary(unitary_matrix))
