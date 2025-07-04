import numpy as np

def is_symmetric(A):
    """
    Checks if a square matrix A is symmetric (A == A.T).
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                return False
    return True

def is_antisymmetric(A):
    """
    Checks if a square matrix A is antisymmetric (A == -A.T).
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i][j] != -A[j][i]:
                return False
    return True

# Example usage:
if __name__ == "__main__":
    symmetric_matrix = np.array([[1, 2, 3],
                                 [2, 5, 6],
                                 [3, 6, 9]])

    antisymmetric_matrix = np.array([[0, 2, -3],
                                     [-2, 0, 4],
                                     [3, -4, 0]])

    print("Symmetric Matrix Test:", is_symmetric(symmetric_matrix))
    print("Antisymmetric Matrix Test:", is_antisymmetric(symmetric_matrix))

    print("\nSymmetric Matrix Test:", is_symmetric(antisymmetric_matrix))
    print("Antisymmetric Matrix Test:", is_antisymmetric(antisymmetric_matrix))
