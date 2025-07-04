import numpy as np

def is_hermitian(A):
    """
    Checks if a square matrix A is Hermitian (A == A^H).
    A^H is the conjugate transpose of A.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i][j] != np.conj(A[j][i]):
                return False
    return True

def is_antihermitian(A):
    """
    Checks if a square matrix A is Anti-Hermitian (A == -A^H).
    A^H is the conjugate transpose of A.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i][j] != -np.conj(A[j][i]):
                return False
    return True

# Example usage:
if __name__ == "__main__":
    hermitian_matrix = np.array([[2+0j, 1-1j], [1+1j, 3+0j]])
    antihermitian_matrix = np.array([[0+0j, 2-1j], [-2-1j, 0+0j]])

    print("Hermitian Matrix Test:", is_hermitian(hermitian_matrix))
    print("Anti-Hermitian Matrix Test:", is_antihermitian(hermitian_matrix))

    print("\nHermitian Matrix Test:", is_hermitian(antihermitian_matrix))
    print("Anti-Hermitian Matrix Test:", is_antihermitian(antihermitian_matrix))
