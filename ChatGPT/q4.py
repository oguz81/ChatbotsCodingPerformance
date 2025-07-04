import numpy as np

def transpose_matrix(A):
    """
    Returns the transpose of matrix A.
    """
    rows, cols = A.shape
    result = np.zeros((cols, rows), dtype=A.dtype)
    
    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]
    return result

def complex_conjugate_matrix(A):
    """
    Returns the complex conjugate of matrix A.
    """
    rows, cols = A.shape
    result = np.zeros((rows, cols), dtype=A.dtype)
    
    for i in range(rows):
        for j in range(cols):
            result[i][j] = np.conj(A[i][j])
    return result

def hermitian_conjugate_matrix(A):
    """
    Returns the Hermitian conjugate (conjugate transpose) of matrix A.
    """
    rows, cols = A.shape
    result = np.zeros((cols, rows), dtype=A.dtype)
    
    for i in range(rows):
        for j in range(cols):
            result[j][i] = np.conj(A[i][j])
    return result

# Example usage:
if __name__ == "__main__":
    A = np.array([[1+2j, 3+4j], [5-6j, 7+8j]])
    print("Original Matrix A:\n", A)

    print("\nTranspose of A:\n", transpose_matrix(A))
    print("\nComplex Conjugate of A:\n", complex_conjugate_matrix(A))
    print("\nHermitian Conjugate of A:\n", hermitian_conjugate_matrix(A))
