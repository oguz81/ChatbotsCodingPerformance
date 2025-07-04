import numpy as np

def inverse_matrix(A):
    """
    Calculates the inverse of a square matrix A using Gaussian elimination.
    Raises an error if the matrix is singular or not square.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    
    # Create augmented matrix [A | I]
    I = np.identity(n)
    aug = np.hstack((A.astype(float), I))
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        pivot = aug[i, i]
        if pivot == 0:
            # Search for non-zero pivot below
            for j in range(i+1, n):
                if aug[j, i] != 0:
                    aug[[i, j]] = aug[[j, i]]
                    pivot = aug[i, i]
                    break
            else:
                raise ValueError("Matrix is singular and not invertible.")

        # Normalize pivot row
        aug[i] = aug[i] / pivot

        # Eliminate other rows
        for j in range(n):
            if j != i:
                factor = aug[j, i]
                aug[j] = aug[j] - factor * aug[i]

    # Extract inverse from augmented matrix
    inverse = aug[:, n:]
    return inverse

# Example usage:
if __name__ == "__main__":
    A = np.array([[2, 1], [5, 3]])
    print("Original Matrix A:\n", A)
    invA = inverse_matrix(A)
    print("\nInverse of A:\n", invA)
