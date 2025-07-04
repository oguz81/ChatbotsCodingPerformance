import numpy as np

def matrix_inverse(matrix):
    """
    Calculate the inverse of a square matrix using Gauss-Jordan elimination.
    
    Args:
        matrix: Square input matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The inverse of the input matrix
        
    Raises:
        ValueError: If the matrix is not square or is singular (not invertible)
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=float)
    
    # Check if the matrix is square
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Matrix must be square to have an inverse")
    
    # Create an augmented matrix [A|I]
    n = rows
    augmented = np.zeros((n, 2*n), dtype=float)
    augmented[:, :n] = matrix
    augmented[:, n:] = np.eye(n)  # Identity matrix
    
    # Perform Gauss-Jordan elimination
    for i in range(n):
        # Find the pivot (maximum value in the current column)
        pivot_row = i
        pivot_value = abs(augmented[i, i])
        
        for j in range(i+1, n):
            if abs(augmented[j, i]) > pivot_value:
                pivot_row = j
                pivot_value = abs(augmented[j, i])
        
        # Check if the matrix is singular
        if pivot_value < 1e-10:
            raise ValueError("Matrix is singular and cannot be inverted")
        
        # Swap rows if necessary
        if pivot_row != i:
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
        
        # Scale the pivot row
        pivot = augmented[i, i]
        augmented[i] = augmented[i] / pivot
        
        # Eliminate other rows
        for j in range(n):
            if j != i:
                factor = augmented[j, i]
                augmented[j] = augmented[j] - factor * augmented[i]
    
    # Extract the inverse from the right side of the augmented matrix
    inverse = augmented[:, n:]
    
    return inverse


# Example usage:
if __name__ == "__main__":
    # Example 1: Simple 2x2 matrix
    A1 = np.array([
        [4, 7],
        [2, 6]
    ])
    
    print("Matrix A1:")
    print(A1)
    
    try:
        A1_inv = matrix_inverse(A1)
        print("\nInverse of A1:")
        print(A1_inv)
        
        # Verify: A * A^-1 should be approximately the identity matrix
        print("\nA1 * A1_inv (should be approximately identity):")
        print(np.matmul(A1, A1_inv))
        
        # Compare with NumPy's built-in inverse function
        print("\nVerification with NumPy:")
        print("Match:", np.allclose(A1_inv, np.linalg.inv(A1)))
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 2: A larger matrix
    A2 = np.array([
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0]
    ])
    
    print("\nMatrix A2:")
    print(A2)
    
    try:
        A2_inv = matrix_inverse(A2)
        print("\nInverse of A2:")
        print(A2_inv)
        
        # Verify: A * A^-1 should be approximately the identity matrix
        print("\nA2 * A2_inv (should be approximately identity):")
        print(np.matmul(A2, A2_inv))
        
        # Compare with NumPy's built-in inverse function
        print("\nVerification with NumPy:")
        print("Match:", np.allclose(A2_inv, np.linalg.inv(A2)))
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 3: A singular matrix (not invertible)
    A3 = np.array([
        [1, 2, 3],
        [2, 4, 6],  # This row is 2 times the first row
        [0, 1, 7]
    ])
    
    print("\nMatrix A3 (singular):")
    print(A3)
    
    try:
        A3_inv = matrix_inverse(A3)
        print("\nInverse of A3:")
        print(A3_inv)
    except ValueError as e:
        print(f"\nError: {e}")