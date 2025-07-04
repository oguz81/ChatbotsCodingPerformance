import numpy as np

def is_square_matrix(matrix):
    """
    Check if a matrix is square (number of rows equals number of columns).
    
    Args:
        matrix: Input matrix as numpy array or list of lists
        
    Returns:
        bool: True if the matrix is square, False otherwise
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Check if the matrix is square
    rows, cols = matrix.shape
    return rows == cols


def is_symmetric(matrix, tolerance=1e-10):
    """
    Check if a square matrix is symmetric: A = A^T
    A matrix is symmetric if it equals its transpose.
    
    Args:
        matrix: Square matrix as numpy array or list of lists
        tolerance: Numerical tolerance for floating-point comparisons
        
    Returns:
        bool: True if the matrix is symmetric, False otherwise
        
    Raises:
        ValueError: If the input is not a square matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Check if the matrix is square
    if not is_square_matrix(matrix):
        raise ValueError("Input must be a square matrix")
    
    n = matrix.shape[0]  # Size of the matrix
    
    # Check if the matrix equals its transpose
    for i in range(n):
        for j in range(i+1, n):  # Only check upper triangle vs lower triangle
            if abs(matrix[i, j] - matrix[j, i]) > tolerance:
                return False
    
    return True


def is_antisymmetric(matrix, tolerance=1e-10):
    """
    Check if a square matrix is antisymmetric (skew-symmetric): A = -A^T
    A matrix is antisymmetric if it equals the negative of its transpose.
    
    Args:
        matrix: Square matrix as numpy array or list of lists
        tolerance: Numerical tolerance for floating-point comparisons
        
    Returns:
        bool: True if the matrix is antisymmetric, False otherwise
        
    Raises:
        ValueError: If the input is not a square matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Check if the matrix is square
    if not is_square_matrix(matrix):
        raise ValueError("Input must be a square matrix")
    
    n = matrix.shape[0]  # Size of the matrix
    
    # Check diagonal elements (must be zero for antisymmetric matrices)
    for i in range(n):
        if abs(matrix[i, i]) > tolerance:
            return False
    
    # Check if A[i,j] = -A[j,i] for all i ≠ j
    for i in range(n):
        for j in range(i+1, n):  # Only check upper triangle vs lower triangle
            if abs(matrix[i, j] + matrix[j, i]) > tolerance:
                return False
    
    return True


# Example usage with detailed testing:
if __name__ == "__main__":
    # Example 1: A symmetric matrix
    symmetric_matrix = np.array([
        [1, 7, 3],
        [7, 4, -5],
        [3, -5, 6]
    ])
    
    # Example 2: An antisymmetric matrix
    antisymmetric_matrix = np.array([
        [0, 2, -1],
        [-2, 0, 4],
        [1, -4, 0]
    ])
    
    # Example 3: Neither symmetric nor antisymmetric
    neither_matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    # Example 4: Both symmetric and antisymmetric (only possible for zero matrix)
    zero_matrix = np.zeros((3, 3))
    
    # Example 5: Non-square matrix
    non_square_matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    # Example 6: Matrix with floating-point values
    float_symmetric_matrix = np.array([
        [1.0, 2.5, 3.7],
        [2.5, 4.0, 5.2],
        [3.7, 5.2, 6.0]
    ])
    
    # Example 7: Almost symmetric matrix (with small numerical errors)
    almost_symmetric = np.array([
        [1.0, 2.0, 3.0],
        [2.0 + 1e-11, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ])
    
    # Function to test both functions on a matrix and print results
    def test_matrix(matrix, name):
        print(f"\nTesting: {name}")
        print(matrix)
        
        try:
            # Test if the matrix is square
            square = is_square_matrix(matrix)
            print(f"Is square: {square}")
            
            if square:
                # Test if the matrix is symmetric
                symmetric = is_symmetric(matrix)
                print(f"Is symmetric: {symmetric}")
                
                # Test if the matrix is antisymmetric
                antisymmetric = is_antisymmetric(matrix)
                print(f"Is antisymmetric: {antisymmetric}")
                
                # For verification, calculate the transpose and check the conditions
                transposed = matrix.T
                print("\nVerification:")
                print(f"A = A^T: {np.allclose(matrix, transposed)}")
                print(f"A = -A^T: {np.allclose(matrix, -transposed)}")
        except ValueError as e:
            print(f"Error: {e}")
    
    # Run tests on all example matrices
    test_matrix(symmetric_matrix, "Symmetric Matrix")
    test_matrix(antisymmetric_matrix, "Antisymmetric Matrix")
    test_matrix(neither_matrix, "Neither Symmetric nor Antisymmetric")
    test_matrix(zero_matrix, "Zero Matrix (both Symmetric and Antisymmetric)")
    test_matrix(non_square_matrix, "Non-Square Matrix")
    test_matrix(float_symmetric_matrix, "Floating-Point Symmetric Matrix")
    test_matrix(almost_symmetric, "Almost Symmetric Matrix")