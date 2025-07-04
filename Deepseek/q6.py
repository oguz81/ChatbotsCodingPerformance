def is_symmetric(matrix, tolerance=1e-10):
    """
    Check if a square matrix is symmetric (A = A^T).
    
    Args:
        matrix: Input square matrix (list of lists)
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        bool: True if matrix is symmetric, False otherwise
        
    Raises:
        ValueError: If matrix is not square
    """
    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square to check symmetry")
    
    # Check symmetry by comparing elements
    for i in range(n):
        for j in range(i+1, n):  # Only need to check upper triangular part
            if abs(matrix[i][j] - matrix[j][i]) > tolerance:
                return False
    return True

def is_antisymmetric(matrix, tolerance=1e-10):
    """
    Check if a square matrix is antisymmetric (A = -A^T).
    
    Args:
        matrix: Input square matrix (list of lists)
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        bool: True if matrix is antisymmetric, False otherwise
        
    Raises:
        ValueError: If matrix is not square
    """
    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square to check antisymmetry")
    
    # Check antisymmetry by comparing elements
    for i in range(n):
        # Diagonal elements must be zero (within tolerance)
        if abs(matrix[i][i]) > tolerance:
            return False
        
        for j in range(i+1, n):  # Only need to check upper triangular part
            if abs(matrix[i][j] + matrix[j][i]) > tolerance:
                return False
    return True

# Example usage:
if __name__ == "__main__":
    # Test matrices
    symmetric_matrix = [
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6]
    ]
    
    antisymmetric_matrix = [
        [0, 2, -3],
        [-2, 0, 4],
        [3, -4, 0]
    ]
    
    regular_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    print("Symmetric matrix check:", is_symmetric(symmetric_matrix))  # True
    print("Antisymmetric matrix check:", is_antisymmetric(symmetric_matrix))  # False
    
    print("\nAntisymmetric matrix check:", is_antisymmetric(antisymmetric_matrix))  # True
    print("Symmetric matrix check:", is_symmetric(antisymmetric_matrix))  # False
    
    print("\nRegular matrix checks:")
    print("Symmetric:", is_symmetric(regular_matrix))  # False
    print("Antisymmetric:", is_antisymmetric(regular_matrix))  # False
    
    # Test with floating point numbers
    float_symmetric = [
        [1.0, 2.0000001, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ]
    print("\nFloating point symmetric (with tolerance):", 
          is_symmetric(float_symmetric, tolerance=1e-6))  # True