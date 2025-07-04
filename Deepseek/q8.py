def matrix_transpose(matrix):
    """Compute the transpose of a matrix"""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def complex_conjugate(matrix):
    """Compute the element-wise complex conjugate of a matrix"""
    return [[element.conjugate() if isinstance(element, complex) else element 
             for element in row] for row in matrix]

def matrix_multiply(a, b):
    """Multiply two square matrices"""
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def is_identity(matrix, tolerance=1e-10):
    """Check if a matrix is approximately the identity matrix"""
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if i == j:
                if abs(matrix[i][j] - 1) > tolerance:
                    return False
            else:
                if abs(matrix[i][j]) > tolerance:
                    return False
    return True

def is_orthogonal(matrix, tolerance=1e-10):
    """
    Check if a real square matrix is orthogonal (A^T A = I).
    
    Args:
        matrix: Input square matrix (list of lists)
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        bool: True if matrix is orthogonal, False otherwise
        
    Raises:
        ValueError: If matrix is not square
    """
    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square to check orthogonality")
    
    # Check all elements are real
    for row in matrix:
        for element in row:
            if isinstance(element, complex):
                raise ValueError("Orthogonal matrices must be real-valued")
    
    # Compute A^T A and check if it's identity
    a_transpose = matrix_transpose(matrix)
    product = matrix_multiply(a_transpose, matrix)
    return is_identity(product, tolerance)

def is_unitary(matrix, tolerance=1e-10):
    """
    Check if a square matrix is unitary (Aᴴ A = I).
    
    Args:
        matrix: Input square matrix (list of lists)
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        bool: True if matrix is unitary, False otherwise
        
    Raises:
        ValueError: If matrix is not square
    """
    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square to check unitarity")
    
    # Compute Aᴴ A and check if it's identity
    a_conj_transpose = matrix_transpose(complex_conjugate(matrix))
    product = matrix_multiply(a_conj_transpose, matrix)
    return is_identity(product, tolerance)

# Example usage:
if __name__ == "__main__":
    # Test matrices
    orthogonal_matrix = [
        [0.70710678, -0.70710678],
        [0.70710678, 0.70710678]
    ]
    
    unitary_matrix = [
        [0.70710678, -0.70710678j],
        [0.70710678j, 0.70710678]
    ]
    
    regular_matrix = [
        [1, 2],
        [3, 4]
    ]
    
    print("Orthogonal matrix check:", is_orthogonal(orthogonal_matrix))  # True
    print("Unitary matrix check:", is_unitary(orthogonal_matrix))  # Also True (orthogonal implies unitary)
    
    print("\nUnitary matrix check:", is_unitary(unitary_matrix))  # True
    print("Orthogonal matrix check:", is_orthogonal(unitary_matrix))  # False (contains complex numbers)
    
    print("\nRegular matrix checks:")
    print("Orthogonal:", is_orthogonal(regular_matrix))  # False
    print("Unitary:", is_unitary(regular_matrix))  # False
    
    # Test with floating point precision
    nearly_orthogonal = [
        [0.70710678118, -0.70710678118],
        [0.70710678118, 0.70710678118]
    ]
    print("\nNearly orthogonal matrix (with tolerance):", 
          is_orthogonal(nearly_orthogonal, tolerance=1e-8))  # True