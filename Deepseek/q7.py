def is_hermitian(matrix, tolerance=1e-10):
    """
    Check if a square matrix is Hermitian (A = Aᴴ).
    
    Args:
        matrix: Input square matrix (list of lists)
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        bool: True if matrix is Hermitian, False otherwise
        
    Raises:
        ValueError: If matrix is not square
    """
    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square to check Hermitian property")
    
    # Check Hermitian property by comparing elements with their conjugates
    for i in range(n):
        # Diagonal elements must be real (within tolerance)
        if abs(matrix[i][i].imag) > tolerance if isinstance(matrix[i][i], complex) else False:
            return False
        
        for j in range(i+1, n):  # Only need to check upper triangular part
            # A[i][j] should equal conjugate of A[j][i]
            if isinstance(matrix[i][j], complex) or isinstance(matrix[j][i], complex):
                if abs(matrix[i][j] - matrix[j][i].conjugate()) > tolerance:
                    return False
            else:
                if abs(matrix[i][j] - matrix[j][i]) > tolerance:
                    return False
    return True

def is_anti_hermitian(matrix, tolerance=1e-10):
    """
    Check if a square matrix is anti-Hermitian (A = -Aᴴ).
    
    Args:
        matrix: Input square matrix (list of lists)
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        bool: True if matrix is anti-Hermitian, False otherwise
        
    Raises:
        ValueError: If matrix is not square
    """
    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square to check anti-Hermitian property")
    
    # Check anti-Hermitian property
    for i in range(n):
        # Diagonal elements must be imaginary (real part zero within tolerance)
        if isinstance(matrix[i][i], complex):
            if abs(matrix[i][i].real) > tolerance:
                return False
        else:
            if abs(matrix[i][i]) > tolerance:
                return False
        
        for j in range(i+1, n):  # Only need to check upper triangular part
            # A[i][j] should equal negative conjugate of A[j][i]
            if isinstance(matrix[i][j], complex) or isinstance(matrix[j][i], complex):
                if abs(matrix[i][j] + matrix[j][i].conjugate()) > tolerance:
                    return False
            else:
                if abs(matrix[i][j] + matrix[j][i]) > tolerance:
                    return False
    return True

# Example usage:
if __name__ == "__main__":
    # Test matrices
    hermitian_matrix = [
        [3, 2+1j, 4-1j],
        [2-1j, 5, 0+2j],
        [4+1j, 0-2j, 7]
    ]
    
    anti_hermitian_matrix = [
        [0j, 2+1j, -3j],
        [-2+1j, 0j, 4-1j],
        [3j, -4-1j, 0j]
    ]
    
    regular_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    print("Hermitian matrix check:", is_hermitian(hermitian_matrix))  # True
    print("Anti-Hermitian matrix check:", is_anti_hermitian(hermitian_matrix))  # False
    
    print("\nAnti-Hermitian matrix check:", is_anti_hermitian(anti_hermitian_matrix))  # True
    print("Hermitian matrix check:", is_hermitian(anti_hermitian_matrix))  # False
    
    print("\nRegular matrix checks:")
    print("Hermitian:", is_hermitian(regular_matrix))  # False
    print("Anti-Hermitian:", is_anti_hermitian(regular_matrix))  # False
    
    # Test with floating point precision
    nearly_hermitian = [
        [3.0, 2.0+1.0000001j],
        [2.0-1.0j, 5.0]
    ]
    print("\nNearly Hermitian matrix (with tolerance):", 
          is_hermitian(nearly_hermitian, tolerance=1e-6))  # True