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


def is_hermitian(matrix, tolerance=1e-10):
    """
    Check if a square matrix is Hermitian: A = A^H
    A matrix is Hermitian if it equals its conjugate transpose.
    
    Args:
        matrix: Square matrix as numpy array or list of lists
        tolerance: Numerical tolerance for floating-point comparisons
        
    Returns:
        bool: True if the matrix is Hermitian, False otherwise
        
    Raises:
        ValueError: If the input is not a square matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=complex)
    
    # Check if the matrix is square
    if not is_square_matrix(matrix):
        raise ValueError("Input must be a square matrix")
    
    n = matrix.shape[0]  # Size of the matrix
    
    # Check if the matrix equals its conjugate transpose
    for i in range(n):
        # Diagonal elements must be real for Hermitian matrices
        if abs(matrix[i, i].imag) > tolerance:
            return False
        
        # Check off-diagonal elements
        for j in range(i+1, n):
            if abs(matrix[i, j] - np.conj(matrix[j, i])) > tolerance:
                return False
    
    return True


def is_anti_hermitian(matrix, tolerance=1e-10):
    """
    Check if a square matrix is anti-Hermitian (skew-Hermitian): A = -A^H
    A matrix is anti-Hermitian if it equals the negative of its conjugate transpose.
    
    Args:
        matrix: Square matrix as numpy array or list of lists
        tolerance: Numerical tolerance for floating-point comparisons
        
    Returns:
        bool: True if the matrix is anti-Hermitian, False otherwise
        
    Raises:
        ValueError: If the input is not a square matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=complex)
    
    # Check if the matrix is square
    if not is_square_matrix(matrix):
        raise ValueError("Input must be a square matrix")
    
    n = matrix.shape[0]  # Size of the matrix
    
    # Diagonal elements must be imaginary (or zero) for anti-Hermitian matrices
    for i in range(n):
        if abs(matrix[i, i].real) > tolerance:
            return False
    
    # Check if A[i,j] = -conj(A[j,i]) for all i ≠ j
    for i in range(n):
        for j in range(i+1, n):
            if abs(matrix[i, j] + np.conj(matrix[j, i])) > tolerance:
                return False
    
    return True


def compute_conjugate_transpose(matrix):
    """
    Compute the conjugate transpose (Hermitian conjugate) of a matrix.
    
    Args:
        matrix: Input matrix as numpy array
        
    Returns:
        numpy.ndarray: The conjugate transpose of the input matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=complex)
    
    rows, cols = matrix.shape
    result = np.zeros((cols, rows), dtype=complex)
    
    for i in range(rows):
        for j in range(cols):
            result[j, i] = np.conj(matrix[i, j])
    
    return result


# Example usage with detailed testing:
if __name__ == "__main__":
    # Example 1: A Hermitian matrix
    hermitian_matrix = np.array([
        [2, 2+1j, 4-2j],
        [2-1j, 3, 0+1j],
        [4+2j, 0-1j, 1]
    ], dtype=complex)
    
    # Example 2: An anti-Hermitian matrix
    anti_hermitian_matrix = np.array([
        [0j, 2+1j, 4-2j],
        [-2+1j, 0j, 5+3j],
        [-4-2j, -5+3j, 0j]
    ], dtype=complex)
    
    # Example 3: Neither Hermitian nor anti-Hermitian
    neither_matrix = np.array([
        [1+1j, 2+2j, 3+3j],
        [4+4j, 5+5j, 6+6j],
        [7+7j, 8+8j, 9+9j]
    ], dtype=complex)
    
    # Example 4: Real symmetric matrix (also Hermitian)
    real_symmetric = np.array([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6]
    ], dtype=complex)
    
    # Example 5: Real antisymmetric matrix (also anti-Hermitian)
    real_antisymmetric = np.array([
        [0, 2, -3],
        [-2, 0, 1],
        [3, -1, 0]
    ], dtype=complex)
    
    # Example 6: Zero matrix (both Hermitian and anti-Hermitian)
    zero_matrix = np.zeros((3, 3), dtype=complex)
    
    # Example 7: Non-square matrix
    non_square_matrix = np.array([
        [1+1j, 2+2j, 3+3j],
        [4+4j, 5+5j, 6+6j]
    ], dtype=complex)
    
    # Example 8: Matrix with small numerical errors
    almost_hermitian = np.array([
        [2, 2+1j, 4-2j],
        [2-1j, 3, 0+1j],
        [4+2j, 0-1j, 1+1e-11j]  # Small imaginary part on diagonal
    ], dtype=complex)
    
    # Function to test both functions on a matrix and print results
    def test_matrix(matrix, name):
        print(f"\nTesting: {name}")
        print(matrix)
        
        try:
            # Test if the matrix is square
            square = is_square_matrix(matrix)
            print(f"Is square: {square}")
            
            if square:
                # Test if the matrix is Hermitian
                hermitian = is_hermitian(matrix)
                print(f"Is Hermitian: {hermitian}")
                
                # Test if the matrix is anti-Hermitian
                anti_herm = is_anti_hermitian(matrix)
                print(f"Is anti-Hermitian: {anti_herm}")
                
                # For verification, calculate the conjugate transpose and check the conditions
                conj_transpose = compute_conjugate_transpose(matrix)
                print("\nVerification:")
                print(f"A = A^H: {np.allclose(matrix, conj_transpose)}")
                print(f"A = -A^H: {np.allclose(matrix, -conj_transpose)}")
                
                # Additional check for Hermitian: diagonal elements must be real
                if hermitian:
                    diag_real = all(abs(matrix[i, i].imag) < 1e-10 for i in range(matrix.shape[0]))
                    print(f"All diagonal elements are real: {diag_real}")
                
                # Additional check for anti-Hermitian: diagonal elements must be imaginary or zero
                if anti_herm:
                    diag_imag = all(abs(matrix[i, i].real) < 1e-10 for i in range(matrix.shape[0]))
                    print(f"All diagonal elements are imaginary or zero: {diag_imag}")
        except ValueError as e:
            print(f"Error: {e}")
    
    # Run tests on all example matrices
    test_matrix(hermitian_matrix, "Hermitian Matrix")
    test_matrix(anti_hermitian_matrix, "Anti-Hermitian Matrix")
    test_matrix(neither_matrix, "Neither Hermitian nor Anti-Hermitian")
    test_matrix(real_symmetric, "Real Symmetric Matrix (also Hermitian)")
    test_matrix(real_antisymmetric, "Real Antisymmetric Matrix (also Anti-Hermitian)")
    test_matrix(zero_matrix, "Zero Matrix (both Hermitian and Anti-Hermitian)")
    test_matrix(non_square_matrix, "Non-Square Matrix")
    test_matrix(almost_hermitian, "Almost Hermitian Matrix")