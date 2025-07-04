import numpy as np

def matrix_transpose(matrix):
    """
    Calculate the transpose of a matrix.
    
    Args:
        matrix: Input matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The transpose of the input matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Get dimensions
    rows, cols = matrix.shape
    
    # Initialize result matrix with transposed dimensions
    result = np.zeros((cols, rows), dtype=matrix.dtype)
    
    # Compute transpose
    for i in range(rows):
        for j in range(cols):
            result[j, i] = matrix[i, j]
            
    return result


def matrix_complex_conjugate(matrix):
    """
    Calculate the complex conjugate of a matrix.
    
    Args:
        matrix: Input matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The complex conjugate of the input matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=complex)
    
    # Get dimensions
    rows, cols = matrix.shape
    
    # Initialize result matrix
    result = np.zeros((rows, cols), dtype=complex)
    
    # Compute complex conjugate
    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.conj(matrix[i, j])
            
    return result


def matrix_hermitian_conjugate(matrix):
    """
    Calculate the Hermitian conjugate (conjugate transpose) of a matrix.
    
    Args:
        matrix: Input matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The Hermitian conjugate of the input matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=complex)
    
    # Get dimensions
    rows, cols = matrix.shape
    
    # Initialize result matrix with transposed dimensions
    result = np.zeros((cols, rows), dtype=complex)
    
    # Compute Hermitian conjugate (conjugate transpose)
    for i in range(rows):
        for j in range(cols):
            result[j, i] = np.conj(matrix[i, j])
            
    return result


# Example usage:
if __name__ == "__main__":
    # Create a complex matrix example
    matrix = np.array([
        [1 + 2j, 3 - 1j, 2 + 0j],
        [4 + 5j, 6 - 3j, 7 + 2j]
    ])
    
    print("Original Matrix:")
    print(matrix)
    
    # Calculate and display the transpose
    transposed = matrix_transpose(matrix)
    print("\nTranspose:")
    print(transposed)
    print("Verification with NumPy:", np.allclose(transposed, np.transpose(matrix)))
    
    # Calculate and display the complex conjugate
    conjugated = matrix_complex_conjugate(matrix)
    print("\nComplex Conjugate:")
    print(conjugated)
    print("Verification with NumPy:", np.allclose(conjugated, np.conj(matrix)))
    
    # Calculate and display the Hermitian conjugate
    hermitian = matrix_hermitian_conjugate(matrix)
    print("\nHermitian Conjugate (Conjugate Transpose):")
    print(hermitian)
    print("Verification with NumPy:", np.allclose(hermitian, np.conj(matrix).T))
    
    # Demonstrate that the Hermitian conjugate is equivalent to the transpose of the complex conjugate
    print("\nVerifying Hermitian = Transpose(Conjugate):")
    hermitian_alternative = matrix_transpose(matrix_complex_conjugate(matrix))
    print(np.allclose(hermitian, hermitian_alternative))