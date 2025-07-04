def matrix_transpose(matrix):
    """
    Calculate the transpose of a matrix.
    
    Args:
        matrix: Input matrix (list of lists)
        
    Returns:
        Transposed matrix (list of lists)
    """
    # Verify the matrix is not empty
    if not matrix or not matrix[0]:
        return []
    
    # Using nested list comprehension to transpose
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matrix_complex_conjugate(matrix):
    """
    Calculate the complex conjugate of a matrix.
    
    Args:
        matrix: Input matrix with complex numbers (list of lists)
        
    Returns:
        Matrix with each element complex conjugated (list of lists)
    """
    # Verify the matrix is not empty
    if not matrix or not matrix[0]:
        return []
    
    # Function to get conjugate of a number (works for both real and complex)
    def conjugate(x):
        if isinstance(x, complex):
            return x.conjugate()
        return x  # Real numbers are their own conjugate
    
    return [[conjugate(element) for element in row] for row in matrix]

def matrix_hermitian_conjugate(matrix):
    """
    Calculate the Hermitian conjugate (conjugate transpose) of a matrix.
    
    Args:
        matrix: Input matrix with complex numbers (list of lists)
        
    Returns:
        Hermitian conjugate matrix (list of lists)
    """
    # First compute the transpose
    transposed = matrix_transpose(matrix)
    # Then compute the complex conjugate
    return matrix_complex_conjugate(transposed)

# Example usage:
if __name__ == "__main__":
    # Test matrix with real and complex numbers
    test_matrix = [
        [1, 2+3j, 4],
        [5j, 6, 7-2j],
        [8, 9+1j, 10j]
    ]
    
    print("Original Matrix:")
    for row in test_matrix:
        print(row)
    
    print("\nTranspose:")
    transposed = matrix_transpose(test_matrix)
    for row in transposed:
        print(row)
    
    print("\nComplex Conjugate:")
    conjugate = matrix_complex_conjugate(test_matrix)
    for row in conjugate:
        print(row)
    
    print("\nHermitian Conjugate:")
    hermitian = matrix_hermitian_conjugate(test_matrix)
    for row in hermitian:
        print(row)