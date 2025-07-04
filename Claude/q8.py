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


def matrix_multiply(A, B):
    """
    Multiply two matrices without using np.dot or @.
    
    Args:
        A: First matrix
        B: Second matrix
        
    Returns:
        numpy.ndarray: Result of matrix multiplication A * B
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.result_type(A.dtype, B.dtype))
    
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]
                
    return result


def matrix_transpose(matrix):
    """
    Compute the transpose of a matrix without using .T.
    
    Args:
        matrix: Input matrix
        
    Returns:
        numpy.ndarray: Transpose of the input matrix
    """
    rows, cols = matrix.shape
    result = np.zeros((cols, rows), dtype=matrix.dtype)
    
    for i in range(rows):
        for j in range(cols):
            result[j, i] = matrix[i, j]
            
    return result


def conjugate_transpose(matrix):
    """
    Compute the conjugate transpose (Hermitian conjugate) of a matrix.
    
    Args:
        matrix: Input matrix
        
    Returns:
        numpy.ndarray: Conjugate transpose of the input matrix
    """
    rows, cols = matrix.shape
    result = np.zeros((cols, rows), dtype=complex if np.iscomplexobj(matrix) else matrix.dtype)
    
    for i in range(rows):
        for j in range(cols):
            result[j, i] = np.conj(matrix[i, j])
            
    return result


def identity_matrix(n, dtype=float):
    """
    Create an identity matrix of size n×n.
    
    Args:
        n: Size of the identity matrix
        dtype: Data type of the matrix
        
    Returns:
        numpy.ndarray: n×n identity matrix
    """
    result = np.zeros((n, n), dtype=dtype)
    
    for i in range(n):
        result[i, i] = 1
        
    return result


def is_orthogonal(matrix, tolerance=1e-10):
    """
    Check if a square matrix is orthogonal: A^T * A = A * A^T = I
    A matrix is orthogonal if its transpose equals its inverse.
    
    Args:
        matrix: Square matrix as numpy array or list of lists
        tolerance: Numerical tolerance for floating-point comparisons
        
    Returns:
        bool: True if the matrix is orthogonal, False otherwise
        
    Raises:
        ValueError: If the input is not a square matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=float)
    
    # Check if the matrix is square
    if not is_square_matrix(matrix):
        raise ValueError("Input must be a square matrix")
    
    # Check if the matrix is real
    if np.iscomplexobj(matrix):
        return False  # Orthogonal matrices must be real
    
    n = matrix.shape[0]  # Size of the matrix
    
    # Compute A^T * A
    matrix_t = matrix_transpose(matrix)
    product = matrix_multiply(matrix_t, matrix)
    
    # Check if A^T * A = I
    identity = identity_matrix(n, matrix.dtype)
    
    for i in range(n):
        for j in range(n):
            if abs(product[i, j] - identity[i, j]) > tolerance:
                return False
    
    # Optionally check A * A^T = I as well for numerical stability
    product2 = matrix_multiply(matrix, matrix_t)
    
    for i in range(n):
        for j in range(n):
            if abs(product2[i, j] - identity[i, j]) > tolerance:
                return False
    
    return True


def is_unitary(matrix, tolerance=1e-10):
    """
    Check if a square matrix is unitary: A^H * A = A * A^H = I
    A matrix is unitary if its conjugate transpose equals its inverse.
    
    Args:
        matrix: Square matrix as numpy array or list of lists
        tolerance: Numerical tolerance for floating-point comparisons
        
    Returns:
        bool: True if the matrix is unitary, False otherwise
        
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
    
    # Compute A^H * A
    matrix_h = conjugate_transpose(matrix)
    product = matrix_multiply(matrix_h, matrix)
    
    # Check if A^H * A = I
    identity = identity_matrix(n, dtype=complex if np.iscomplexobj(matrix) else matrix.dtype)
    
    for i in range(n):
        for j in range(n):
            if abs(product[i, j] - identity[i, j]) > tolerance:
                return False
    
    # Optionally check A * A^H = I as well for numerical stability
    product2 = matrix_multiply(matrix, matrix_h)
    
    for i in range(n):
        for j in range(n):
            if abs(product2[i, j] - identity[i, j]) > tolerance:
                return False
    
    return True


# Example usage with detailed testing:
if __name__ == "__main__":
    # Example 1: An orthogonal matrix (rotation matrix)
    theta = np.pi / 4  # 45-degree rotation
    orthogonal_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Example 2: A unitary matrix
    unitary_matrix = np.array([
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [1j/np.sqrt(2), -1j/np.sqrt(2)]
    ], dtype=complex)
    
    # Example 3: A real orthogonal matrix that is also unitary
    real_orthogonal = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    
    # Example 4: A non-orthogonal, non-unitary matrix
    neither_matrix = np.array([
        [1, 2],
        [3, 4]
    ])
    
    # Example 5: Identity matrix (both orthogonal and unitary)
    identity = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Example 6: A complex matrix that is unitary but not orthogonal
    complex_unitary = np.array([
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [-1/np.sqrt(2), 1/np.sqrt(2)]
    ]) * np.exp(1j * np.pi / 4)
    
    # Example 7: Almost orthogonal matrix (with small numerical errors)
    almost_orthogonal = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta) + 1e-11]
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
                # Test if the matrix is orthogonal
                orthogonal = is_orthogonal(matrix)
                print(f"Is orthogonal: {orthogonal}")
                
                # Test if the matrix is unitary
                unitary = is_unitary(matrix)
                print(f"Is unitary: {unitary}")
                
                # Verification
                print("\nVerification:")
                
                # For orthogonality: A^T * A = I
                if not np.iscomplexobj(matrix):
                    matrix_t = matrix_transpose(matrix)
                    product = matrix_multiply(matrix_t, matrix)
                    print("A^T * A =")
                    print(product)
                    error = np.max(np.abs(product - identity_matrix(matrix.shape[0], matrix.dtype)))
                    print(f"Maximum deviation from identity: {error}")
                
                # For unitarity: A^H * A = I
                matrix_h = conjugate_transpose(matrix)
                product = matrix_multiply(matrix_h, matrix)
                print("A^H * A =")
                print(product)
                error = np.max(np.abs(product - identity_matrix(matrix.shape[0], 
                                                              dtype=complex if np.iscomplexobj(matrix) else matrix.dtype)))
                print(f"Maximum deviation from identity: {error}")
                
                # Preservation of vector lengths (for both orthogonal and unitary matrices)
                if orthogonal or unitary:
                    print("\nLength preservation check:")
                    # Create a random vector
                    v = np.random.rand(matrix.shape[1])
                    if np.iscomplexobj(matrix):
                        v = v + 1j * np.random.rand(matrix.shape[1])
                    
                    # Calculate the original length
                    v_length = np.sqrt(np.sum(np.abs(v)**2))
                    
                    # Apply the transformation
                    Av = np.zeros(matrix.shape[0], dtype=np.result_type(matrix.dtype, v.dtype))
                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[1]):
                            Av[i] += matrix[i, j] * v[j]
                    
                    # Calculate the new length
                    Av_length = np.sqrt(np.sum(np.abs(Av)**2))
                    
                    print(f"Original vector length: {v_length}")
                    print(f"Transformed vector length: {Av_length}")
                    print(f"Ratio: {Av_length/v_length}")
                    print(f"Length preserved? {abs(Av_length/v_length - 1) < 1e-10}")
        
        except ValueError as e:
            print(f"Error: {e}")
    
    # Run tests on all example matrices
    test_matrix(orthogonal_matrix, "Orthogonal Matrix (Rotation Matrix)")
    test_matrix(unitary_matrix, "Unitary Matrix")
    test_matrix(real_orthogonal, "Real Orthogonal Matrix (also Unitary)")
    test_matrix(neither_matrix, "Neither Orthogonal nor Unitary")
    test_matrix(identity, "Identity Matrix (both Orthogonal and Unitary)")
    test_matrix(complex_unitary, "Complex Unitary Matrix (not Orthogonal)")
    test_matrix(almost_orthogonal, "Almost Orthogonal Matrix")