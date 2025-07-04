import numpy as np

def multiply_6x4_and_4x8(matrix_a, matrix_b):
    """
    Multiply a 6x4 matrix with a 4x8 matrix.
    
    Args:
        matrix_a: A 6x4 matrix (numpy array or list of lists)
        matrix_b: A 4x8 matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The resulting 6x8 matrix after multiplication
    """
    # Convert to numpy arrays if they're not already
    if not isinstance(matrix_a, np.ndarray):
        matrix_a = np.array(matrix_a, dtype=float)
    if not isinstance(matrix_b, np.ndarray):
        matrix_b = np.array(matrix_b, dtype=float)
    
    # Check dimensions
    if matrix_a.shape != (6, 4) or matrix_b.shape != (4, 8):
        raise ValueError("Matrix A must be 6x4 and Matrix B must be 4x8")
    
    # Initialize result matrix (6x8)
    result = np.zeros((6, 8))
    
    # Matrix multiplication
    for i in range(6):
        for j in range(8):
            for k in range(4):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
                
    return result


def multiply_12x4_and_4x16(matrix_a, matrix_b):
    """
    Multiply a 12x4 matrix with a 4x16 matrix.
    
    Args:
        matrix_a: A 12x4 matrix (numpy array or list of lists)
        matrix_b: A 4x16 matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The resulting 12x16 matrix after multiplication
    """
    # Convert to numpy arrays if they're not already
    if not isinstance(matrix_a, np.ndarray):
        matrix_a = np.array(matrix_a, dtype=float)
    if not isinstance(matrix_b, np.ndarray):
        matrix_b = np.array(matrix_b, dtype=float)
    
    # Check dimensions
    if matrix_a.shape != (12, 4) or matrix_b.shape != (4, 16):
        raise ValueError("Matrix A must be 12x4 and Matrix B must be 4x16")
    
    # Initialize result matrix (12x16)
    result = np.zeros((12, 16))
    
    # Matrix multiplication
    for i in range(12):
        for j in range(16):
            for k in range(4):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
                
    return result


def multiply_5x20_and_20x4(matrix_a, matrix_b):
    """
    Multiply a 5x20 matrix with a 20x4 matrix.
    
    Args:
        matrix_a: A 5x20 matrix (numpy array or list of lists)
        matrix_b: A 20x4 matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The resulting 5x4 matrix after multiplication
    """
    # Convert to numpy arrays if they're not already
    if not isinstance(matrix_a, np.ndarray):
        matrix_a = np.array(matrix_a, dtype=float)
    if not isinstance(matrix_b, np.ndarray):
        matrix_b = np.array(matrix_b, dtype=float)
    
    # Check dimensions
    if matrix_a.shape != (5, 20) or matrix_b.shape != (20, 4):
        raise ValueError("Matrix A must be 5x20 and Matrix B must be 20x4")
    
    # Initialize result matrix (5x4)
    result = np.zeros((5, 4))
    
    # Matrix multiplication
    for i in range(5):
        for j in range(4):
            for k in range(20):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
                
    return result


# Example usage:
if __name__ == "__main__":
    # Example 6x4 and 4x8 matrices
    matrix_a_6x4 = np.random.rand(6, 4)
    matrix_b_4x8 = np.random.rand(4, 8)
    result_6x8 = multiply_6x4_and_4x8(matrix_a_6x4, matrix_b_4x8)
    print("Result of 6x4 × 4x8 multiplication (6x8 matrix):")
    print(result_6x8)
    
    # Example 12x4 and 4x16 matrices
    matrix_a_12x4 = np.random.rand(12, 4)
    matrix_b_4x16 = np.random.rand(4, 16)
    result_12x16 = multiply_12x4_and_4x16(matrix_a_12x4, matrix_b_4x16)
    print("\nResult of 12x4 × 4x16 multiplication (12x16 matrix):")
    print(result_12x16)
    
    # Example 5x20 and 20x4 matrices
    matrix_a_5x20 = np.random.rand(5, 20)
    matrix_b_20x4 = np.random.rand(20, 4)
    result_5x4 = multiply_5x20_and_20x4(matrix_a_5x20, matrix_b_20x4)
    print("\nResult of 5x20 × 20x4 multiplication (5x4 matrix):")
    print(result_5x4)
    
    # Verify results with NumPy's built-in function
    print("\nVerification with NumPy:")
    print("6x8 Match:", np.allclose(result_6x8, np.matmul(matrix_a_6x4, matrix_b_4x8)))
    print("12x16 Match:", np.allclose(result_12x16, np.matmul(matrix_a_12x4, matrix_b_4x16)))
    print("5x4 Match:", np.allclose(result_5x4, np.matmul(matrix_a_5x20, matrix_b_20x4)))