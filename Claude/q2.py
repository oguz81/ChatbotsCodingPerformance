import numpy as np

def multiply_matrices_7x7(matrix_a, matrix_b):
    """
    Multiply two 7x7 matrices.
    
    Args:
        matrix_a: First 7x7 matrix (numpy array or list of lists)
        matrix_b: Second 7x7 matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The resulting 7x7 matrix after multiplication
    """
    # Convert to numpy arrays if they're not already
    if not isinstance(matrix_a, np.ndarray):
        matrix_a = np.array(matrix_a, dtype=float)
    if not isinstance(matrix_b, np.ndarray):
        matrix_b = np.array(matrix_b, dtype=float)
    
    # Check dimensions
    if matrix_a.shape != (7, 7) or matrix_b.shape != (7, 7):
        raise ValueError("Both inputs must be 7x7 matrices")
    
    # Initialize result matrix
    result = np.zeros((7, 7))
    
    # Matrix multiplication
    for i in range(7):
        for j in range(7):
            for k in range(7):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
                
    return result


def multiply_matrices_15x15(matrix_a, matrix_b):
    """
    Multiply two 15x15 matrices.
    
    Args:
        matrix_a: First 15x15 matrix (numpy array or list of lists)
        matrix_b: Second 15x15 matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The resulting 15x15 matrix after multiplication
    """
    # Convert to numpy arrays if they're not already
    if not isinstance(matrix_a, np.ndarray):
        matrix_a = np.array(matrix_a, dtype=float)
    if not isinstance(matrix_b, np.ndarray):
        matrix_b = np.array(matrix_b, dtype=float)
    
    # Check dimensions
    if matrix_a.shape != (15, 15) or matrix_b.shape != (15, 15):
        raise ValueError("Both inputs must be 15x15 matrices")
    
    # Initialize result matrix
    result = np.zeros((15, 15))
    
    # Matrix multiplication
    for i in range(15):
        for j in range(15):
            for k in range(15):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
                
    return result


def multiply_matrices_20x20(matrix_a, matrix_b):
    """
    Multiply two 20x20 matrices.
    
    Args:
        matrix_a: First 20x20 matrix (numpy array or list of lists)
        matrix_b: Second 20x20 matrix (numpy array or list of lists)
        
    Returns:
        numpy.ndarray: The resulting 20x20 matrix after multiplication
    """
    # Convert to numpy arrays if they're not already
    if not isinstance(matrix_a, np.ndarray):
        matrix_a = np.array(matrix_a, dtype=float)
    if not isinstance(matrix_b, np.ndarray):
        matrix_b = np.array(matrix_b, dtype=float)
    
    # Check dimensions
    if matrix_a.shape != (20, 20) or matrix_b.shape != (20, 20):
        raise ValueError("Both inputs must be 20x20 matrices")
    
    # Initialize result matrix
    result = np.zeros((20, 20))
    
    # Matrix multiplication
    for i in range(20):
        for j in range(20):
            for k in range(20):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
                
    return result


# Example usage:
if __name__ == "__main__":
    # Example 7x7 matrices
    matrix_a_7x7 = np.random.rand(7, 7)
    matrix_b_7x7 = np.random.rand(7, 7)
    result_7x7 = multiply_matrices_7x7(matrix_a_7x7, matrix_b_7x7)
    print("7x7 Matrix Multiplication Result:")
    print(result_7x7)
    
    # Example 15x15 matrices
    matrix_a_15x15 = np.random.rand(15, 15)
    matrix_b_15x15 = np.random.rand(15, 15)
    result_15x15 = multiply_matrices_15x15(matrix_a_15x15, matrix_b_15x15)
    print("\n15x15 Matrix Multiplication Result:")
    print(result_15x15)
    
    # Example 20x20 matrices
    matrix_a_20x20 = np.random.rand(20, 20)
    matrix_b_20x20 = np.random.rand(20, 20)
    result_20x20 = multiply_matrices_20x20(matrix_a_20x20, matrix_b_20x20)
    print("\n20x20 Matrix Multiplication Result:")
    print(result_20x20)
    
    # Verify results with NumPy's built-in function
    print("\nVerification with NumPy:")
    print("7x7 Match:", np.allclose(result_7x7, np.matmul(matrix_a_7x7, matrix_b_7x7)))
    print("15x15 Match:", np.allclose(result_15x15, np.matmul(matrix_a_15x15, matrix_b_15x15)))
    print("20x20 Match:", np.allclose(result_20x20, np.matmul(matrix_a_20x20, matrix_b_20x20)))