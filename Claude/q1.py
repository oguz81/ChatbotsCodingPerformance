import numpy as np

def determinant_7x7(matrix):
    """
    Calculate the determinant of a 7x7 matrix.
    
    Args:
        matrix: A 7x7 numpy array or list of lists
        
    Returns:
        float: The determinant of the matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=float)
    
    # Check dimensions
    if matrix.shape != (7, 7):
        raise ValueError("Input must be a 7x7 matrix")
    
    return np.linalg.det(matrix)


def determinant_11x11(matrix):
    """
    Calculate the determinant of a 11x11 matrix.
    
    Args:
        matrix: A 11x11 numpy array or list of lists
        
    Returns:
        float: The determinant of the matrix
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=float)
    
    # Check dimensions
    if matrix.shape != (11, 11):
        raise ValueError("Input must be a 11x11 matrix")
    
    return np.linalg.det(matrix)


# Example usage:
if __name__ == "__main__":
    # Example 7x7 matrix
    matrix_7x7 = np.random.rand(7, 7)
    print("7x7 Matrix:")
    print(matrix_7x7)
    print("Determinant:", determinant_7x7(matrix_7x7))
    
    # Example 11x11 matrix
    matrix_11x11 = np.random.rand(11, 11)
    print("\n11x11 Matrix:")
    print(matrix_11x11)
    print("Determinant:", determinant_11x11(matrix_11x11))