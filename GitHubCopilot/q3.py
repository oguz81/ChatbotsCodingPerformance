import numpy as np

def multiply_6x4_4x8(matrix_a, matrix_b):
    """
    Multiply a 6x4 matrix with a 4x8 matrix.

    Args:
        matrix_a (list or numpy.ndarray): A 6x4 matrix.
        matrix_b (list or numpy.ndarray): A 4x8 matrix.

    Returns:
        numpy.ndarray: The resulting 6x8 matrix after multiplication.
    """
    if len(matrix_a) != 6 or any(len(row) != 4 for row in matrix_a):
        raise ValueError("Matrix A must be 6x4.")
    if len(matrix_b) != 4 or any(len(row) != 8 for row in matrix_b):
        raise ValueError("Matrix B must be 4x8.")
    
    return np.dot(np.array(matrix_a), np.array(matrix_b))

def multiply_12x4_4x16(matrix_a, matrix_b):
    """
    Multiply a 12x4 matrix with a 4x16 matrix.

    Args:
        matrix_a (list or numpy.ndarray): A 12x4 matrix.
        matrix_b (list or numpy.ndarray): A 4x16 matrix.

    Returns:
        numpy.ndarray: The resulting 12x16 matrix after multiplication.
    """
    if len(matrix_a) != 12 or any(len(row) != 4 for row in matrix_a):
        raise ValueError("Matrix A must be 12x4.")
    if len(matrix_b) != 4 or any(len(row) != 16 for row in matrix_b):
        raise ValueError("Matrix B must be 4x16.")
    
    return np.dot(np.array(matrix_a), np.array(matrix_b))

def multiply_5x20_20x4(matrix_a, matrix_b):
    """
    Multiply a 5x20 matrix with a 20x4 matrix.

    Args:
        matrix_a (list or numpy.ndarray): A 5x20 matrix.
        matrix_b (list or numpy.ndarray): A 20x4 matrix.

    Returns:
        numpy.ndarray: The resulting 5x4 matrix after multiplication.
    """
    if len(matrix_a) != 5 or any(len(row) != 20 for row in matrix_a):
        raise ValueError("Matrix A must be 5x20.")
    if len(matrix_b) != 20 or any(len(row) != 4 for row in matrix_b):
        raise ValueError("Matrix B must be 20x4.")
    
    return np.dot(np.array(matrix_a), np.array(matrix_b))

# Example usage:
if __name__ == "__main__":
    # Example 6x4 and 4x8 matrices
    matrix_a_6x4 = np.random.randint(0, 10, (6, 4)).tolist()
    matrix_b_4x8 = np.random.randint(0, 10, (4, 8)).tolist()

    # Example 12x4 and 4x16 matrices
    matrix_a_12x4 = np.random.randint(0, 10, (12, 4)).tolist()
    matrix_b_4x16 = np.random.randint(0, 10, (4, 16)).tolist()

    # Example 5x20 and 20x4 matrices
    matrix_a_5x20 = np.random.randint(0, 10, (5, 20)).tolist()
    matrix_b_20x4 = np.random.randint(0, 10, (20, 4)).tolist()

    print("6x4 and 4x8 Matrix Multiplication Result:")
    print(multiply_6x4_4x8(matrix_a_6x4, matrix_b_4x8))

    print("\n12x4 and 4x16 Matrix Multiplication Result:")
    print(multiply_12x4_4x16(matrix_a_12x4, matrix_b_4x16))

    print("\n5x20 and 20x4 Matrix Multiplication Result:")
    print(multiply_5x20_20x4(matrix_a_5x20, matrix_b_20x4))