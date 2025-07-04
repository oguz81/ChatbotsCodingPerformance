import numpy as np

def multiply_7x7(matrix_a, matrix_b):
    """
    Multiply two 7x7 matrices.

    Args:
        matrix_a (list or numpy.ndarray): A 7x7 matrix.
        matrix_b (list or numpy.ndarray): A 7x7 matrix.

    Returns:
        numpy.ndarray: The resulting 7x7 matrix after multiplication.
    """
    if len(matrix_a) != 7 or any(len(row) != 7 for row in matrix_a):
        raise ValueError("Matrix A must be 7x7.")
    if len(matrix_b) != 7 or any(len(row) != 7 for row in matrix_b):
        raise ValueError("Matrix B must be 7x7.")
    
    return np.dot(np.array(matrix_a), np.array(matrix_b))

def multiply_15x15(matrix_a, matrix_b):
    """
    Multiply two 15x15 matrices.

    Args:
        matrix_a (list or numpy.ndarray): A 15x15 matrix.
        matrix_b (list or numpy.ndarray): A 15x15 matrix.

    Returns:
        numpy.ndarray: The resulting 15x15 matrix after multiplication.
    """
    if len(matrix_a) != 15 or any(len(row) != 15 for row in matrix_a):
        raise ValueError("Matrix A must be 15x15.")
    if len(matrix_b) != 15 or any(len(row) != 15 for row in matrix_b):
        raise ValueError("Matrix B must be 15x15.")
    
    return np.dot(np.array(matrix_a), np.array(matrix_b))

def multiply_20x20(matrix_a, matrix_b):
    """
    Multiply two 20x20 matrices.

    Args:
        matrix_a (list or numpy.ndarray): A 20x20 matrix.
        matrix_b (list or numpy.ndarray): A 20x20 matrix.

    Returns:
        numpy.ndarray: The resulting 20x20 matrix after multiplication.
    """
    if len(matrix_a) != 20 or any(len(row) != 20 for row in matrix_a):
        raise ValueError("Matrix A must be 20x20.")
    if len(matrix_b) != 20 or any(len(row) != 20 for row in matrix_b):
        raise ValueError("Matrix B must be 20x20.")
    
    return np.dot(np.array(matrix_a), np.array(matrix_b))

# Example usage:
if __name__ == "__main__":
    # Example 7x7 matrices
    matrix_a_7x7 = np.random.randint(0, 10, (7, 7)).tolist()
    matrix_b_7x7 = np.random.randint(0, 10, (7, 7)).tolist()
    
    # Example 15x15 matrices
    matrix_a_15x15 = np.random.randint(0, 10, (15, 15)).tolist()
    matrix_b_15x15 = np.random.randint(0, 10, (15, 15)).tolist()
    
    # Example 20x20 matrices
    matrix_a_20x20 = np.random.randint(0, 10, (20, 20)).tolist()
    matrix_b_20x20 = np.random.randint(0, 10, (20, 20)).tolist()
    
    print("7x7 Matrix Multiplication Result:")
    print(multiply_7x7(matrix_a_7x7, matrix_b_7x7))

    print("\n15x15 Matrix Multiplication Result:")
    print(multiply_15x15(matrix_a_15x15, matrix_b_15x15))
    
    print("\n20x20 Matrix Multiplication Result:")
    print(multiply_20x20(matrix_a_20x20, matrix_b_20x20))