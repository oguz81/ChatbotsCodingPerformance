import numpy as np

def determinant_7x7(matrix):
    """
    Calculate the determinant of a 7x7 matrix.
    
    Args:
        matrix (list or numpy.ndarray): A 7x7 matrix.
    
    Returns:
        float: Determinant of the matrix.
    """
    if len(matrix) != 7 or any(len(row) != 7 for row in matrix):
        raise ValueError("Input matrix must be 7x7.")
    return np.linalg.det(np.array(matrix))

def determinant_11x11(matrix):
    """
    Calculate the determinant of an 11x11 matrix.
    
    Args:
        matrix (list or numpy.ndarray): An 11x11 matrix.
    
    Returns:
        float: Determinant of the matrix.
    """
    if len(matrix) != 11 or any(len(row) != 11 for row in matrix):
        raise ValueError("Input matrix must be 11x11.")
    return np.linalg.det(np.array(matrix))

# Example usage:
if __name__ == "__main__":
    # Example 7x7 matrix
    matrix_7x7 = [
        [1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1],
        [2, 4, 6, 8, 10, 12, 14],
        [14, 12, 10, 8, 6, 4, 2],
        [3, 6, 9, 12, 15, 18, 21],
        [21, 18, 15, 12, 9, 6, 3],
        [4, 8, 12, 16, 20, 24, 28]
    ]
    
    # Example 11x11 matrix
    matrix_11x11 = [
        [i + j for j in range(11)] for i in range(11)
    ]
    
    print("Determinant of 7x7 matrix:", determinant_7x7(matrix_7x7))
    print("Determinant of 11x11 matrix:", determinant_11x11(matrix_11x11))