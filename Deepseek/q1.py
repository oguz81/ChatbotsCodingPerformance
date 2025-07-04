def determinant_7x7(matrix):
    """
    Calculate the determinant of a 7x7 matrix using recursive approach.
    
    Args:
        matrix: A 7x7 list of lists (square matrix)
        
    Returns:
        float: The determinant of the matrix
    """
    # Base case for 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]
    
    # Base case for 2x2 matrix (for recursion)
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for col in range(len(matrix)):
        # Calculate the minor matrix
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        # Recursive call for determinant of minor
        det += ((-1) ** col) * matrix[0][col] * determinant_7x7(minor)
    
    return det

def determinant_11x11(matrix):
    """
    Calculate the determinant of an 11x11 matrix using recursive approach.
    
    Args:
        matrix: An 11x11 list of lists (square matrix)
        
    Returns:
        float: The determinant of the matrix
    """
    # Base case for 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]
    
    # Base case for 2x2 matrix (for recursion)
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for col in range(len(matrix)):
        # Calculate the minor matrix
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        # Recursive call for determinant of minor
        det += ((-1) ** col) * matrix[0][col] * determinant_11x11(minor)
    
    return det

# Example usage:

# Create a 7x7 identity matrix
matrix_7x7 = [[1 if i == j else 0 for j in range(7)] for i in range(7)]
print("Determinant of 7x7 identity matrix:", determinant_7x7(matrix_7x7))  # Should print 1.0

# Create an 11x11 identity matrix
matrix_11x11 = [[1 if i == j else 0 for j in range(11)] for i in range(11)]
print("Determinant of 11x11 identity matrix:", determinant_11x11(matrix_11x11))  # Should print 1.0

# Create a more complex 7x7 matrix
complex_7x7 = [
    [1, 2, 3, 4, 5, 6, 7],
    [2, 3, 4, 5, 6, 7, 1],
    [3, 4, 5, 6, 7, 1, 2],
    [4, 5, 6, 7, 1, 2, 3],
    [5, 6, 7, 1, 2, 3, 4],
    [6, 7, 1, 2, 3, 4, 5],
    [7, 1, 2, 3, 4, 5, 6]
]
print("Determinant of complex 7x7 matrix:", determinant_7x7(complex_7x7))