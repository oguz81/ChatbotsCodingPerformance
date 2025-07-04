def multiply_7x7(matrix_a, matrix_b):
    """
    Multiply two 7x7 matrices.
    
    Args:
        matrix_a: First 7x7 matrix (list of lists)
        matrix_b: Second 7x7 matrix (list of lists)
        
    Returns:
        The product matrix (7x7 list of lists)
    """
    size = 7
    result = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result

def multiply_15x15(matrix_a, matrix_b):
    """
    Multiply two 15x15 matrices.
    
    Args:
        matrix_a: First 15x15 matrix (list of lists)
        matrix_b: Second 15x15 matrix (list of lists)
        
    Returns:
        The product matrix (15x15 list of lists)
    """
    size = 15
    result = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result

def multiply_20x20(matrix_a, matrix_b):
    """
    Multiply two 20x20 matrices.
    
    Args:
        matrix_a: First 20x20 matrix (list of lists)
        matrix_b: Second 20x20 matrix (list of lists)
        
    Returns:
        The product matrix (20x20 list of lists)
    """
    size = 20
    result = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result

# Example usage:

# Create identity matrices for testing
def create_identity(size):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

# Test 7x7 multiplication
matrix_a_7x7 = create_identity(7)
matrix_b_7x7 = create_identity(7)
product_7x7 = multiply_7x7(matrix_a_7x7, matrix_b_7x7)
print("7x7 Identity × Identity (first row):", product_7x7[0])  # Should print [1, 0, 0, 0, 0, 0, 0]

# Test 15x15 multiplication
matrix_a_15x15 = create_identity(15)
matrix_b_15x15 = create_identity(15)
product_15x15 = multiply_15x15(matrix_a_15x15, matrix_b_15x15)
print("15x15 Identity × Identity (first row):", product_15x15[0][:5])  # First 5 elements of first row

# Test 20x20 multiplication
matrix_a_20x20 = create_identity(20)
matrix_b_20x20 = create_identity(20)
product_20x20 = multiply_20x20(matrix_a_20x20, matrix_b_20x20)
print("20x20 Identity × Identity (first row):", product_20x20[0][:5])  # First 5 elements of first row