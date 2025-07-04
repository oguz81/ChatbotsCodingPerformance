def multiply_6x4_4x8(matrix_a, matrix_b):
    """
    Multiply a 6x4 matrix with a 4x8 matrix.
    
    Args:
        matrix_a: 6x4 matrix (list of lists)
        matrix_b: 4x8 matrix (list of lists)
        
    Returns:
        6x8 result matrix (list of lists)
    """
    rows_a = 6
    cols_a = 4
    cols_b = 8
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result

def multiply_12x4_4x16(matrix_a, matrix_b):
    """
    Multiply a 12x4 matrix with a 4x16 matrix.
    
    Args:
        matrix_a: 12x4 matrix (list of lists)
        matrix_b: 4x16 matrix (list of lists)
        
    Returns:
        12x16 result matrix (list of lists)
    """
    rows_a = 12
    cols_a = 4
    cols_b = 16
    
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result

def multiply_5x20_20x4(matrix_a, matrix_b):
    """
    Multiply a 5x20 matrix with a 20x4 matrix.
    
    Args:
        matrix_a: 5x20 matrix (list of lists)
        matrix_b: 20x4 matrix (list of lists)
        
    Returns:
        5x4 result matrix (list of lists)
    """
    rows_a = 5
    cols_a = 20
    cols_b = 4
    
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result

# Helper function to create test matrices
def create_matrix(rows, cols, value_func):
    """Create a test matrix with specified dimensions"""
    return [[value_func(i, j) for j in range(cols)] for i in range(rows)]

# Test cases
if __name__ == "__main__":
    # Test 6x4 * 4x8
    matrix_6x4 = create_matrix(6, 4, lambda i, j: i + j)
    matrix_4x8 = create_matrix(4, 8, lambda i, j: i * j)
    result_6x8 = multiply_6x4_4x8(matrix_6x4, matrix_4x8)
    print("6x4 * 4x8 result dimensions:", len(result_6x8), "x", len(result_6x8[0]))
    print("Sample element [3][5]:", result_6x8[3][5])
    
    # Test 12x4 * 4x16
    matrix_12x4 = create_matrix(12, 4, lambda i, j: i - j)
    matrix_4x16 = create_matrix(4, 16, lambda i, j: i + j + 1)
    result_12x16 = multiply_12x4_4x16(matrix_12x4, matrix_4x16)
    print("\n12x4 * 4x16 result dimensions:", len(result_12x16), "x", len(result_12x16[0]))
    print("Sample element [7][10]:", result_12x16[7][10])
    
    # Test 5x20 * 20x4
    matrix_5x20 = create_matrix(5, 20, lambda i, j: (i + 1) * (j + 1))
    matrix_20x4 = create_matrix(20, 4, lambda i, j: i - j)
    result_5x4 = multiply_5x20_20x4(matrix_5x20, matrix_20x4)
    print("\n5x20 * 20x4 result dimensions:", len(result_5x4), "x", len(result_5x4[0]))
    print("Sample element [2][3]:", result_5x4[2][3])