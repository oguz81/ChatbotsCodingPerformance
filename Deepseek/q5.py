def matrix_inverse(matrix):
    """
    Calculate the inverse of a square matrix using Gaussian elimination.
    
    Args:
        matrix: A square matrix (list of lists)
        
    Returns:
        The inverse matrix if it exists, otherwise raises ValueError
        
    Raises:
        ValueError: If matrix is not square or is singular (non-invertible)
    """
    n = len(matrix)
    
    # Check if matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square to compute its inverse")
    
    # Create augmented matrix [matrix|identity]
    augmented = [row[:] + [0]*i + [1] + [0]*(n-i-1) for i, row in enumerate(matrix)]
    
    # Perform Gaussian elimination
    for col in range(n):
        # Partial pivoting: find row with maximum element in current column
        max_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        
        # Swap current row with max_row if needed
        if augmented[max_row][col] == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        
        augmented[col], augmented[max_row] = augmented[max_row], augmented[col]
        
        # Normalize the pivot row
        pivot = augmented[col][col]
        augmented[col] = [x / pivot for x in augmented[col]]
        
        # Eliminate other rows
        for r in range(n):
            if r != col and augmented[r][col] != 0:
                factor = augmented[r][col]
                augmented[r] = [x - factor * augmented[col][i] for i, x in enumerate(augmented[r])]
    
    # Extract the inverse from the right half of the augmented matrix
    inverse = [row[n:] for row in augmented]
    
    return inverse

# Example usage:
if __name__ == "__main__":
    # Test matrix
    test_matrix = [
        [4, 7],
        [2, 6]
    ]
    
    print("Original Matrix:")
    for row in test_matrix:
        print(row)
    
    try:
        inv_matrix = matrix_inverse(test_matrix)
        print("\nInverse Matrix:")
        for row in inv_matrix:
            print([round(x, 6) for x in row])  # Rounding for display
    except ValueError as e:
        print(f"\nError: {e}")

    # Another test case
    singular_matrix = [
        [1, 2],
        [2, 4]  # This matrix is singular (determinant = 0)
    ]
    
    print("\nTesting singular matrix:")
    try:
        inv_singular = matrix_inverse(singular_matrix)
    except ValueError as e:
        print(f"Expected error: {e}")