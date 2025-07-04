def is_symmetric(matrix):
    """
    Check if a square matrix is symmetric.

    Args:
        matrix (list of list): A square matrix.

    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """
    n = len(matrix)
    
    # Check if the matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("The input matrix must be square.")
    
    # Check symmetry: A[i][j] == A[j][i] for all i, j
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True


def is_antisymmetric(matrix):
    """
    Check if a square matrix is antisymmetric.

    Args:
        matrix (list of list): A square matrix.

    Returns:
        bool: True if the matrix is antisymmetric, False otherwise.
    """
    n = len(matrix)
    
    # Check if the matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("The input matrix must be square.")
    
    # Check antisymmetry: A[i][j] == -A[j][i] for all i, j
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != -matrix[j][i]:
                return False
    return True


# Example usage
if __name__ == "__main__":
    symmetric_matrix = [
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6]
    ]
    
    antisymmetric_matrix = [
        [0, -2, 3],
        [2, 0, -4],
        [-3, 4, 0]
    ]
    
    non_square_matrix = [
        [1, 2],
        [3, 4, 5]
    ]

    print("Symmetric Matrix Check:")
    print(is_symmetric(symmetric_matrix))  # Should return True
    print(is_symmetric(antisymmetric_matrix))  # Should return False

    print("\nAntisymmetric Matrix Check:")
    print(is_antisymmetric(antisymmetric_matrix))  # Should return True
    print(is_antisymmetric(symmetric_matrix))  # Should return False

    try:
        print("\nNon-Square Matrix Check:")
        print(is_symmetric(non_square_matrix))  # Should raise an error
    except ValueError as e:
        print(e)