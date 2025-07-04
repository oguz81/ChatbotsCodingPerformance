def is_hermitian(matrix):
    """
    Check if a square matrix is Hermitian.

    Args:
        matrix (list of list): A square matrix (with complex numbers).

    Returns:
        bool: True if the matrix is Hermitian, False otherwise.
    """
    n = len(matrix)
    
    # Check if the matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("The input matrix must be square.")
    
    # Check Hermitian property: A[i][j] == conjugate(A[j][i]) for all i, j
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != complex(matrix[j][i]).conjugate():
                return False
    return True


def is_anti_hermitian(matrix):
    """
    Check if a square matrix is anti-Hermitian.

    Args:
        matrix (list of list): A square matrix (with complex numbers).

    Returns:
        bool: True if the matrix is anti-Hermitian, False otherwise.
    """
    n = len(matrix)
    
    # Check if the matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("The input matrix must be square.")
    
    # Check anti-Hermitian property: A[i][j] == -conjugate(A[j][i]) for all i, j
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != -complex(matrix[j][i]).conjugate():
                return False
    return True


# Example usage
if __name__ == "__main__":
    hermitian_matrix = [
        [1+0j, 2-1j, 3+4j],
        [2+1j, 5+0j, 6-2j],
        [3-4j, 6+2j, 8+0j]
    ]
    
    anti_hermitian_matrix = [
        [0+0j, -2+1j, -3-4j],
        [2-1j, 0+0j, -6+2j],
        [3+4j, 6-2j, 0+0j]
    ]
    
    non_square_matrix = [
        [1, 2],
        [3, 4, 5]
    ]

    print("Hermitian Matrix Check:")
    print(is_hermitian(hermitian_matrix))  # Should return True
    print(is_hermitian(anti_hermitian_matrix))  # Should return False

    print("\nAnti-Hermitian Matrix Check:")
    print(is_anti_hermitian(anti_hermitian_matrix))  # Should return True
    print(is_anti_hermitian(hermitian_matrix))  # Should return False

    try:
        print("\nNon-Square Matrix Check:")
        print(is_hermitian(non_square_matrix))  # Should raise an error
    except ValueError as e:
        print(e)