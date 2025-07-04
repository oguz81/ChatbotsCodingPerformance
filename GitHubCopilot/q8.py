def is_orthogonal(matrix):
    """
    Check if a square matrix is orthogonal.

    A matrix is orthogonal if A * A^T = I, 
    where A^T is the transpose of the matrix and I is the identity matrix.

    Args:
        matrix (list of list): A square matrix.

    Returns:
        bool: True if the matrix is orthogonal, False otherwise.
    """
    n = len(matrix)

    # Check if the matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("The input matrix must be square.")
    
    # Compute A * A^T
    transpose = [[matrix[j][i] for j in range(n)] for i in range(n)]  # Transpose of the matrix
    product = [[sum(matrix[i][k] * transpose[k][j] for k in range(n)) for j in range(n)] for i in range(n)]  # A * A^T

    # Compare the product with the identity matrix
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    return all(abs(product[i][j] - identity[i][j]) < 1e-9 for i in range(n) for j in range(n))  # Allow for floating-point errors


def is_unitary(matrix):
    """
    Check if a square matrix is unitary.

    A matrix is unitary if A * A^H = I,
    where A^H is the conjugate transpose (Hermitian transpose) of the matrix and I is the identity matrix.

    Args:
        matrix (list of list): A square matrix with complex numbers.

    Returns:
        bool: True if the matrix is unitary, False otherwise.
    """
    n = len(matrix)

    # Check if the matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("The input matrix must be square.")
    
    # Compute A^H (conjugate transpose)
    conjugate_transpose = [[complex(matrix[j][i]).conjugate() for j in range(n)] for i in range(n)]
    
    # Compute A * A^H
    product = [[sum(matrix[i][k] * conjugate_transpose[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

    # Compare the product with the identity matrix
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    return all(abs(product[i][j] - identity[i][j]) < 1e-9 for i in range(n) for j in range(n))  # Allow for floating-point errors


# Example usage
if __name__ == "__main__":
    orthogonal_matrix = [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]
    
    unitary_matrix = [
        [1/2**0.5 + 0j, 1/2**0.5 + 0j],
        [1/2**0.5 + 0j, -1/2**0.5 + 0j]
    ]
    
    non_square_matrix = [
        [1, 2],
        [3, 4, 5]
    ]

    print("Orthogonal Matrix Check:")
    print(is_orthogonal(orthogonal_matrix))  # Should return True

    print("\nUnitary Matrix Check:")
    print(is_unitary(unitary_matrix))  # Should return True

    try:
        print("\nNon-Square Matrix Check:")
        print(is_orthogonal(non_square_matrix))  # Should raise an error
    except ValueError as e:
        print(e)