import cmath

def is_orthogonal(matrix, tolerance=1e-9):
    """
    Checks if a square matrix is orthogonal.

    A real square matrix A is orthogonal if its transpose is equal to its inverse: A^T = A^-1,
    which implies A * A^T = I (the identity matrix).

    Args:
        matrix: A list of lists representing the real square matrix.
        tolerance: A small value to account for floating-point inaccuracies.

    Returns:
        True if the matrix is orthogonal, False otherwise.
    """
    rows = len(matrix)
    if rows == 0:
        return True  # An empty matrix can be considered orthogonal

    # Check if it's a square matrix
    for row in matrix:
        if len(row) != rows:
            return False

    identity = [[1.0 if i == j else 0.0 for j in range(rows)] for i in range(rows)]
    transpose = [[matrix[j][i] for j in range(rows)] for i in range(rows)]

    # Multiply the matrix by its transpose
    product = [[sum(matrix[i][k] * transpose[k][j] for k in range(rows))
                for j in range(rows)] for i in range(rows)]

    # Check if the product is close to the identity matrix
    for i in range(rows):
        for j in range(rows):
            if abs(product[i][j] - identity[i][j]) > tolerance:
                return False
    return True

def is_unitary(matrix, tolerance=1e-9):
    """
    Checks if a square matrix is unitary.

    A complex square matrix A is unitary if its conjugate transpose (Hermitian conjugate)
    is equal to its inverse: A^H = A^-1, which implies A * A^H = I (the identity matrix).

    Args:
        matrix: A list of lists representing the complex square matrix.
        tolerance: A small value to account for floating-point inaccuracies.

    Returns:
        True if the matrix is unitary, False otherwise.
    """
    rows = len(matrix)
    if rows == 0:
        return True  # An empty matrix can be considered unitary

    # Check if it's a square matrix
    for row in matrix:
        if len(row) != rows:
            return False

    identity = [[1.0 if i == j else 0.0 for j in range(rows)] for i in range(rows)]
    conjugate_transpose = [[cmath.conjugate(complex(matrix[j][i])) for j in range(rows)]
                           for i in range(rows)]

    # Multiply the matrix by its conjugate transpose
    product = [[sum(complex(matrix[i][k]) * conjugate_transpose[k][j] for k in range(rows))
                for j in range(rows)] for i in range(rows)]

    # Check if the product is close to the identity matrix
    for i in range(rows):
        for j in range(rows):
            if abs(product[i][j] - identity[i][j]) > tolerance:
                return False
    return True

if __name__ == "__main__":
    # Orthogonal matrices
    orthogonal_matrix1 = [
        [1, 0],
        [0, 1]
    ]
    print(f"Is orthogonal_matrix1 orthogonal? {is_orthogonal(orthogonal_matrix1)}")

    orthogonal_matrix2 = [
        [0, -1],
        [1, 0]
    ]
    print(f"Is orthogonal_matrix2 orthogonal? {is_orthogonal(orthogonal_matrix2)}")

    orthogonal_matrix3 = [
        [0.8, 0.6],
        [-0.6, 0.8]
    ]
    print(f"Is orthogonal_matrix3 orthogonal? {is_orthogonal(orthogonal_matrix3)}")

    not_orthogonal_matrix1 = [
        [1, 1],
        [0, 1]
    ]
    print(f"Is not_orthogonal_matrix1 orthogonal? {is_orthogonal(not_orthogonal_matrix1)}")

    # Unitary matrices
    unitary_matrix1 = [
        [1, 0],
        [0, 1]
    ]
    print(f"Is unitary_matrix1 unitary? {is_unitary(unitary_matrix1)}")

    unitary_matrix2 = [
        [1j, 0],
        [0, -1j]
    ]
    print(f"Is unitary_matrix2 unitary? {is_unitary(unitary_matrix2)}")

    unitary_matrix3 = [
        [1/cmath.sqrt(2), 1j/cmath.sqrt(2)],
        [1j/cmath.sqrt(2), 1/cmath.sqrt(2)]
    ]
    print(f"Is unitary_matrix3 unitary? {is_unitary(unitary_matrix3)}")

    not_unitary_matrix1 = [
        [1, 1j],
        [0, 1]
    ]
    print(f"Is not_unitary_matrix1 unitary? {is_unitary(not_unitary_matrix1)}")