def is_hermitian(matrix):
    """
    Checks if a square matrix is Hermitian.

    A matrix A is Hermitian if its conjugate transpose is equal to itself: A^H = A,
    where A^H[i][j] = conjugate(A[j][i]).

    Args:
        matrix: A list of lists representing the square matrix (can contain complex numbers).

    Returns:
        True if the matrix is Hermitian, False otherwise.
    """
    rows = len(matrix)
    if rows == 0:
        return True  # An empty matrix can be considered Hermitian

    # Check if it's a square matrix
    for row in matrix:
        if len(row) != rows:
            return False

    for i in range(rows):
        for j in range(i, rows):
            complex_val_ij = complex(matrix[i][j])
            complex_val_ji = complex(matrix[j][i])
            if complex_val_ij != complex_val_ji.conjugate():
                return False
    return True

def is_anti_hermitian(matrix):
    """
    Checks if a square matrix is anti-Hermitian (or skew-Hermitian).

    A matrix A is anti-Hermitian if its conjugate transpose is equal to the negative of itself: A^H = -A,
    where A^H[i][j] = conjugate(A[j][i]). The diagonal elements must be purely imaginary or zero.

    Args:
        matrix: A list of lists representing the square matrix (can contain complex numbers).

    Returns:
        True if the matrix is anti-Hermitian, False otherwise.
    """
    rows = len(matrix)
    if rows == 0:
        return True  # An empty matrix can be considered anti-Hermitian

    # Check if it's a square matrix
    for row in matrix:
        if len(row) != rows:
            return False

    for i in range(rows):
        for j in range(rows):
            complex_val_ij = complex(matrix[i][j])
            complex_val_ji = complex(matrix[j][i])
            if i == j:
                if complex_val_ij.real != 0:
                    return False
            elif complex_val_ij != -complex_val_ji.conjugate():
                return False
    return True

if __name__ == "__main__":
    # Hermitian matrices
    hermitian_matrix1 = [
        [1, 2 - 1j, 3],
        [2 + 1j, 4, 1 - 2j],
        [3, 1 + 2j, 6]
    ]
    print(f"Is hermitian_matrix1 Hermitian? {is_hermitian(hermitian_matrix1)}")

    hermitian_matrix2 = [
        [5]
    ]
    print(f"Is hermitian_matrix2 Hermitian? {is_hermitian(hermitian_matrix2)}")

    hermitian_matrix3 = []
    print(f"Is hermitian_matrix3 Hermitian? {is_hermitian(hermitian_matrix3)}")

    not_hermitian_matrix1 = [
        [1, 2],
        [3, 1]
    ]
    print(f"Is not_hermitian_matrix1 Hermitian? {is_hermitian(not_hermitian_matrix1)}")

    not_hermitian_matrix2 = [
        [1, 2 - 1j],
        [2 - 1j, 4]
    ]
    print(f"Is not_hermitian_matrix2 Hermitian? {is_hermitian(not_hermitian_matrix2)}")

    # Anti-Hermitian matrices
    anti_hermitian_matrix1 = [
        [0, 1 - 2j, 3j],
        [-1 - 2j, 0, 2 + 1j],
        [-3j, -2 + 1j, 0]
    ]
    print(f"Is anti_hermitian_matrix1 anti-Hermitian? {is_anti_hermitian(anti_hermitian_matrix1)}")

    anti_hermitian_matrix2 = [
        [2j]
    ]
    print(f"Is anti_hermitian_matrix2 anti-Hermitian? {is_anti_hermitian(anti_hermitian_matrix2)}")

    anti_hermitian_matrix3 = []
    print(f"Is anti_hermitian_matrix3 anti-Hermitian? {is_anti_hermitian(anti_hermitian_matrix3)}")

    not_anti_hermitian_matrix1 = [
        [1j, 1],
        [-1, -1j]
    ]
    print(f"Is not_anti_hermitian_matrix1 anti-Hermitian? {is_anti_hermitian(not_anti_hermitian_matrix1)}")

    not_anti_hermitian_matrix2 = [
        [0, 1 - 2j],
        [1 + 2j, 0]
    ]
    print(f"Is not_anti_hermitian_matrix2 anti-Hermitian? {is_anti_hermitian(not_anti_hermitian_matrix2)}")