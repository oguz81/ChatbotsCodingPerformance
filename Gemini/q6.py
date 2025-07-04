def is_symmetric(matrix):
    """
    Checks if a square matrix is symmetric.

    A matrix A is symmetric if A[i][j] == A[j][i] for all i and j.

    Args:
        matrix: A list of lists representing the square matrix.

    Returns:
        True if the matrix is symmetric, False otherwise.
    """
    rows = len(matrix)
    if rows == 0:
        return True  # An empty matrix can be considered symmetric

    # Check if it's a square matrix
    for row in matrix:
        if len(row) != rows:
            return False

    for i in range(rows):
        for j in range(i + 1, rows):  # Only need to check the upper triangle
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

def is_antisymmetric(matrix):
    """
    Checks if a square matrix is antisymmetric (or skew-symmetric).

    A matrix A is antisymmetric if A[i][j] == -A[j][i] for all i and j,
    and the diagonal elements A[i][i] must be 0.

    Args:
        matrix: A list of lists representing the square matrix.

    Returns:
        True if the matrix is antisymmetric, False otherwise.
    """
    rows = len(matrix)
    if rows == 0:
        return True  # An empty matrix can be considered antisymmetric

    # Check if it's a square matrix
    for row in matrix:
        if len(row) != rows:
            return False

    for i in range(rows):
        for j in range(rows):
            if i == j:
                if matrix[i][j] != 0:
                    return False
            elif matrix[i][j] != -matrix[j][i]:
                return False
    return True

if __name__ == "__main__":
    # Symmetric matrices
    sym_matrix1 = [
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6]
    ]
    print(f"Is sym_matrix1 symmetric? {is_symmetric(sym_matrix1)}")

    sym_matrix2 = [
        [1, -2],
        [-2, 0]
    ]
    print(f"Is sym_matrix2 symmetric? {is_symmetric(sym_matrix2)}")

    sym_matrix3 = []
    print(f"Is sym_matrix3 symmetric? {is_symmetric(sym_matrix3)}")

    not_sym_matrix1 = [
        [1, 2, 3],
        [2, 4, 6],
        [3, 5, 6]
    ]
    print(f"Is not_sym_matrix1 symmetric? {is_symmetric(not_sym_matrix1)}")

    not_square_sym = [
        [1, 2],
        [2, 4],
        [3, 5]
    ]
    print(f"Is not_square_sym symmetric? {is_symmetric(not_square_sym)}")

    # Antisymmetric matrices
    anti_sym_matrix1 = [
        [0, 1, -2],
        [-1, 0, 3],
        [2, -3, 0]
    ]
    print(f"Is anti_sym_matrix1 antisymmetric? {is_antisymmetric(anti_sym_matrix1)}")

    anti_sym_matrix2 = [
        [0, -5],
        [5, 0]
    ]
    print(f"Is anti_sym_matrix2 antisymmetric? {is_antisymmetric(anti_sym_matrix2)}")

    anti_sym_matrix3 = []
    print(f"Is anti_sym_matrix3 antisymmetric? {is_antisymmetric(anti_sym_matrix3)}")

    not_anti_sym_matrix1 = [
        [1, 1, -2],
        [-1, 0, 3],
        [2, -3, 0]
    ]
    print(f"Is not_anti_sym_matrix1 antisymmetric? {is_antisymmetric(not_anti_sym_matrix1)}")

    not_anti_sym_matrix2 = [
        [0, 1, 2],
        [-1, 0, -3],
        [2, 3, 0]
    ]
    print(f"Is not_anti_sym_matrix2 antisymmetric? {is_antisymmetric(not_anti_sym_matrix2)}")

    not_square_anti_sym = [
        [0, 1],
        [-1, 0],
        [2, -3]
    ]
    print(f"Is not_square_anti_sym antisymmetric? {is_antisymmetric(not_square_anti_sym)}")