import sympy as sp

def eigen_7x7_manual(A):
    """
    Calculates eigenvalues and eigenvectors of a 7x7 matrix A without using numpy.linalg.
    Uses sympy to compute characteristic polynomial and eigenvectors.
    """
    if A.shape != (7, 7):
        raise ValueError("Matrix must be 7x7.")

    # Symbolic variable for eigenvalues
    lam = sp.symbols('λ')

    # Characteristic polynomial: det(A - λI) = 0
    I = sp.eye(7)
    char_poly = (A - lam * I).det()

    # Solve characteristic polynomial for eigenvalues
    eigenvalues = sp.solve(char_poly, lam)

    # Compute eigenvectors (nullspace of A - λI for each eigenvalue)
    eigenvectors = []
    for eig in eigenvalues:
        null_space = (A - eig * I).nullspace()
        if null_space:
            eigenvectors.append(null_space[0].normalized())
        else:
            eigenvectors.append(sp.Matrix([0]*7))

    return eigenvalues, eigenvectors

# Example usage:
if __name__ == "__main__":
    # Define a symbolic or numeric 7x7 matrix
    A = sp.Matrix([
        [2, -1, 0, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0, 0],
        [0, -1, 2, -1, 0, 0, 0],
        [0, 0, -1, 2, -1, 0, 0],
        [0, 0, 0, -1, 2, -1, 0],
        [0, 0, 0, 0, -1, 2, -1],
        [0, 0, 0, 0, 0, -1, 2]
    ])

    print("Matrix A:\n")
    sp.pprint(A)

    eigenvalues, eigenvectors = eigen_7x7_manual(A)

    print("\nEigenvalues of A:\n", eigenvalues)

    print("\nEigenvectors of A:\n")
    for vec in eigenvectors:
        sp.pprint(vec)
        print()
