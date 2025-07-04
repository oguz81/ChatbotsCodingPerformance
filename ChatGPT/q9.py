import sympy as sp

def eigen_3x3_manual(A):
    """
    Calculates eigenvalues and eigenvectors of a 3x3 matrix A without using numpy.linalg.
    Uses sympy to compute characteristic polynomial and solve eigenvectors.
    """
    if A.shape != (3, 3):
        raise ValueError("Matrix must be 3x3.")

    # Define symbolic variable
    lam = sp.symbols('λ')

    # Characteristic equation: det(A - λI) = 0
    I = sp.eye(3)
    char_eq = (A - lam * I).det()

    # Solve characteristic polynomial
    eigenvalues = sp.solve(char_eq, lam)

    # Compute eigenvectors
    eigenvectors = []
    for eig in eigenvalues:
        null_space = (A - eig * I).nullspace()
        if null_space:
            eigenvectors.append(null_space[0].normalized())
        else:
            eigenvectors.append(sp.Matrix([0, 0, 0]))

    return eigenvalues, eigenvectors

# Example usage
if __name__ == "__main__":
    A = sp.Matrix([[2, -1, 0],
                   [-1, 2, -1],
                   [0, -1, 2]])

    print("Matrix A:\n", A)

    eigenvalues, eigenvectors = eigen_3x3_manual(A)

    print("\nEigenvalues of A:\n", eigenvalues)
    print("\nEigenvectors of A:\n")
    for vec in eigenvectors:
        sp.pprint(vec)
