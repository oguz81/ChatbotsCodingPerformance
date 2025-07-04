import math
import cmath
from copy import deepcopy

def matrix_mult(a, b):
    """Matrix multiplication for n×n matrices"""
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def matrix_transpose(m):
    """Matrix transpose"""
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

def vector_norm(v):
    """Euclidean norm of a vector"""
    return math.sqrt(sum(x**2 for x in v))

def normalize_vector(v):
    """Normalize a vector to unit length"""
    norm = vector_norm(v)
    return [x/norm for x in v] if norm != 0 else v

def qr_decomposition(matrix):
    """QR decomposition using Gram-Schmidt process"""
    n = len(matrix)
    Q = [[0.0]*n for _ in range(n)]
    R = [[0.0]*n for _ in range(n)]
    
    for j in range(n):
        v = [matrix[i][j] for i in range(n)]
        
        for i in range(j):
            R[i][j] = sum(Q[k][i] * matrix[k][j] for k in range(n))
            v = [v[k] - R[i][j] * Q[k][i] for k in range(n)]
        
        R[j][j] = vector_norm(v)
        if R[j][j] != 0:
            Q[j] = [v[k]/R[j][j] for k in range(n)]
    
    return Q, R

def qr_algorithm(matrix, max_iter=100, tol=1e-10):
    """QR algorithm for finding eigenvalues"""
    n = len(matrix)
    A = deepcopy(matrix)
    
    for _ in range(max_iter):
        Q, R = qr_decomposition(A)
        A = matrix_mult(R, Q)
        
        # Check for convergence (lower triangular approaches eigenvalues)
        converged = True
        for i in range(n):
            for j in range(i):
                if abs(A[i][j]) > tol:
                    converged = False
                    break
            if not converged:
                break
        if converged:
            break
    
    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues

def inverse_iteration(matrix, eigenvalue, max_iter=100, tol=1e-10):
    """Inverse iteration to find eigenvector for a given eigenvalue"""
    n = len(matrix)
    A = deepcopy(matrix)
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # Shift matrix by eigenvalue
    A_shifted = [[A[i][j] - eigenvalue * I[i][j] for j in range(n)] for i in range(n)]
    
    # Initial random vector
    v = [1.0]*n
    
    for _ in range(max_iter):
        try:
            # Solve A_shifted * w = v (simplified here - in practice use LU decomposition)
            w = [sum(A_shifted[i][j] * v[j] for j in range(n)) for i in range(n)]
            v_new = normalize_vector(w)
            
            # Check convergence
            diff = sum(abs(v_new[i] - v[i]) for i in range(n))
            if diff < tol:
                break
            v = v_new
        except:
            break
    
    return v

def eigenvalues_and_eigenvectors_7x7(matrix, max_iter=100, tol=1e-8):
    """Calculate eigenvalues and eigenvectors of a 7×7 matrix"""
    if len(matrix) != 7 or any(len(row) != 7 for row in matrix):
        raise ValueError("Matrix must be 7×7")
    
    # Compute eigenvalues using QR algorithm
    eigenvalues = qr_algorithm(matrix, max_iter, tol)
    
    # Compute eigenvectors using inverse iteration
    eigenvectors = []
    for λ in eigenvalues:
        v = inverse_iteration(matrix, λ, max_iter, tol)
        eigenvectors.append(v)
    
    return eigenvalues, eigenvectors

# Example usage:
if __name__ == "__main__":
    # Create a sample 7×7 symmetric matrix (easier for eigenvalue computation)
    test_matrix = [
        [4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0]
    ]
    
    print("7×7 Test Matrix:")
    for row in test_matrix:
        print([f"{x:.1f}" for x in row])
    
    eigenvalues, eigenvectors = eigenvalues_and_eigenvectors_7x7(test_matrix)
    
    print("\nEigenvalues:")
    for i, λ in enumerate(eigenvalues):
        print(f"λ{i+1} = {λ:.6f}")
    
    print("\nEigenvectors (normalized):")
    for i, v in enumerate(eigenvectors):
        print(f"v{i+1} =", [f"{x:.6f}" for x in v])