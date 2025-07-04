import numpy as np
import math

def calculate_eigenvalues_eigenvectors_7x7(matrix, max_iterations=1000, tolerance=1e-10):
    """
    Calculate eigenvalues and eigenvectors of a 7x7 matrix using the power method
    and deflation, without using np.linalg.
    
    Args:
        matrix: A 7x7 numpy array or list of lists
        max_iterations: Maximum number of iterations for power method
        tolerance: Convergence tolerance
        
    Returns:
        tuple: (eigenvalues, eigenvectors)
            - eigenvalues: numpy array of 7 eigenvalues
            - eigenvectors: numpy array where each column is an eigenvector
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=float)
    
    # Check dimensions
    if matrix.shape != (7, 7):
        raise ValueError("Input must be a 7x7 matrix")
    
    n = 7  # Matrix dimension
    eigenvalues = np.zeros(n)
    eigenvectors = np.zeros((n, n))
    
    # Create a copy of the original matrix for deflation
    A = matrix.copy()
    
    # Find all eigenvalues and eigenvectors using power method with deflation
    for i in range(n):
        eigenvalue, eigenvector = power_method(A, max_iterations, tolerance)
        
        # Store results
        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = eigenvector
        
        # Deflate the matrix for the next iteration (if not the last one)
        if i < n - 1:
            # Matrix deflation: A' = A - λ·v·vᵀ
            deflation_matrix = eigenvalue * np.outer(eigenvector, eigenvector)
            A = A - deflation_matrix
    
    # Verify against the original matrix
    for i in range(n):
        # Check if A·v ≈ λ·v for the original matrix
        Av = matrix @ eigenvectors[:, i]
        lambda_v = eigenvalues[i] * eigenvectors[:, i]
        
        # If verification fails, try to improve the eigenvector using inverse iteration
        if not np.allclose(Av, lambda_v, rtol=1e-3, atol=1e-3):
            improved_eigenvector = inverse_iteration(matrix, eigenvalues[i], max_iterations, tolerance)
            eigenvectors[:, i] = improved_eigenvector
    
    return eigenvalues, eigenvectors


def power_method(matrix, max_iterations=1000, tolerance=1e-10):
    """
    Find the dominant eigenvalue and corresponding eigenvector using the power method.
    
    Args:
        matrix: Square matrix
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        tuple: (eigenvalue, eigenvector)
    """
    n = matrix.shape[0]
    
    # Start with a random vector
    v = np.random.rand(n)
    v = v / np.sqrt(np.sum(v**2))  # Normalize
    
    lambda_prev = 0
    
    # Power iteration
    for _ in range(max_iterations):
        # Multiply the matrix by the vector
        Av = matrix @ v
        
        # Calculate Rayleigh quotient for eigenvalue
        lambda_curr = np.dot(v, Av) / np.dot(v, v)
        
        # Normalize the new vector
        v_norm = np.sqrt(np.sum(Av**2))
        if v_norm < tolerance:  # Handle the case of a zero vector
            break
            
        v = Av / v_norm
        
        # Check for convergence
        if abs(lambda_curr - lambda_prev) < tolerance:
            break
            
        lambda_prev = lambda_curr
    
    return lambda_curr, v


def inverse_iteration(matrix, eigenvalue_approx, max_iterations=100, tolerance=1e-10):
    """
    Find an eigenvector for a specific eigenvalue using inverse iteration.
    
    Args:
        matrix: Square matrix
        eigenvalue_approx: Approximate eigenvalue
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        numpy.ndarray: Eigenvector
    """
    n = matrix.shape[0]
    
    # Create the shifted matrix (A - λI)
    shifted_matrix = matrix.copy()
    for i in range(n):
        shifted_matrix[i, i] -= eigenvalue_approx
    
    # Add a small perturbation to avoid singularity
    for i in range(n):
        shifted_matrix[i, i] += 1e-8
    
    # Start with a random vector
    v = np.random.rand(n)
    v = v / np.sqrt(np.sum(v**2))  # Normalize
    
    # Inverse iteration
    for _ in range(max_iterations):
        # Solve (A - λI)w = v
        w = solve_linear_system(shifted_matrix, v)
        
        # Normalize
        w_norm = np.sqrt(np.sum(w**2))
        if w_norm < tolerance:  # Handle the case of a zero vector
            break
            
        w = w / w_norm
        
        # Check for convergence
        if np.sqrt(np.sum((w - v)**2)) < tolerance:
            break
            
        v = w
    
    return v


def solve_linear_system(A, b):
    """
    Solve the linear system Ax = b using Gaussian elimination with partial pivoting.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        
    Returns:
        numpy.ndarray: Solution vector
    """
    n = len(b)
    
    # Create augmented matrix [A|b]
    augmented = np.column_stack((A.copy(), b.copy()))
    
    # Forward elimination with partial pivoting
    for i in range(n-1):
        # Find pivot
        max_row = i
        max_val = abs(augmented[i, i])
        
        for j in range(i+1, n):
            if abs(augmented[j, i]) > max_val:
                max_row = j
                max_val = abs(augmented[j, i])
        
        # Swap rows if necessary
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
        
        # Eliminate below the pivot
        for j in range(i+1, n):
            if abs(augmented[i, i]) < 1e-14:  # Near-zero pivot
                continue
                
            factor = augmented[j, i] / augmented[i, i]
            augmented[j, i:n+1] -= factor * augmented[i, i:n+1]
    
    # Back substitution
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        sum_ax = 0
        for j in range(i+1, n):
            sum_ax += augmented[i, j] * x[j]
            
        if abs(augmented[i, i]) < 1e-14:  # Near-zero pivot
            x[i] = 0  # Set as free variable
        else:
            x[i] = (augmented[i, n] - sum_ax) / augmented[i, i]
    
    return x


def gram_schmidt_orthogonalization(vectors):
    """
    Apply Gram-Schmidt orthogonalization to a set of vectors.
    
    Args:
        vectors: Matrix where each column is a vector
        
    Returns:
        numpy.ndarray: Matrix with orthogonal columns
    """
    n = vectors.shape[1]  # Number of vectors
    orthogonal = vectors.copy()
    
    for i in range(n):
        # Normalize the current vector
        norm = np.sqrt(np.sum(orthogonal[:, i]**2))
        if norm > 1e-10:
            orthogonal[:, i] = orthogonal[:, i] / norm
        
        # Make all subsequent vectors orthogonal to this one
        for j in range(i+1, n):
            projection = np.dot(orthogonal[:, i], orthogonal[:, j])
            orthogonal[:, j] = orthogonal[:, j] - projection * orthogonal[:, i]
    
    # Normalize all columns
    for i in range(n):
        norm = np.sqrt(np.sum(orthogonal[:, i]**2))
        if norm > 1e-10:
            orthogonal[:, i] = orthogonal[:, i] / norm
    
    return orthogonal


# Example usage:
if __name__ == "__main__":
    # Create a symmetric 7x7 matrix for testing
    # Symmetric matrices have real eigenvalues and orthogonal eigenvectors
    np.random.seed(42)  # For reproducibility
    
    # Create a random matrix
    A = np.random.rand(7, 7)
    
    # Make it symmetric by adding its transpose and dividing by 2
    matrix = (A + A.T) / 2
    
    print("Original Matrix:")
    print(matrix)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = calculate_eigenvalues_eigenvectors_7x7(matrix)
    
    # Sort eigenvalues and corresponding eigenvectors by absolute value
    idx = np.argsort(np.abs(eigenvalues))[::-1]  # Descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("\nEigenvalues:")
    for i, eigenvalue in enumerate(eigenvalues):
        print(f"λ{i+1} = {eigenvalue:.6f}")
    
    print("\nEigenvectors (as columns):")
    print(eigenvectors)
    
    # Verify: A * v = λ * v for each eigenvector
    print("\nVerification:")
    for i in range(7):
        v = eigenvectors[:, i]
        Av = matrix @ v
        lambda_v = eigenvalues[i] * v
        error = np.linalg.norm(Av - lambda_v)
        print(f"Eigenvalue {i+1}: Error = {error:.6e}")
    
    # Check orthogonality of eigenvectors
    print("\nOrthogonality check:")
    orthogonal_vectors = gram_schmidt_orthogonalization(eigenvectors)
    for i in range(7):
        for j in range(i+1, 7):
            dot_product = np.dot(orthogonal_vectors[:, i], orthogonal_vectors[:, j])
            print(f"v{i+1}·v{j+1} = {dot_product:.6e}")