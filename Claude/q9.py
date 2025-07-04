import numpy as np
import math

def calculate_eigenvalues_eigenvectors_3x3(matrix):
    """
    Calculate eigenvalues and eigenvectors of a 3x3 matrix without using np.linalg.
    
    Args:
        matrix: A 3x3 numpy array or list of lists
        
    Returns:
        tuple: (eigenvalues, eigenvectors)
            - eigenvalues: numpy array of 3 eigenvalues
            - eigenvectors: numpy array where each column is an eigenvector
    """
    # Convert to numpy array if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=float)
    
    # Check dimensions
    if matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix")
    
    # Step 1: Find the characteristic polynomial coefficients
    # For a 3x3 matrix, the characteristic polynomial is:
    # λ³ - (trace)λ² + (sum of principal minors)λ - det(A)
    
    # Coefficient of λ³ is always 1
    # Coefficient of λ² is -trace(A)
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    
    # Coefficient of λ¹ is sum of principal minors
    minor_sum = (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) + \
                (matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0]) + \
                (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    
    # Coefficient of λ⁰ is -det(A)
    det = matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) - \
          matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0]) + \
          matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    
    # Characteristic polynomial coefficients: [1, -trace, minor_sum, -det]
    poly_coeffs = [1, -trace, minor_sum, -det]
    
    # Step 2: Find the eigenvalues by solving the characteristic polynomial
    eigenvalues = find_roots_of_cubic(poly_coeffs)
    
    # Step 3: Find the eigenvectors for each eigenvalue
    eigenvectors = np.zeros((3, 3))
    
    for i, eigenvalue in enumerate(eigenvalues):
        eigenvector = find_eigenvector(matrix, eigenvalue)
        eigenvectors[:, i] = eigenvector
    
    return eigenvalues, eigenvectors


def find_roots_of_cubic(coeffs):
    """
    Find the roots of a cubic polynomial with coefficients [a, b, c, d]
    where the polynomial is a*x^3 + b*x^2 + c*x + d
    
    Args:
        coeffs: List of 4 coefficients [a, b, c, d]
        
    Returns:
        numpy.ndarray: Array of 3 roots (eigenvalues)
    """
    a, b, c, d = coeffs
    
    if abs(a) < 1e-10:  # If the leading coefficient is effectively zero
        raise ValueError("Not a cubic polynomial")
    
    # Normalize the coefficients
    b, c, d = b/a, c/a, d/a
    
    # Convert to depressed cubic t^3 + pt + q = 0 via x = t - b/3a
    p = c - b*b/3
    q = d + 2*b*b*b/27 - b*c/3
    
    # Discriminant
    discriminant = (q*q/4) + (p*p*p/27)
    
    roots = np.zeros(3, dtype=complex)
    
    if abs(discriminant) < 1e-10:  # Discriminant is approximately zero
        if abs(p) < 1e-10:  # Triple root
            roots[0] = roots[1] = roots[2] = -b/3
        else:  # One single root and one double root
            u = np.cbrt(-q/2)
            roots[0] = 2*u - b/3
            roots[1] = roots[2] = -u - b/3
    elif discriminant > 0:  # One real root and two complex conjugate roots
        u = np.cbrt(-q/2 + math.sqrt(discriminant))
        v = np.cbrt(-q/2 - math.sqrt(discriminant))
        
        roots[0] = u + v - b/3
        roots[1] = -0.5*(u + v) - b/3 + 0.5*1j*math.sqrt(3)*(u - v)
        roots[2] = -0.5*(u + v) - b/3 - 0.5*1j*math.sqrt(3)*(u - v)
    else:  # Three distinct real roots
        theta = math.acos(-q/2 / math.sqrt(abs(p*p*p)/27))
        
        for k in range(3):
            roots[k] = 2 * math.sqrt(abs(p)/3) * math.cos((theta + 2*k*math.pi)/3) - b/3
    
    # Take the real part if the imaginary part is negligible
    for i in range(3):
        if abs(roots[i].imag) < 1e-10:
            roots[i] = roots[i].real
    
    return roots


def find_eigenvector(matrix, eigenvalue):
    """
    Find an eigenvector for a given eigenvalue using Gaussian elimination.
    
    Args:
        matrix: The 3x3 matrix
        eigenvalue: The eigenvalue to find the eigenvector for
        
    Returns:
        numpy.ndarray: Normalized eigenvector
    """
    # Create the matrix (A - λI)
    A_lambda = matrix.copy()
    for i in range(3):
        A_lambda[i, i] -= eigenvalue
    
    # Apply Gaussian elimination to find the null space
    # Create an augmented matrix [A-λI | 0]
    augmented = np.column_stack((A_lambda, np.zeros(3)))
    
    # Forward elimination
    for i in range(2):  # We only need two iterations for a 3x3 matrix
        # If the pivot is zero, swap rows
        if abs(augmented[i, i]) < 1e-10:
            # Find a row below with non-zero entry in the current column
            for j in range(i+1, 3):
                if abs(augmented[j, i]) > 1e-10:
                    augmented[[i, j]] = augmented[[j, i]]  # Swap rows
                    break
            else:  # If no such row is found, continue to the next column
                continue
        
        # Eliminate entries below the pivot
        for j in range(i+1, 3):
            factor = augmented[j, i] / augmented[i, i]
            augmented[j] -= factor * augmented[i]
    
    # Back-substitution to find the eigenvector
    eigenvector = np.ones(3)  # Start with [1, 1, 1]
    
    # If the matrix is close to singular, we can set one variable to 1
    # and solve for the others using back-substitution
    
    # Set x3 = 1 and solve for x2 and x1
    if abs(augmented[1, 1]) > 1e-10:
        eigenvector[1] = -augmented[1, 2] / augmented[1, 1]  # x2
        
        if abs(augmented[0, 0]) > 1e-10:
            eigenvector[0] = -(augmented[0, 1] * eigenvector[1] + augmented[0, 2]) / augmented[0, 0]  # x1
        else:
            # If a00 is almost zero, try using another equation
            if abs(augmented[0, 1]) > 1e-10:
                eigenvector[0] = -(augmented[0, 2] + augmented[0, 1] * eigenvector[1]) / augmented[0, 0]
    
    # Normalize the eigenvector
    return eigenvector / np.linalg.norm(eigenvector)


# Example usage:
if __name__ == "__main__":
    # Create a 3x3 matrix
    matrix = np.array([
        [4, -1, 1],
        [-1, 3, -2],
        [1, -2, 3]
    ])
    
    print("Original Matrix:")
    print(matrix)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = calculate_eigenvalues_eigenvectors_3x3(matrix)
    
    print("\nEigenvalues:")
    for i, eigenvalue in enumerate(eigenvalues):
        print(f"λ{i+1} = {eigenvalue:.6f}")
    
    print("\nEigenvectors (as columns):")
    print(eigenvectors)
    
    # Verify: A * v = λ * v for each eigenvector
    print("\nVerification:")
    for i in range(3):
        v = eigenvectors[:, i]
        Av = matrix @ v
        lambda_v = eigenvalues[i] * v
        print(f"A * v{i+1} = {Av}")
        print(f"λ{i+1} * v{i+1} = {lambda_v}")
        print(f"Match: {np.allclose(Av, lambda_v, rtol=1e-5, atol=1e-5)}")
        print()