import numpy as np

def calculate_inverse(matrix):
    """
    Calculate the inverse of a square matrix.

    Args:
        matrix (list or numpy.ndarray): Input square matrix.

    Returns:
        numpy.ndarray: Inverse of the matrix.
    
    Raises:
        ValueError: If the matrix is not square or singular (non-invertible).
    """
    # Convert the matrix to a numpy array
    matrix = np.array(matrix)

    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square to calculate its inverse.")

    # Check if the matrix is invertible (non-singular)
    determinant = np.linalg.det(matrix)
    if determinant == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")

    # Calculate the inverse using numpy
    return np.linalg.inv(matrix)

# Example usage
if __name__ == "__main__":
    # Example square matrix
    matrix = [
        [4, 7],
        [2, 6]
    ]
    
    try:
        print("Original Matrix:")
        print(np.array(matrix))
        
        print("\nInverse of the Matrix:")
        print(calculate_inverse(matrix))
    except ValueError as e:
        print(e)