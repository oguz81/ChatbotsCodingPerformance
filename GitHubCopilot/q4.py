import numpy as np

def calculate_transpose(matrix):
    """
    Calculate the transpose of a matrix.

    Args:
        matrix (list or numpy.ndarray): Input matrix.

    Returns:
        numpy.ndarray: Transpose of the matrix.
    """
    return np.transpose(np.array(matrix))

def calculate_complex_conjugate(matrix):
    """
    Calculate the complex conjugate of a matrix.

    Args:
        matrix (list or numpy.ndarray): Input matrix.

    Returns:
        numpy.ndarray: Complex conjugate of the matrix.
    """
    return np.conj(np.array(matrix))

def calculate_hermitian_conjugate(matrix):
    """
    Calculate the Hermitian conjugate (conjugate transpose) of a matrix.

    Args:
        matrix (list or numpy.ndarray): Input matrix.

    Returns:
        numpy.ndarray: Hermitian conjugate of the matrix.
    """
    return np.conj(np.transpose(np.array(matrix)))

# Example usage:
if __name__ == "__main__":
    # Example matrix with complex numbers
    matrix = [
        [1+2j, 2-1j, 3+4j],
        [4+0j, -5+1j, 6-2j],
        [7-3j, 8+4j, -9+0j]
    ]

    print("Original Matrix:")
    print(np.array(matrix))

    print("\nTranspose of the Matrix:")
    print(calculate_transpose(matrix))

    print("\nComplex Conjugate of the Matrix:")
    print(calculate_complex_conjugate(matrix))

    print("\nHermitian Conjugate of the Matrix:")
    print(calculate_hermitian_conjugate(matrix))