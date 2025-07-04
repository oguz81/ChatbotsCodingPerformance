import numpy as np

def calculate_transpose(matrix):
  """
  Calculates the transpose of a matrix.

  Args:
    matrix: A list of lists representing the matrix.

  Returns:
    A list of lists representing the transpose of the matrix.
  """
  np_matrix = np.array(matrix)
  return np_matrix.T.tolist()

def calculate_complex_conjugate(matrix):
  """
  Calculates the complex conjugate of a matrix.

  Args:
    matrix: A list of lists representing the matrix (can contain complex numbers).

  Returns:
    A list of lists representing the complex conjugate of the matrix.
  """
  np_matrix = np.array(matrix, dtype=np.complex_)
  return np.conjugate(np_matrix).tolist()

def calculate_hermitian_conjugate(matrix):
  """
  Calculates the Hermitian conjugate (conjugate transpose) of a matrix.

  Args:
    matrix: A list of lists representing the matrix (can contain complex numbers).

  Returns:
    A list of lists representing the Hermitian conjugate of the matrix.
  """
  np_matrix = np.array(matrix, dtype=np.complex_)
  return np_matrix.conj().T.tolist()

if __name__ == "__main__":
  # Example matrix (with complex numbers)
  matrix = [
      [1 + 2j, 3 - 1j, 4],
      [5, 6 + 0j, 7 - 3j]
  ]

  # Calculate transpose
  transpose_matrix = calculate_transpose(matrix)
  print("Original Matrix:")
  for row in matrix:
    print(row)
  print("\nTranspose:")
  for row in transpose_matrix:
    print(row)

  # Calculate complex conjugate
  conjugate_matrix = calculate_complex_conjugate(matrix)
  print("\nComplex Conjugate:")
  for row in conjugate_matrix:
    print(row)

  # Calculate Hermitian conjugate
  hermitian_conjugate_matrix = calculate_hermitian_conjugate(matrix)
  print("\nHermitian Conjugate:")
  for row in hermitian_conjugate_matrix:
    print(row)

  # Example real-valued matrix
  real_matrix = [
      [1, 2, 3],
      [4, 5, 6]
  ]

  real_transpose = calculate_transpose(real_matrix)
  print("\nOriginal Real Matrix:")
  for row in real_matrix:
    print(row)
  print("\nTranspose of Real Matrix:")
  for row in real_transpose:
    print(row)

  real_conjugate = calculate_complex_conjugate(real_matrix)
  print("\nComplex Conjugate of Real Matrix:")
  for row in real_conjugate:
    print(row)

  real_hermitian_conjugate = calculate_hermitian_conjugate(real_matrix)
  print("\nHermitian Conjugate of Real Matrix:")
  for row in real_hermitian_conjugate:
    print(row)