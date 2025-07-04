import numpy as np

def calculate_inverse(matrix):
  """
  Calculates the inverse of a square matrix.

  Args:
    matrix: A list of lists representing the square matrix.

  Returns:
    A list of lists representing the inverse of the matrix.
    Returns None if the matrix is singular (non-invertible).
  """
  try:
    np_matrix = np.array(matrix, dtype=float)
    inverse_matrix = np.linalg.inv(np_matrix)
    return inverse_matrix.tolist()
  except np.linalg.LinAlgError:
    return None

if __name__ == "__main__":
  # Example invertible matrix
  invertible_matrix = [
      [1, 2],
      [3, 4]
  ]
  inverse = calculate_inverse(invertible_matrix)
  if inverse:
    print("Original Invertible Matrix:")
    for row in invertible_matrix:
      print(row)
    print("\nInverse Matrix:")
    for row in inverse:
      print(row)

    # Verify the inverse (A * A_inverse should be close to the identity matrix)
    original_np = np.array(invertible_matrix)
    inverse_np = np.array(inverse)
    identity = np.dot(original_np, inverse_np)
    print("\nOriginal Matrix multiplied by its Inverse:")
    print(identity)
  else:
    print("The matrix is singular and does not have an inverse.")

  # Example singular (non-invertible) matrix
  singular_matrix = [
      [1, 2],
      [2, 4]
  ]
  inverse_singular = calculate_inverse(singular_matrix)
  if inverse_singular is None:
    print("\nOriginal Singular Matrix:")
    for row in singular_matrix:
      print(row)
    print("\nThis matrix is singular and does not have an inverse.")
  else:
    print("\nInverse of the singular matrix (this should not happen):")
    for row in inverse_singular:
      print(row)