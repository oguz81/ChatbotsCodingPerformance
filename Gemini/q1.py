import numpy as np

def calculate_determinant_7x7(matrix):
  """
  Calculates the determinant of a 7x7 matrix.

  Args:
    matrix: A list of lists representing a 7x7 matrix.

  Returns:
    The determinant of the matrix.
  """
  if len(matrix) != 7 or any(len(row) != 7 for row in matrix):
    raise ValueError("Input matrix must be 7x7.")
  return np.linalg.det(np.array(matrix))

def calculate_determinant_11x11(matrix):
  """
  Calculates the determinant of an 11x11 matrix.

  Args:
    matrix: A list of lists representing an 11x11 matrix.

  Returns:
    The determinant of the matrix.
  """
  if len(matrix) != 11 or any(len(row) != 11 for row in matrix):
    raise ValueError("Input matrix must be 11x11.")
  return np.linalg.det(np.array(matrix))

if __name__ == "__main__":
  # Example 7x7 matrix
  matrix_7x7 = [
      [1, 2, 3, 4, 5, 6, 7],
      [8, 9, 10, 11, 12, 13, 14],
      [15, 16, 17, 18, 19, 20, 21],
      [22, 23, 24, 25, 26, 27, 28],
      [29, 30, 31, 32, 33, 34, 35],
      [36, 37, 38, 39, 40, 41, 42],
      [43, 44, 45, 46, 47, 48, 49]
  ]
  det_7x7 = calculate_determinant_7x7(matrix_7x7)
  print(f"Determinant of the 7x7 matrix: {det_7x7}")

  # Example 11x11 matrix
  matrix_11x11 = [
      [i + j for j in range(11)] for i in range(11)
  ]
  det_11x11 = calculate_determinant_11x11(matrix_11x11)
  print(f"Determinant of the 11x11 matrix: {det_11x11}")