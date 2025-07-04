import numpy as np

def multiply_matrices_6x4_4x8(matrix_a, matrix_b):
  """
  Multiplies a 6x4 matrix by a 4x8 matrix.

  Args:
    matrix_a: A list of lists representing the 6x4 matrix.
    matrix_b: A list of lists representing the 4x8 matrix.

  Returns:
    A list of lists representing the 6x8 product matrix.
  """
  if (len(matrix_a) != 6 or any(len(row) != 4 for row in matrix_a) or
      len(matrix_b) != 4 or any(len(row) != 8 for row in matrix_b)):
    raise ValueError("Input matrices must be of shapes 6x4 and 4x8.")
  return (np.array(matrix_a) @ np.array(matrix_b)).tolist()

def multiply_matrices_12x4_4x16(matrix_a, matrix_b):
  """
  Multiplies a 12x4 matrix by a 4x16 matrix.

  Args:
    matrix_a: A list of lists representing the 12x4 matrix.
    matrix_b: A list of lists representing the 4x16 matrix.

  Returns:
    A list of lists representing the 12x16 product matrix.
  """
  if (len(matrix_a) != 12 or any(len(row) != 4 for row in matrix_a) or
      len(matrix_b) != 4 or any(len(row) != 16 for row in matrix_b)):
    raise ValueError("Input matrices must be of shapes 12x4 and 4x16.")
  return (np.array(matrix_a) @ np.array(matrix_b)).tolist()

def multiply_matrices_5x20_20x4(matrix_a, matrix_b):
  """
  Multiplies a 5x20 matrix by a 20x4 matrix.

  Args:
    matrix_a: A list of lists representing the 5x20 matrix.
    matrix_b: A list of lists representing the 20x4 matrix.

  Returns:
    A list of lists representing the 5x4 product matrix.
  """
  if (len(matrix_a) != 5 or any(len(row) != 20 for row in matrix_a) or
      len(matrix_b) != 20 or any(len(row) != 4 for row in matrix_b)):
    raise ValueError("Input matrices must be of shapes 5x20 and 20x4.")
  return (np.array(matrix_a) @ np.array(matrix_b)).tolist()

if __name__ == "__main__":
  # Example 6x4 and 4x8 matrices
  matrix_6x4 = [[i + j for j in range(4)] for i in range(6)]
  matrix_4x8 = [[i * j for j in range(8)] for i in range(4)]
  product_6x8 = multiply_matrices_6x4_4x8(matrix_6x4, matrix_4x8)
  print("Product of 6x4 and 4x8 matrices (first 3 rows):")
  for row in product_6x8[:3]:
    print(row)
  print("...")

  # Example 12x4 and 4x16 matrices
  matrix_12x4 = [[i + j for j in range(4)] for i in range(12)]
  matrix_4x16 = [[i * j for j in range(16)] for i in range(4)]
  product_12x16 = multiply_matrices_12x4_4x16(matrix_12x4, matrix_4x16)
  print("\nProduct of 12x4 and 4x16 matrices (first 3 rows):")
  for row in product_12x16[:3]:
    print(row)
  print("...")

  # Example 5x20 and 20x4 matrices
  matrix_5x20 = [[i + j for j in range(20)] for i in range(5)]
  matrix_20x4 = [[i * j for j in range(4)] for i in range(20)]
  product_5x4 = multiply_matrices_5x20_20x4(matrix_5x20, matrix_20x4)
  print("\nProduct of 5x20 and 20x4 matrices:")
  for row in product_5x4:
    print(row)