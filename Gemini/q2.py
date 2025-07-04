import numpy as np

def multiply_matrices_7x7(matrix1, matrix2):
  """
  Multiplies two 7x7 square matrices.

  Args:
    matrix1: A list of lists representing the first 7x7 matrix.
    matrix2: A list of lists representing the second 7x7 matrix.

  Returns:
    A list of lists representing the product of the two matrices.
  """
  if (len(matrix1) != 7 or any(len(row) != 7 for row in matrix1) or
      len(matrix2) != 7 or any(len(row) != 7 for row in matrix2)):
    raise ValueError("Input matrices must be 7x7.")
  return (np.array(matrix1) @ np.array(matrix2)).tolist()

def multiply_matrices_15x15(matrix1, matrix2):
  """
  Multiplies two 15x15 square matrices.

  Args:
    matrix1: A list of lists representing the first 15x15 matrix.
    matrix2: A list of lists representing the second 15x15 matrix.

  Returns:
    A list of lists representing the product of the two matrices.
  """
  if (len(matrix1) != 15 or any(len(row) != 15 for row in matrix1) or
      len(matrix2) != 15 or any(len(row) != 15 for row in matrix2)):
    raise ValueError("Input matrices must be 15x15.")
  return (np.array(matrix1) @ np.array(matrix2)).tolist()

def multiply_matrices_20x20(matrix1, matrix2):
  """
  Multiplies two 20x20 square matrices.

  Args:
    matrix1: A list of lists representing the first 20x20 matrix.
    matrix2: A list of lists representing the second 20x20 matrix.

  Returns:
    A list of lists representing the product of the two matrices.
  """
  if (len(matrix1) != 20 or any(len(row) != 20 for row in matrix1) or
      len(matrix2) != 20 or any(len(row) != 20 for row in matrix2)):
    raise ValueError("Input matrices must be 20x20.")
  return (np.array(matrix1) @ np.array(matrix2)).tolist()

if __name__ == "__main__":
  # Example 7x7 matrices
  matrix_a_7x7 = [[i + j for j in range(7)] for i in range(7)]
  matrix_b_7x7 = [[i * j for j in range(7)] for i in range(7)]
  product_7x7 = multiply_matrices_7x7(matrix_a_7x7, matrix_b_7x7)
  print("Product of 7x7 matrices:")
  for row in product_7x7[:3]:  # Print first 3 rows for brevity
    print(row)
  print("...")

  # Example 15x15 matrices
  matrix_a_15x15 = [[i + j for j in range(15)] for i in range(15)]
  matrix_b_15x15 = [[i * j for j in range(15)] for i in range(15)]
  product_15x15 = multiply_matrices_15x15(matrix_a_15x15, matrix_b_15x15)
  print("\nProduct of 15x15 matrices (first 3 rows):")
  for row in product_15x15[:3]:
    print(row)
  print("...")

  # Example 20x20 matrices
  matrix_a_20x20 = [[i + j for j in range(20)] for i in range(20)]
  matrix_b_20x20 = [[i * j for j in range(20)] for i in range(20)]
  product_20x20 = multiply_matrices_20x20(matrix_a_20x20, matrix_b_20x20)
  print("\nProduct of 20x20 matrices (first 3 rows):")
  for row in product_20x20[:3]:
    print(row)
  print("...")