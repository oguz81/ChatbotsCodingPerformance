import numpy as np

def multiply_6x4_4x8(A, B):
    if A.shape != (6, 4) or B.shape != (4, 8):
        raise ValueError("Matrices must be of shapes 6x4 and 4x8 respectively.")
    
    result = np.zeros((6, 8))
    for i in range(6):
        for j in range(8):
            for k in range(4):
                result[i][j] += A[i][k] * B[k][j]
    return result

def multiply_12x4_4x16(A, B):
    if A.shape != (12, 4) or B.shape != (4, 16):
        raise ValueError("Matrices must be of shapes 12x4 and 4x16 respectively.")
    
    result = np.zeros((12, 16))
    for i in range(12):
        for j in range(16):
            for k in range(4):
                result[i][j] += A[i][k] * B[k][j]
    return result

def multiply_5x20_20x4(A, B):
    if A.shape != (5, 20) or B.shape != (20, 4):
        raise ValueError("Matrices must be of shapes 5x20 and 20x4 respectively.")
    
    result = np.zeros((5, 4))
    for i in range(5):
        for j in range(4):
            for k in range(20):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Example usage:
if __name__ == "__main__":
    A1 = np.random.randint(1, 10, size=(6, 4))
    B1 = np.random.randint(1, 10, size=(4, 8))
    print("Result of 6x4 * 4x8:\n", multiply_6x4_4x8(A1, B1))

    A2 = np.random.randint(1, 10, size=(12, 4))
    B2 = np.random.randint(1, 10, size=(4, 16))
    print("\nResult of 12x4 * 4x16:\n", multiply_12x4_4x16(A2, B2))

    A3 = np.random.randint(1, 10, size=(5, 20))
    B3 = np.random.randint(1, 10, size=(20, 4))
    print("\nResult of 5x20 * 20x4:\n", multiply_5x20_20x4(A3, B3))
