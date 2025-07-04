import numpy as np

def multiply_7x7(A, B):
    if A.shape != (7, 7) or B.shape != (7, 7):
        raise ValueError("Both matrices must be 7x7.")
    
    result = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            for k in range(7):
                result[i][j] += A[i][k] * B[k][j]
    return result

def multiply_15x15(A, B):
    if A.shape != (15, 15) or B.shape != (15, 15):
        raise ValueError("Both matrices must be 15x15.")
    
    result = np.zeros((15, 15))
    for i in range(15):
        for j in range(15):
            for k in range(15):
                result[i][j] += A[i][k] * B[k][j]
    return result

def multiply_20x20(A, B):
    if A.shape != (20, 20) or B.shape != (20, 20):
        raise ValueError("Both matrices must be 20x20.")
    
    result = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            for k in range(20):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Example usage
if __name__ == "__main__":
    A7 = np.random.randint(1, 10, size=(7, 7))
    B7 = np.random.randint(1, 10, size=(7, 7))
    print("Multiplication result of 7x7:\n", multiply_7x7(A7, B7))

    A15 = np.random.randint(1, 10, size=(15, 15))
    B15 = np.random.randint(1, 10, size=(15, 15))
    print("\nMultiplication result of 15x15:\n", multiply_15x15(A15, B15))

    A20 = np.random.randint(1, 10, size=(20, 20))
    B20 = np.random.randint(1, 10, size=(20, 20))
    print("\nMultiplication result of 20x20:\n", multiply_20x20(A20, B20))
