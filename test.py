import numpy as np

import numpy as np

def generate_and_invert_matrix(size):
    # Generate a random matrix
    matrix = np.random.rand(size, size)
    
    try:
        # Calculate the inverse of the matrix
        inverse_matrix = np.linalg.inv(matrix)
        
        # Save the original matrix
        np.savetxt("_original.txt", matrix, fmt="%f", delimiter="\t")
        
        # Save the inverse matrix
        np.savetxt("_inverse.txt", inverse_matrix, fmt="%f", delimiter="\t")
        
        return True
    except np.linalg.LinAlgError:
        # Matrix is not invertible
        return False


if __name__ == "__main__":
    generate_and_invert_matrix(size=10)