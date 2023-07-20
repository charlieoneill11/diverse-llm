import numpy as np
from huggingface_hub import snapshot_download
from datasets import disable_caching

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
    #generate_and_invert_matrix(size=10) 
    directory = "/g/data/y89/cn1951"
    snapshot_download(repo_id="universeTBD/falcon-7b-abstracts-tiny", local_dir=directory+"/falcon-7b-abstracts-tiny", cache_dir=directory)
