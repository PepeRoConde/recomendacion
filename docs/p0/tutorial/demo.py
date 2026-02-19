import zipfile
import json
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

# Path to the ZIP file
zip_path = "data.zip"

# Lists to store rows, columns, and values of the sparse matrix
rows, cols, values = [], [], []

# Open the ZIP file
with zipfile.ZipFile(zip_path, "r") as zipf:
    # Iterate over each file in the ZIP
    for file in zipf.namelist():
        # Check if the file is a JSON
        if file.endswith(".json"):
            # Read the content of the JSON file
            with zipf.open(file) as f:
                file_data = json.loads(f.read())
                # Extract row, column, and value data
                rows.append(file_data["row"])
                cols.append(file_data["col"])
                values.append(file_data["value"])

# Create the sparse CSR matrix
matrix = csr_matrix((values, (rows, cols)), shape=(3, 3))

# Save the matrix to a .npz file
save_npz("sparse_matrix.npz", matrix, compressed=False)
print("CSR matrix created and saved to 'sparse_matrix.npz'.")

# Load the matrix from the .npz file
matrix2 = load_npz("sparse_matrix.npz")
print("\nMatrix loaded from 'sparse_matrix.npz':")
print(matrix2.toarray())

# Define a vector to multiply with the matrix
vector = np.array([1, 2, 3])

# Multiply the matrix by the vector
result = matrix2.dot(vector)
print("\nResult of multiplying the matrix by the vector [1, 2, 3]:")
print(result)
