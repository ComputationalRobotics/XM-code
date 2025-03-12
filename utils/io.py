import numpy as np

# for saving output in txt
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()  

    def flush(self):
        for f in self.files:
            f.flush()

def save_matrix_to_bin(filename, matrix, byte = 4):
    try:
        with open(filename, "wb") as file:
            rows, cols = matrix.shape
            # default is 4-bit int
            if byte == 4:
                file.write(rows.to_bytes(4, byteorder='little'))
                file.write(cols.to_bytes(4, byteorder='little'))
            elif byte == 8:     
                file.write(rows.to_bytes(8, byteorder='little', signed=False))
                file.write(cols.to_bytes(8, byteorder='little', signed=False))
            
            matrix.T.tofile(file)  
            
            print(f"Matrix saved to {filename}. Rows: {rows}, Cols: {cols}")
    except IOError:
        print("Cannot write to file")

def load_matrix_from_bin(filename, byte = 4):
    try:
        with open(filename, "rb") as file:
            if byte == 4:
                rows = int.from_bytes(file.read(4), byteorder='little')
                cols = int.from_bytes(file.read(4), byteorder='little')
            elif byte == 8:
                rows = int.from_bytes(file.read(8), byteorder='little', signed=False)
                cols = int.from_bytes(file.read(8), byteorder='little', signed=False)
            
            n = rows 

            matrix = np.fromfile(file, dtype=np.double, count=rows * cols)
            matrix = matrix.reshape((rows, cols), order='F')
            
            print(f"rows: {rows}, cols: {cols}")
            return matrix, n
    except IOError:
        print("Cannot open file")
        return None, None
 
def save_array_to_bin(filename, array):
    """
    Save a 1D NumPy array to a binary file.
    
    Args:
        filename (str): The name of the file to save the array.
        array (np.ndarray): The 1D array to save.
    """
    try:
        with open(filename, "wb") as file:
            # Write the length of the array as a 4-byte integer
            length = array.size
            file.write(length.to_bytes(4, byteorder='little'))
            
            # Write the array data
            array.tofile(file)
            
            print(f"Array saved to {filename}. Length: {length}")
    except IOError:
        print("Error: Cannot write to file.")

def load_array_from_bin(filename):
    """
    Load a 1D NumPy array from a binary file.
    
    Args:
        filename (str): The name of the file to read the array from.
    
    Returns:
        np.ndarray: The loaded 1D array.
    """
    try:
        with open(filename, "rb") as file:
            # Read the length of the array
            length = int.from_bytes(file.read(4), byteorder='little')
            
            # Read the array data
            array = np.fromfile(file, dtype=np.double, count=length)
            
            print(f"Array loaded from {filename}. Length: {length}")
            return array
    except IOError:
        print("Error: Cannot read file.")
        return None

    