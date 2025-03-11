import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/'))
import numpy as np

import XM

from utils.io import save_matrix_to_bin, load_matrix_from_bin

# The XM solver takes in the Q matrix for the SDP problem 
# (more details refer to our paper: https://arxiv.org/abs/2502.04640)
# and it output the R matrix and the s vector. note these may in rank higher than 3
# so you will need more information for previous matrix process to recover the XM solution

# full XM
XM.solve("./assets/1-Solver/",3,1e-3,10.0,1000)

# XM with rank-3
XM.solve_rank3("./assets/1-Solver/",3,1e-3,10.0,1000)

# full XM
XM.solve("./assets/2-Solver/",5,1e-3,10000.0,1000)

# XM with rank-3
XM.solve_rank3("./assets/2-Solver/",3,1e-3,10000.0,1000)




