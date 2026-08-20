import numpy as np

vals = [-2.7e-4, -1.9e-6, -0.0, 0.0, 1.9e-6, 2.7e-4, 1.0, 100.0]
np.array(vals, dtype="<f4").tofile("in-x.bin")
