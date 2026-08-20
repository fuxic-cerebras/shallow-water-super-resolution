import numpy as np

# Condition values: negative, negative-tiny, -0.0, +0.0, positive-tiny, small, one, large.
np.array([-2.7e-4, -1.9e-6, -0.0, 0.0, 1.9e-6, 2.7e-4, 1.0, 100.0], "<f4").tofile("in-x.bin")
np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], "<f4").tofile("in-a.bin")
np.array([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], "<f4").tofile("in-b.bin")
