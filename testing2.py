import numpy as np
from scipy.signal import convolve

width, height, depth = 10, 10, 5

gridA = np.random.uniform(-1, 1, (width, height, depth))
kernalA = np.random.uniform(-1, 1, (3, 3))

gridA = np.transpose(gridA, (2, 0, 1))
print(gridA.shape, kernalA.shape)
gridA_new = np.transpose(convolve(gridA, kernalA[None,:, :], mode='same'), (1, 2, 0))

gridB = gridA.copy()

kernalB = np.array([kernalB for i in range(depth)])

gridB_new = np.transpose(convolve(gridB, kernalB, mode='same'), (1, 2, 0))

print(np.round(gridA_new-gridB_new, 4))
