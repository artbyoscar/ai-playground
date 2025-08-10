import numpy as np
K, N = 2048, 256
rng = np.random.default_rng(42)
B = rng.uniform(-1, 1, (K, N)).astype('f4')
np.save('B.npy', B)
