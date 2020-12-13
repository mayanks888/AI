import numpy as np

test = np.random.randn(3, 40, 40)
result = np.repeat(test[np.newaxis, ...], 10, axis=0)
print(result.shape)
