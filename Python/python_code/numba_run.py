import numpy as np
from numba import jit


def sort_by(ary):
    return ary


def main():
    init_pop = np.random.standard_normal((10, 2)).astype(np.float32)
    a = np.zeros((10, 2), dtype=np.float32)
    z = np.expand_dims(np.zeros((10,)), 1)
    v1 = np.concatenate((init_pop, a, z), axis=1)
    v3 = np.reshape(np.take(v1, range(4), axis=1), (10, 2, 2))
    v2 = np.random.randint(10, size=(2, 3), dtype=np.int32)
    v4 = np.random.randint(2, size=(3, 1, 2), dtype=np.int32).astype(np.float32)
    return sort_by(v1)[:10][0, 4]


print(main())
print(jit(forceobj=True)(main)())
