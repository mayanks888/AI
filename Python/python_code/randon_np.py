import numpy as np

fisrt_array=(np.random.randint(0,2,(3,2)))
second_array=(np.random.randint(0,2,(5,2)))

print(fisrt_array)
matmul=np.matmul(fisrt_array,np.transpose(second_array))
print(matmul)

third_np=(np.random.randint(0,2,(6,6)))
