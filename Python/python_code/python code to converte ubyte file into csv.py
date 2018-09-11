from mlxtend.data import loadlocal_mnist
import numpy as np
X, y = loadlocal_mnist(
        images_path='/home/mayank/PycharmProjects/Datasets/MNIST_data/train-images-idx3-ubyte',
        labels_path='/home/mayank/PycharmProjects/Datasets/MNIST_data/train-labels-idx1-ubyte')




print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])

np.savetxt(fname='/home/mayank/PycharmProjects/Datasets/MNIST_data/train_images.csv',
           X=X, delimiter=',', fmt='%d')
np.savetxt(fname='/home/mayank/PycharmProjects/Datasets/MNIST_data/train_labels.csv',
           X=y, delimiter=',', fmt='%d')
# /home/mayank/PycharmProjects/Datasets/MNIST_data/train-images-idx3-ubyte.gz