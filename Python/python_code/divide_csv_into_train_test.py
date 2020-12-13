import numpy as np
import pandas as pd

df = pd.read_csv('/home/mayank_sati/codebase/git/AI/Machine_learning/Python/Open_CV/myfarm_crop.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.95

train = df[msk]
test = df[~msk]

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
df = pd.DataFrame(train, columns=columns)
df.to_csv('myfarm_crop_train.csv', index=False)

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
df = pd.DataFrame(test, columns=columns)
df.to_csv('myfarm_crop_test.csv', index=False)
