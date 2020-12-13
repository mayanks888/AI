import pandas as pd
import torch
import torch.nn as nn
# from fastai.learner import *
from fastai.column_data import *

path = '/home/mayank-s/PycharmProjects/Datasets/ml-latest-small/'

ratings = pd.read_csv(path + 'ratings.csv')
ratings.head()

movies = pd.read_csv(path + 'movies.csv')
movies.head()

val_idxs = get_cv_idxs(len(ratings))
wd = 2e-4
n_factors = 50
cf = CollabFilterDataset.from_csv(path, 'ratings.csv', 'userId', 'movieId', 'rating')
learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn=optim.Adam)

learn.fit(1e-2, 2, wds=wd, cycle_len=1, cycle_mult=2)
u_uniq = ratings.userId.unique()
user2idx = {o: i for i, o in enumerate(u_uniq)}
ratings.userId = ratings.userId.apply(lambda x: user2idx[x])

m_uniq = ratings.movieId.unique()
movie2idx = {o: i for i, o in enumerate(m_uniq)}
ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])

n_users = int(ratings.userId.nunique())
n_movies = int(ratings.movieId.nunique())


class Dotprodcut(nn.Module):

    def forward(self, u, m):
        return ((u * m).sum())


model = Dotprodcut()
a = torch.tensor([[4.0], [5], [9]])
b = torch.tensor([[5.0], [7], [11]])
# a=torch.ones(size=(2,2))
# b=torch.ones(size=(2,2))
# b=3
print(model(a, b))


class EmbeddingDot(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.u.weight.data.uniform_(0, 0.05)
        self.m.weight.data.uniform_(0, 0.05)

    def forward(self, cats, conts):
        users, movies = cats[:, 0], cats[:, 1]
        u, m = self.u(users), self.m(movies)
        return (u * m).sum(1)


x = ratings.drop(['rating', 'timestamp'], axis=1)
y = ratings['rating']
data = ColumnarModelData.from_data_frame(path, val_idxs, x, y, ['userId', 'movieId'], 64)
wd = 1e-5
model = EmbeddingDot(n_users, n_movies)
opt = optim.SGD(model.parameters(), 1e-1, weight_decay=wd, momentum=0.9)
fit(model, data, 3, opt, F.mse_loss)
