import torch
from torchvision import transforms

mu = 2
std = 0.5
t = torch.Tensor([1,2,3])
print(t)
print((t - 2)/0.5)
# or if t is an image
# transforms.Normalize(2, 0.5)(t)

################################################################
# standardisation
norm1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
norm = transforms.Normalize((0.5), (0.5))
x = torch.randn(3, 224, 224)
val_in=x.detach().numpy()
out = norm(x)

val=out.detach().numpy()

out_all = norm1(x)
val_all=out_all.detach().numpy()
print(out)

#######################################33333
# min max scaler
# rescale vectors to a desired range
x = torch.tensor([2, -1.0, 5, 6, 7])
if not all(x == 0): # non all-zero vector
    # linear rescale to range [0, 1]
    x -= x.min() # bring the lower range to 0
    x /= x.max() # bring the upper range to 1
    x # tensor([0.3750, 0.0000, 0.7500, 0.8750, 1.0000])
    # linear rescale to range [-1, 1]
    # x = 2*x - 1
    # x # tensor([-0.2500, -1.0000,  0.5000,  0.7500,  1.0000])

 #############################################
# min max scaler
# rescale vectors to a desired range
x1 = torch.tensor([2, -1.0, 5, 6, 7])
if not all(x1 == 0): # non all-zero vector
    # linear rescale to range [0, 1]
    x1 -= x1.min() # bring the lower range to 0
    x1 /= (x1.max()-x1.min()) # bring the upper range to 1
    x1 # tensor([0.3750, 0.0000, 0.7500, 0.8750, 1.0000])
    # linear rescale to range [-1, 1]
    x1 = 2*x1 - 1
    x1 # tensor([