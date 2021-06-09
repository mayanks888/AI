from flopth import flopth
import torch.nn as nn


class TwoLinear(nn.Module):
    def __init__(self):
        super(TwoLinear, self).__init__()

        self.l1 = nn.Linear(10, 1994)
        self.l2 = nn.Linear(1994, 10)

    def forward(self, x, y):
        x = self.l1(x) * y
        x = self.l2(x) * y

        return x


m = TwoLinear()

# sum_flops = flopth(m, in_size=[[10], [10]])
sum_flops = flopth(m, in_size=[10])
print(sum_flops)