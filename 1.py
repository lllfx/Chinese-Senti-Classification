import torch
from torch.nn import functional as F

input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)
loss = F.cross_entropy(input, target)

print(input.shape)
print(target.shape)