import torch

a = torch.tensor([1,2,3,4,5])
b = a.tolist()
b[3] = 6
a[3] = 7
print(a.tolist(), b)