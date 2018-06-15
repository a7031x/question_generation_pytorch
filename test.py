import torch

a = torch.tensor([-1, 1]).float()
b = torch.tensor([0, 1])

criterion = torch.nn.CrossEntropyLoss()
#w = torch.einsum('bsd,bd->bs', (w0, w1))
w = torch.einsum('bs,bsd->bd', (w1, w0))
print('w', w)