import torch

w0 = torch.tensor([
    [[1,1],[2,2],[3,3]],
    [[4,4],[5,5],[6,6]]
])
w1 = torch.tensor([
    [3,3,3],
    [1,1,1]
])

#w = torch.einsum('bsd,bd->bs', (w0, w1))
w = torch.einsum('bs,bsd->bd', (w1, w0))
print('w', w)