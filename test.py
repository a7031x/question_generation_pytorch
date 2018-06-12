import torch

w0 = torch.nn.Linear(2, 2)
w1 = torch.nn.Linear(2, 2)

a = torch.tensor([1, 2]).float()
b = torch.tensor([2, 3]).float()

criterion = torch.nn.MSELoss()
optimizer0 = torch.optim.Adam(w0.parameters())
optimizer1 = torch.optim.Adam(list(w0.parameters()) + list(w1.parameters()))

print('w0', w0.weight.tolist())
print('w1', w1.weight.tolist())

while True:
    loss = criterion(w0(a) - w1(b), torch.tensor([0,0]).float())
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    if loss.tolist() < 0.0001:
        break

print('w0', w0.weight.tolist())
print('w1', w1.weight.tolist())
print('w0*a', w0(a))
print('w1*b', w1(b))