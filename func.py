import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor(v):
    if isinstance(v, torch.Tensor):
        return v
    else:
        return torch.tensor(v, device=device)


def sparse_softmax_loss(logit, target, mask=1, size_average=False, reduce=False):
    logit = torch.nn.functional.log_softmax(logit, -1)
    for dim in range(len(logit.shape)-1, 1, -1):
        logit = logit.transpose(dim, dim-1)
    loss = torch.nn.NLLLoss(reduce=False, size_average=False)(logit, target) * mask.float()
    if reduce:
        loss = loss.sum()
    if size_average:
        loss = loss / tensor(mask).sum()
    return loss