import torch
def get_nsamples(data_loader, N):
    x = []
    n = 0
    while n < N:
        x_next, _ = next(iter(data_loader))
        x.append(x_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    return x