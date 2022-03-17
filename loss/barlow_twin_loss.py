import torch.nn as nn
import torch



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        assert z1.shape[1] == z2.shape[1]
        feature_dim = z1.shape[1]
        # normalization layer for the representations z1 and z2
        bn = nn.BatchNorm1d(feature_dim, affine=False)

        # empirical cross-correlation matrix
        c = bn(z1).T @ bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss