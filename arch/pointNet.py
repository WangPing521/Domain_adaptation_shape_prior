import torch.nn as nn
import numpy as np
from sklearn.metrics import pairwise_distances
from torch.nn import functional as F

class PointNet(nn.Module):
    def __init__(self, num_points=300, fc_inch=121, conv_inch=256, ext=False):
        super().__init__()
        self.num_points = num_points
        self.ReLU = nn.LeakyReLU(inplace=True)
        # Final convolution is initialized differently form the rest
        if ext:
            self.conv1 = nn.Conv2d(conv_inch, conv_inch * 2, kernel_size=(3,3), padding=(1,1))
            self.conv2 = nn.Conv2d(conv_inch * 2, conv_inch, kernel_size=(3,3), padding=(1,1))
        self.final_conv = nn.Conv2d(conv_inch, self.num_points, kernel_size=(6,6))
        self.final_fc = nn.Linear(fc_inch, 3)
        self._ext = ext

    def forward(self, x):
        if self._ext:
            x = self.ReLU(self.conv1(x))
            x = self.ReLU(self.conv2(x))
        x = self.ReLU(self.final_conv(x))
        x = x.view(x.size(0), x.size(1), -1)
        x = self.final_fc(x)
        return x # [n, 300, 3]

class STN3d(nn.Module):
    """
    Spatial transformer network
    Compute transformation matrix of the inputted pointcloud
    """
    def __init__(self, dim=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.in1 = nn.InstanceNorm1d(64, track_running_stats=True)
        self.in2 = nn.InstanceNorm1d(128, track_running_stats=True)
        self.in3 = nn.InstanceNorm1d(1024, track_running_stats=True)
        self.in4 = nn.InstanceNorm1d(512, track_running_stats=True)
        self.in5 = nn.InstanceNorm1d(256, track_running_stats=True)


    def forward(self, x):
        batchsize = x.size()[0]
        if batchsize > 1:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.in1(self.conv1(x)))
            x = F.relu(self.in2(self.conv2(x)))
            x = F.relu(self.in3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = F.relu(self.in4(self.fc1(x)))
            x = F.relu(self.in5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, sample_transform=True, kernel_size=1, stride=1, in_channel=3, dim=3, ext=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(dim=dim)
        self._ext = ext
        if self._ext:
            self.conv1 = torch.nn.Conv1d(in_channel, 8, kernel_size, stride, kernel_size // 2)
            self.bn1 = nn.BatchNorm1d(8)
            self.conv1_1 = torch.nn.Conv1d(8, 64, kernel_size, stride, kernel_size // 2)
            self.bn1_1 = nn.BatchNorm1d(64)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size, stride, kernel_size // 2)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv2_1 = torch.nn.Conv1d(128, 256, kernel_size, stride, kernel_size // 2)
            self.bn2_1 = nn.BatchNorm1d(256)
            self.conv3 = torch.nn.Conv1d(256, 512, kernel_size, stride, kernel_size // 2)
            self.bn3 = nn.BatchNorm1d(512)
            self.conv3_1 = torch.nn.Conv1d(512, 1024, kernel_size, stride, kernel_size // 2)
            self.bn3_1 = nn.BatchNorm1d(1024)
        else:
            self.conv1 = torch.nn.Conv1d(in_channel, 64, kernel_size, stride, kernel_size // 2)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size, stride, kernel_size // 2)
            self.conv3 = torch.nn.Conv1d(128, 1024, kernel_size, stride, kernel_size // 2)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self._sample_transform = sample_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = None
        if self._sample_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self._ext:
            x = F.relu(self.bn1_1(self.conv1_1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        if self._ext:
            x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn3(self.conv3(x))
        if self._ext:
            x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, feature_transform=False, sample_transform=True, kernel_size=1, stride=1, in_channel=3, dim=3, ext=False, drop=0.3):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, sample_transform=sample_transform,
                                 kernel_size=kernel_size, stride=stride, in_channel=in_channel, dim=dim, ext=ext)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=drop)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.in1 = nn.InstanceNorm1d(512, track_running_stats=True)
        self.in2 = nn.InstanceNorm1d(256, track_running_stats=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        batchsize = x.size()[0]
        if batchsize > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.in1(self.fc1(x)))
            x = F.relu(self.in2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.relu(x)
        return x, trans, trans_feat


def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    """

    N = 300
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)

import torch
if __name__ == '__main__':
    # img = torch.randn(5, 256, 16, 16)
    # pointnet = PointNet()
    # out1 = pointnet(img)

    t = np.linspace(0, 2 * np.pi, 101)[0:100]
    X1 = np.zeros((len(t), 2))
    X1[:, 0] = np.cos(t)
    X1[:, 1] = np.sin(t)
    t = np.linspace(0, 2 * np.pi, 1001)[0:1000]
    X2 = np.zeros((len(t), 2))
    X2[:, 0] = 2 * np.cos(t) + 5
    X2[:, 1] = 2 * np.sin(t)
    X3 = np.zeros((len(t), 2))
    X3[:, 0] = 3 * np.cos(t)
    X3[:, 1] = 3 * np.sin(t) + 3
    X = np.concatenate((X1, X2, X3), 0)

    D = pairwise_distances(X, metric='euclidean')
    (perm, lambdas) = getGreedyPerm(D)
