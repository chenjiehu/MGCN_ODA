import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class MultiGCN(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)

        self.gcn = GraphConvolution(input_dim, 1000)

        self.aifa1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.aifa2 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.aifa3 = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.weight = Parameter(torch.FloatTensor(input_dim, 1000))
        self.aifa1.data.fill_(0)
        self.aifa2.data.fill_(0)
        self.aifa3.data.fill_(0)


        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):

        A = self.MultiAdjacencyCompute(features)
        x = self.gcn(A, features)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None, largest=True)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)

        data = 0.5

        aifa = F.softmax(torch.cat([self.aifa1,self.aifa2,self.aifa3],dim=0),dim=0)

        adjacency = aifa[0]*torch.eye(N).cuda() + aifa[1]*adjacencyn + aifa[2]*torch.mm(adjacencyn,adjacencyn)

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

class MultiGCN_relation(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)

        self.gcn = GraphConvolution(input_dim, 1000)
        self.relation = RelationNetwork()

        self.aifa1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.aifa2 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.aifa3 = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.weight = Parameter(torch.FloatTensor(input_dim, 1000))
        self.aifa1.data.fill_(0)
        self.aifa2.data.fill_(0)
        self.aifa3.data.fill_(0)

        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):

        A = self.MultiAdjacencyCompute(features)
        x = self.gcn(A, features)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        eps = np.finfo(float).eps
        N = features.size(0)
        sigma = self.relation(features)
        features = features / (sigma + eps)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)

        adjacency_e = torch.exp(-temp.pow(2)).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None, largest=True)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)

        data = 0.5

        aifa = F.softmax(torch.cat([self.aifa1,self.aifa2,self.aifa3],dim=0),dim=0)

        adjacency = aifa[0]*torch.eye(N).cuda() + aifa[1]*adjacencyn + aifa[2]*torch.mm(adjacencyn,adjacencyn)

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

class RelationNetwork(nn.Module):
    """Graph Construction Module"""

    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x):
        x = x.view(-1, 64, 5, 5)

        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out

