import torch
from sklearn import metrics
import sklearn.metrics
import numpy as np
from predict_A_distance import proxy_a_distance

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1), None, gamma)
    return K

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def wmmd(source, target, target_pred, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    length_source = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)

    length_pred = int(target_pred.size()[0])
    length_target = int(target.size()[0])
    loss = 0
    if length_pred == length_target:

        Pi = torch.sum(target_pred)
        target_pred_matrix = target_pred.unsqueeze(1)
        Mtt = torch.mm(target_pred_matrix, target_pred_matrix.t())/Pi**2
        target_pred_matrix = target_pred.repeat(length_source,1)
        Mst = (-1/(Pi*length_source))*torch.mul(torch.ones(length_source, length_target).cuda(),target_pred_matrix)
        Mss = (1/length_source**2)*torch.ones(length_source, length_source).cuda()
        M1 = torch.cat([Mss,Mst],dim = 1)
        M2 = torch.cat([Mst.t(),Mtt], dim = 1)
        M = torch.cat([M1,M2],dim = 0)

        loss = torch.trace(torch.mm(kernels.float(),M))
    return loss

#加权BDA算法
def BDA(source, source_label ,target, target_pred_label):

    loss = 0

    length_source = int(source.size()[0])
    length_target = int(target.size()[0])
    length_common = min(length_source, length_target)

    length_pred_label = int(target_pred_label.size()[0])
    length_source_label = int(source_label.size()[0])

    n = length_source + length_target

    if (length_source == length_source_label) & (length_pred_label == length_target):
#首先计算条件分布差异
        C = len(np.unique(source_label))
        for c in range(C):
            e = np.zeros((n, 1))
            Ns = len(source_label[np.where(source_label == c)])
            Nt = len(target_pred_label[np.where(target_pred_label == c)])
            if (Nt > 1) & (Ns > 1):
                tt = source_label == c
                e[np.where(tt == True)] = 1 / Ns
                ind = np.where(tt == True)
                inds_source = [item for item in ind]

                yy = target_pred_label == c
                ind = np.where(yy == True)
                inds_target = [item for item in ind]
                e[tuple(inds_target)] = -1 / Nt
                e[np.isinf(e)] = 0

                source_c = source[inds_source,:]
                source_c = np.squeeze(source_c)
                target_c = target[inds_target,:]
                target_c = np.squeeze(target_c)
                loss_c = mmd(source_c,target_c)
                loss = loss + loss_c
#计算边缘分布
        loss_B = mmd(source[:length_common], target[:length_common])
        loss_all = 1/C*loss + loss_B
    else:
        print('imput length eooro!')
    return  loss_all

def wBDA(source, source_label ,target, target_pred_label, target_pred_p):

    loss = 0
    mu = 0.1

    length_source = int(source.size()[0])
    length_target = int(target.size()[0])

    length_pred_p = int(target_pred_p.size()[0])
    length_pred_label = int(target_pred_label.size()[0])
    length_source_label = int(source_label.size()[0])

    n = length_source + length_target

    if (length_source == length_source_label) & (length_pred_label == length_target):
        C = len(np.unique(source_label))
        for c in range(C):
            e = np.zeros((n, 1))
            Ns = len(source_label[np.where(source_label == c)])
            Nt = len(target_pred_label[np.where(target_pred_label == c)])
            if (Nt > 1) & (Ns > 1):
                tt = source_label == c
                e[np.where(tt == True)] = 1 / Ns
                ind = np.where(tt == True)
                inds_source = [item for item in ind]

                yy = target_pred_label == c
                ind = np.where(yy == True)
                inds_target = [item for item in ind]
                e[tuple(inds_target)] = -1 / Nt
                e[np.isinf(e)] = 0

                source_c = source[inds_source,:]
                source_c = np.squeeze(source_c)
                target_c = target[inds_target,:]
                target_c = np.squeeze(target_c)
                target_pred_c = target_pred_p[inds_target]
                loss_c = wmmd(source_c,target_c,target_pred_c)
                loss = loss + loss_c
#计算边缘分布
        loss_B = wmmd(source, target, target_pred_p)
        #loss_all = 1/C*loss + loss_B
        loss_all = (1 - mu) * loss / C + mu * loss_B
    else:
        print('imput length eooro!')
    return  loss_all

def coral_wBDA(source, source_label ,target, target_pred_label, target_pred_p, coral_W):

    loss = 0
    loss_all = 0
    mu = 0.1

    length_source = int(source.size()[0])
    length_target = int(target.size()[0])

    length_pred_p = int(target_pred_p.size()[0])
    length_pred_label = int(target_pred_label.size()[0])
    length_source_label = int(source_label.size()[0])

    n = length_source + length_target

    if (length_source == length_source_label) & (length_pred_label == length_target):

        C = len(np.unique(source_label))
        if C != coral_W.size(0):
            coral_W = torch.ones(C)
        for c in range(C):
            e = np.zeros((n, 1))
            Ns = len(source_label[np.where(source_label == c)])
            Nt = len(target_pred_label[np.where(target_pred_label == c)])
            if (Nt > 1) & (Ns > 1):
                tt = source_label == c
                e[np.where(tt == True)] = 1 / Ns
                ind = np.where(tt == True)
                inds_source = [item for item in ind]

                yy = target_pred_label == c
                ind = np.where(yy == True)
                inds_target = [item for item in ind]
                e[tuple(inds_target)] = -1 / Nt
                e[np.isinf(e)] = 0

                source_c = source[inds_source,:]
                source_c = np.squeeze(source_c)
                target_c = target[inds_target,:]
                target_c = np.squeeze(target_c)
                target_pred_c = target_pred_p[inds_target]
                loss_c = wmmd(source_c,target_c,target_pred_c)
                loss = loss + loss_c * coral_W[c]
        loss_B = wmmd(source, target, target_pred_p)
        loss_all = (1-mu) * loss/C  +  mu * loss_B
    else:
        print('imput length error!')
    return  loss_all


def coral_wBDA_A_distance(source, source_label ,target, target_pred_label, target_pred_p, coral_W):

    loss = 0

    length_source = int(source.size()[0])
    length_target = int(target.size()[0])

    length_pred_p = int(target_pred_p.size()[0])
    length_pred_label = int(target_pred_label.size()[0])
    length_source_label = int(source_label.size()[0])

    n = length_source + length_target
    dc = 0

    if (length_source == length_source_label) & (length_pred_label == length_target):

        C = len(np.unique(source_label))
        if C != coral_W.size(0):
            coral_W = torch.ones(C)
        for c in range(C):
            e = np.zeros((n, 1))
            Ns = len(source_label[np.where(source_label == c)])
            Nt = len(target_pred_label[np.where(target_pred_label == c)])
            if (Nt > 1) & (Ns > 1):
                tt = source_label == c
                e[np.where(tt == True)] = 1 / Ns
                ind = np.where(tt == True)
                inds_source = [item for item in ind]

                yy = target_pred_label == c
                ind = np.where(yy == True)
                inds_target = [item for item in ind]
                e[tuple(inds_target)] = -1 / Nt
                e[np.isinf(e)] = 0

                source_c = source[inds_source,:]
                source_c = np.squeeze(source_c)
                target_c = target[inds_target,:]
                target_c = np.squeeze(target_c)
                target_pred_c = target_pred_p[inds_target]
                loss_c = wmmd(source_c,target_c,target_pred_c)
                dc = dc + proxy_a_distance(source_c.cpu().detach().numpy(), target_c.cpu().detach().numpy())
                loss = loss + loss_c * coral_W[c]

        loss_B = wmmd(source, target, target_pred_p)
        dm = proxy_a_distance(source.cpu().detach().numpy(),target.cpu().detach().numpy())
        mu = 1 - dm/(dm+dc)
        loss_all = (1-mu)*1/C*loss  +  mu*loss_B
        loss_all = 1.6 * loss_all
    else:
        print('imput length eooro!')
    return  loss_all

def aifa_wBDA_A_distance(source, source_label ,target, target_pred_label, target_pred_p, coral_W):

    loss = 0

    length_source = int(source.size()[0])
    length_target = int(target.size()[0])

    length_pred_p = int(target_pred_p.size()[0])
    length_pred_label = int(target_pred_label.size()[0])
    length_source_label = int(source_label.size()[0])

    n = length_source + length_target
    dc = 0

    if (length_source == length_source_label) & (length_pred_label == length_target):
        C = len(np.unique(source_label))
        if C != coral_W.size(0):
            coral_W = torch.ones(C)

        for c in range(C):
            e = np.zeros((n, 1))
            Ns = len(source_label[np.where(source_label == c)])
            Nt = len(target_pred_label[np.where(target_pred_label == c)])
            if (Nt > 1) & (Ns > 1):
                tt = source_label == c
                e[np.where(tt == True)] = 1 / Ns
                ind = np.where(tt == True)
                inds_source = [item for item in ind]

                yy = target_pred_label == c
                ind = np.where(yy == True)
                inds_target = [item for item in ind]
                e[tuple(inds_target)] = -1 / Nt
                e[np.isinf(e)] = 0

                source_c = source[inds_source,:]
                source_c = np.squeeze(source_c)
                target_c = target[inds_target,:]
                target_c = np.squeeze(target_c)
                target_pred_c = target_pred_p[inds_target]
                loss_c = wmmd(source_c,target_c,target_pred_c)
                dc = dc + proxy_a_distance(source_c.cpu().detach().numpy(), target_c.cpu().detach().numpy())
                loss = loss + loss_c * coral_W[c]

        loss_B = wmmd(source, target, target_pred_p)
        dm = proxy_a_distance(source.cpu().detach().numpy(),target.cpu().detach().numpy())
        mu = 1 - dm/(dm+dc)
        loss_all = (1-mu)*1/C*loss  +  mu*loss_B
        loss_all = 1.6 * loss_all
    else:
        print('imput length ERROR!')
    return  loss_all


