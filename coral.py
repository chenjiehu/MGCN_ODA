import numpy as np
import torch
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def AutoCORAL(target):  #coupute autocorrelation of one class
    d = target.size(1)
    nt = target.size(0)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    weight = ct.pow(2).sum().sqrt()
    weight = weight / (4 * d * d)

    return weight

def AutoCORAL2(target):  #coupute autocorrelation of one class
    d = target.size(1)
    nt = target.size(0)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    weight = ct.pow(2).sum().sqrt()

    return weight

def compute_mean_for_every_class(feature, label):
    C = len(np.unique(label))
    M,N =feature.size(0), feature.size(1)
    weight = torch.ones(C, N).to(DEVICE)
    C_count = 0
    for c in range(C):
        feature_c = feature[np.where(label == c)]
        Nc = len(feature_c)
        if Nc > 1:
            weight[c,:] = torch.mean(feature_c, dim=0)
            C_count = C_count + 1
        elif Nc == 1:
            weight[c,:] = feature_c
        else:
            print('compute mean error!')
    # if C_count == C:
    #     weight = weight
    # else:
    #     weight = torch.ones(C).to(DEVICE)
    return weight

def compute_distance_matrix(feature, label):
    C = len(np.unique(label))
    A = torch.zeros(C,C)
    M,N =feature.size(0), feature.size(1)
    mean = compute_mean_for_every_class(feature, label)
    if mean.size(0) == C:
        temp = torch.norm(mean.repeat(C,1) - mean.repeat(1,C).view(-1,N),dim=1)
        A = temp.view(C,C)
    else:
        print('error occur when compute A')
    return A

def compute_aifa_for_every_class(feature, target_label, source_label):  #compute autocorelation for eatch classes
    C = len(np.unique(source_label))
    eps = np.finfo(float).eps
    weight = torch.ones(C).to(DEVICE)
    if C == len(np.unique(target_label)):
        C_count = 0
        matrix_temp = compute_distance_matrix(feature, target_label.cpu())
        _, position = torch.topk(matrix_temp, 2, dim=1, sorted=False, largest=False,out=None)
        #position_temp = position[:,1]
        matrix_temp = matrix_temp + eps
        #distance_nearest = matrix_temp[position_temp] + eps
        for c in range(C):
            feature_c = feature[np.where(target_label == c)]
            Nc = len(feature_c)
            if Nc > 1:
                weight[c] = AutoCORAL2(feature_c)/matrix_temp[c,position[c,1]]
                C_count = C_count + 1

    # print('weight:',weight)
    return weight

def compute_aifa_W(feature, target_label, source_label):  #compute autocorelation for eatch classes
    C = len(np.unique(source_label))
    eps = np.finfo(float).eps
    aifa = 0.8
    weight = torch.ones(C).to(DEVICE)
    if C == len(np.unique(target_label)):
        C_count = 0
        matrix_temp = compute_distance_matrix(feature, target_label.cpu())
        _, position = torch.topk(matrix_temp, 2, dim=1, sorted=False, largest=False,out=None)
        #position_temp = position[:,1]
        matrix_temp = matrix_temp + eps
        #distance_nearest = matrix_temp[position_temp] + eps
        for c in range(C):
            feature_c = feature[np.where(target_label == c)]
            Nc = len(feature_c)
            if Nc > 1:
                weight[c] = AutoCORAL2(feature_c)/matrix_temp[c,position[c,1]]
                C_count = C_count + 1
        if C_count == C:
            weight_mean = torch.mean(weight)
            weight = weight/weight_mean
        weight = F.sigmoid(aifa * weight)
        weight_mean = torch.mean(weight)
        weight = weight / weight_mean
        weight = weight * 1
    return weight

def compute_aifa_for_class(feature, target_label, source_label):  #compute autocorelation for eatch classes
    C = len(np.unique(source_label))
    eps = np.finfo(float).eps
    aifa = 0.65
    weight = torch.ones(C).to(DEVICE)
    if C == len(np.unique(target_label)):
        C_count = 0
        matrix_temp = compute_distance_matrix(feature, target_label.cpu())
        _, position = torch.topk(matrix_temp, 2, dim=1, sorted=False, largest=False,out=None)
        #position_temp = position[:,1]
        matrix_temp = matrix_temp + eps
        #distance_nearest = matrix_temp[position_temp] + eps
        for c in range(C):
            feature_c = feature[np.where(target_label == c)]
            Nc = len(feature_c)
            if Nc > 1:
                weight[c] = AutoCORAL2(feature_c)/matrix_temp[c,position[c,1]]
                C_count = C_count + 1

    print('weight:',weight)
    return weight

if __name__ =='__main__':
    feature = torch.randn(1000, 30).to(DEVICE)
    label = torch.arange(5).repeat(200)
    print(label)
    feature1 = torch.ones(10,3).to(DEVICE)
    feature1[:,1] = 3
    feature1[:, 2] = 2
    feature1[3,:] = 3
    print(feature1)
    auto = AutoCORAL(feature1)
    print(auto)