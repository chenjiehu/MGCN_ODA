import torch



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


def wmmd(source, target, target_pred, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    length_source = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)

    length_pred = int(target_pred.size()[0])
    length_target = int(target.size()[0])
    if length_pred == length_target:

        Pi = torch.sum(target_pred)
        target_pred_matrix = target_pred.unsqueeze(1)
        Mtt = torch.mm(target_pred_matrix, target_pred_matrix.t())/Pi**2
        target_pred_matrix = target_pred.repeat(length_source,1)
        Mst = (-1/(Pi*length_source))*torch.mul(torch.ones(length_source, length_target).cuda(),target_pred_matrix)
        Mss = (1/length_source**2)*torch.ones(length_source, length_source).cuda()
        M1 = torch.cat([Mss,Mst],dim=1)
        M2 = torch.cat([Mst.t(),Mtt], dim=1)
        M = torch.cat([M1,M2],dim = 0)

        loss = torch.trace(torch.mm(kernels.float(),M))

    else:
        XX = kernels[:length_source, :length_source]  # Source<->Source
        YY = kernels[length_source:, length_source:]  # Target<->Target
        XY = kernels[:length_source, length_source:]  # Source<->Target
        YX = kernels[length_source:, :length_source]  # Target<->Source
        loss = torch.mean(XX + YY - XY - YX)  #

    return loss

def wmmd2(source, target, target_pred, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
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



