import torch


def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)


if __name__=='__main__':

    # p1 = torch.tensor(0.1)
    # p2 = torch.tensor(0.4)
    # p = torch.cat([p1,p2])

    p1 = torch.tensor([0.5,0.9]).unsqueeze(1)
    p2 = torch.tensor([0.1,0.8]).unsqueeze(1)
    p = torch.cat([p1,p2],dim=1)

    loss = EntropyLoss(p)
    print(loss)
