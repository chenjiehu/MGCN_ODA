#首先使用Nway Kshot的方法，利用源域数据对模型进行训练，然后利用模型对目标域数据进行测试，然后对预测出属于已知类的目标域数据集
#进行域适应（利用ＭＭＤ），对ＭＭＤ进行修正，使得概率大的样本带来的损失函数值较大，概率小的样本计算出的损失函数值较小
import argparse
import os.path as osp
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn.functional as F
import xlsxwriter
from torch.utils.data import DataLoader

from SOURCE import SOURCE_DATA, TARGET_DATA
from MMD import mmd
from samplers import CategoriesSampler, CategoriesSampler_shot
from convnet import Convnet
from GCN_layers import GCN
from Multi_gcn import MultiGCN, MultiGCN_sigma, MultiGCN_progation
from backbone import ResNet34, Conv6, ResNet10, ResNet18, ResNet34, Conv4
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 初始化参数
    parser.add_argument('--max-epoch', type=int, default=60)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)          ###    #####    ##
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=10)
    parser.add_argument('--test-way', type=int, default=20)
    parser.add_argument('--load', default='./save/Conv4/step1/RA/10way_5shot')
    parser.add_argument('--save-path', default='./save/Conv4/step2/RA/10way_5shot')      ###    #####    #######    #########################
    parser.add_argument('--gpu', default='1')
    # parser.add_argument('--model', default='ResNet10',
    #                     help='model: Conv{4|6} / ResNet{10|18|34|50|101}')  # 50 and 101 are not used in the paper
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    torch.manual_seed(3)
    ensure_path(args.save_path)

    trainset = SOURCE_DATA('RA_10wayRSI')       ###    ###    #####     #####################################
    train_sampler = CategoriesSampler(trainset.label, 50,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    valset = TARGET_DATA('RA_20wayAID_val')       ###    ###    #####     #####################################
    val_sampler = CategoriesSampler(valset.label, 50,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)


    model_CNN = Conv4().cuda()                                        #========================================================
    #model_GCN = GCN(nfeat=1600,nhid=1000,nclass=1000,dropout=0.5,init='xavier',N_way=args.test_way).cuda()
    model_GCN = MultiGCN(input_dim=1600, N_way=args.test_way).cuda()

    optimizer_CNN = torch.optim.Adam(model_CNN.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_CNN, step_size=20, gamma=0.5)
    optimizer_GCN = torch.optim.Adam(model_GCN.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_GCN, step_size=20, gamma=0.5)

    model_CNN.load_state_dict(torch.load(osp.join(args.load, 'proto-max-acc.pth')))

    for epoch in range(1, args.max_epoch + 1):

        def save_model(name):
            torch.save(model_CNN.state_dict(), osp.join(args.save_path, name + 'CNN.pth'))
            torch.save(model_GCN.state_dict(), osp.join(args.save_path, name + 'GCN.pth'))

        workbook = xlsxwriter.Workbook('./datasave.xlsx')
        worksheet = workbook.add_worksheet()

        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0

        timer = Timer()
        lr_scheduler.step()  # 补偿自动调整

        model_CNN.train()  # 模型开始训练
        model_GCN.train()

        start_time = time.time()

        tl = Averager()
        ta = Averager()

        for i, source_batch in enumerate(train_loader):
            data, source_label = [_.cuda() for _ in source_batch]
            p = args.shot * args.train_way# 选取每个batch的前p列作为suppot集，后面作为查询级

            source_feature = model_CNN(data)# CNN提取特征
            source_feature = model_GCN(source_feature) #GCN提取特征
            proto = source_feature[:p]
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            query_feature = source_feature[p:]

            logits = euclidean_metric(query_feature, proto)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))
            loss_all = loss #+ MMD_loss
            print('loss all:', loss_all)
            tl.add(loss.item())
            ta.add(acc)

            optimizer_CNN.zero_grad()
            optimizer_GCN.zero_grad()
            loss_all.backward()
            optimizer_CNN.step()
            optimizer_GCN.step()

            proto = None;
            logits = None;
            loss = None

        end_time = time.time()
        print(end_time-start_time)
        tl = tl.item()
        ta = ta.item()

        with torch.no_grad():
            model_CNN.eval()
            model_GCN.eval()

            vl = Averager()
            va = Averager()

            for i, batch in enumerate(val_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way

                target_feature = model_CNN(data)
                target_feature = model_GCN(target_feature)
                proto = target_feature[:p]
                proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
                query_feature = target_feature[p:]

                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)

                logits = euclidean_metric(query_feature, proto)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)

                vl.add(loss.item())
                va.add(acc)

                proto = None;
                logits = None;
                loss = None

            vl = vl.item()
            va = va.item()
            print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

            if va > trlog['max_acc']:
                trlog['max_acc'] = va
                save_model('proto-max-acc')

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl)
            trlog['val_acc'].append(va)

            torch.save(trlog, osp.join(args.save_path, 'trlog'))

            save_model('epoch-last')

            if epoch % args.save_epoch == 0:
                save_model('epoch-{}'.format(epoch))

            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))