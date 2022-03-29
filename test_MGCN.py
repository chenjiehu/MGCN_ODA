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
import pandas as pd
from backbone import ResNet10, ResNet18, ResNet34, Conv4, Conv6
from torch.utils.data import DataLoader

from SOURCE import SOURCE_DATA, TARGET_DATA, get_shot_data,QUERY_DATA
from MMD import mmd
from samplers import CategoriesSampler, CategoriesSampler_shot
from convnet import Convnet
from utils import pprint, set_gpu, delete_path, ensure_path, Averager, Timer, count_acc, euclidean_metric, cal_acc2, find_p_less_index, Averager_vector, find_p_greater_index
from MMD import mmd
from wMMD import wmmd2
from wBDA import wBDA, coral_wBDA
from plot_confusion_matrix import plot_confusion_matrix
from GCN_layers import GCN
from Multi_gcn import ProtoGCN, MultiGCN, MultiGCN_sigma, MGCN
from EntropyLoss import EntropyLoss
from coral import compute_coral_W, compute_aifa_for_every_class
NUM_shot = 5

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 初始化参数
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    # parser.add_argument('--train-way', type=int, default=6)
    parser.add_argument('--test-way', type=int, default=10)     ###    #####    #######    #########################
    #parser.add_argument('--load', default='./save/train_SADU_WBDA_MGCN_step2/Tibet/3way_6way_1shot')
    #parser.add_argument('--load', default='./save/train_SADU_WBDA_MGCN_step2_Tibet/Tibet/3way_6way_1shot')
    parser.add_argument('--load', default='./save/Conv4/step2/AR/10way_5shot')   #close denotes the closed set DA
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--same-class-num', default=4)
    args = parser.parse_args()
    pprint(vars(args))
    torch.manual_seed(10)

    set_gpu(args.gpu)

    # 训练模型
    model_CNN = Conv4().cuda()
    #model_GCN = MultiGCN(input_dim=1600, N_way=args.test_way).cuda()
    model_GCN = MGCN(input_dim=1600, N_way=args.test_way).cuda()

    workbook = xlsxwriter.Workbook('./datasave.xlsx')
    worksheet = workbook.add_worksheet()

    trlog = {}

    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []

    timer = Timer()

    tl = Averager()
    ta = Averager()

    ave_acc = Averager()
    ave_acc_test = Averager()
    vv = Averager_vector(args.test_way)
    coral_for_every_class = Averager_vector(args.test_way)
    coral_mean = Averager_vector(args.test_way)
    coral_for_every_class_CNN = Averager_vector(args.test_way)
    coral_mean_CNN = Averager_vector(args.test_way)


    label_rem = []
    pred_label_rem = []
    feature_rem = []
    label_rem = []

    with torch.no_grad():


        for ii in range(0,NUM_shot):

            model_CNN.load_state_dict(torch.load(osp.join(args.load, 'proto-max-accCNN.pth')))
            model_GCN.load_state_dict(torch.load(osp.join(args.load, 'proto-max-accGCN.pth')))
            model_CNN.eval()
            model_GCN.eval()

            target_shot_data, target_shot_label = get_shot_data('./shot_AR/5_shot' + str(ii) + '.csv')
            target_shot_data = target_shot_data.cuda()
            target_shot_label = target_shot_label.cuda()

            valset = QUERY_DATA('./query_AR/5_query' + str(
                ii) + '.csv')  ###    ###    #####     #####################################
            val_sampler = CategoriesSampler(valset.label, 20,
                                            args.test_way, args.query)
            val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                    num_workers=8, pin_memory=True)


            start_time = time.time()
            for i, batch in enumerate(val_loader, 1):
                data, label = [_.cuda() for _ in batch]
                data = torch.cat([target_shot_data, data], dim=0)
                target_feature_CNN = model_CNN(data)
                target_feature = model_GCN(target_feature_CNN)


                p = args.shot * args.test_way
                target_query_feature = target_feature[p:]  # it is easy to make mistake here
                target_shot_proto = target_feature[:p]
                target_shot_proto = target_shot_proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
                logits = euclidean_metric(target_query_feature, target_shot_proto)
                target_pred = F.softmax(logits, dim=1)

                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                ave_acc_test.add(acc)

                target_pred_label = target_pred.argmax(dim=1)
                pred_class = cal_acc2(label.cpu(), target_pred_label.cpu(), args.test_way)
                vv.add(pred_class)
                label_rem.append(label)
                pred_label_rem.append(target_pred_label)
                coral_temp = compute_aifa_for_every_class(target_query_feature, label[p:].cpu(),label[p:].cpu())
                coral_for_every_class.add(coral_temp.cpu())
                coral_temp = compute_aifa_for_every_class(target_feature_CNN, label[p:].cpu(),label[p:].cpu())
                coral_for_every_class_CNN.add(coral_temp.cpu())


                proto = None;
                logits = None;
                loss = None


                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc_test.item() * 100, acc * 100))


            end_time = time.time()
            print('time: %s' % (end_time - start_time))
            print(vv.item())

            print(coral_for_every_class_CNN.item())

            mean_coral_CNN = torch.mean(coral_for_every_class_CNN.item())

            print(mean_coral_CNN)

            print(coral_for_every_class.item())

            mean_coral = torch.mean(coral_for_every_class.item())

            print(mean_coral)



            print('aifa:', coral_for_every_class.item())
            aifa_mean = coral_for_every_class.item()
            aifa_com = torch.mean(aifa_mean[:args.same_class_num])
            aifa_pri = torch.mean(aifa_mean[args.same_class_num:])
            print('aifa_com:', aifa_com)
            print('aifa_pri:', aifa_pri)
            aifa_mean = torch.mean(aifa_mean)
            print('aifa_mean:',aifa_mean)





#plot_confusion_matrix(label_rem, pred_label_rem, title='Confusion Matrix', dataset='Tibet')