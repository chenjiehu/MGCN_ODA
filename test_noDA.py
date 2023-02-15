import argparse
import os.path as osp
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn.functional as F
import xlsxwriter
import pandas as pd
from model.convert import Convnet
from torch.utils.data import DataLoader
from Dataset.Dataset import SOURCE_DATA, TARGET_DATA, get_shot_data,QUERY_DATA
from samplers import CategoriesSampler
from utils import pprint, set_gpu, Averager, Timer, count_acc, euclidean_metric, cal_acc2, Averager_vector
from utils import save_path_step2, NUM_shot, filename_shot, filename_query
from model.Multi_gcn import MultiGCN_relation
from coral import compute_aifa_for_every_class
from model.convert import Convnet


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=10)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=12)
    parser.add_argument('--load', default=save_path_step2)   #close denotes the closed set DA
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))
    torch.manual_seed(10)
    set_gpu(args.gpu)

    # 训练模型
    model_CNN = Convnet().cuda()
    #model_GCN = MultiGCN(input_dim=1600, N_way=args.test_way).cuda()
    model_GCN = MultiGCN_relation(input_dim=1600, N_way=args.test_way).cuda()

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

            target_shot_data, target_shot_label = get_shot_data(os.path.join(filename_shot,'5_shot' + str(ii) + '.csv'))
            target_shot_data = target_shot_data.cuda()
            target_shot_label = target_shot_label.cuda()

            valset = QUERY_DATA(os.path.join(filename_query,'5_query' + str(
                ii) + '.csv'))
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
                target_query_feature = target_feature[p:]
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
                #coral_temp = compute_aifa_for_every_class(target_query_feature, label[p:].cpu(),label[p:].cpu())
                #coral_for_every_class.add(coral_temp.cpu())
                #coral_temp = compute_aifa_for_every_class(target_feature_CNN, label[p:].cpu(),label[p:].cpu())
                #coral_for_every_class_CNN.add(coral_temp.cpu())


                proto = None;
                logits = None;
                loss = None

                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc_test.item() * 100, acc * 100))

            end_time = time.time()
            print('time: %s' % (end_time - start_time))
            print(vv.item())




