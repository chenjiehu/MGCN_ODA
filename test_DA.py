
import os.path as osp
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn.functional as F
import xlsxwriter
import numpy as np
from torch.utils.data import DataLoader

from Dataset import SOURCE_DATA, TARGET_DATA, QUERY_DATA, get_shot_data
from samplers import CategoriesSampler
from utils import pprint, ensure_path, Averager, count_acc, euclidean_metric, cal_acc2, find_p_less_index, Averager_vector, find_p_greater_index
from wBDA import coral_wBDA
from Multi_gcn import MultiGCN
from EntropyLoss import EntropyLoss
from backbone import Conv4
from coral import compute_aifa_W


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=6)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--repeat-number', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=10)
    parser.add_argument('--test-way', type=int, default=10)
    parser.add_argument('--test-way2', type=int, default=10)
    parser.add_argument('--save-path', default='./save/Conv4/step3_DA/RA/10way_5shot')
    parser.add_argument('--load', default='./save/Conv4/step2/RA/10way_5shot')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--same-class-num', default=7)
    args = parser.parse_args()
    # set_gpu(args.gpu)
    pprint(vars(args))
    torch.manual_seed(3)

    ensure_path(args.save_path)

    trainset = SOURCE_DATA('RA_10wayRSI')
    train_sampler = CategoriesSampler(trainset.label,50,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    model_CNN = Conv4().cuda()
    model_GCN = MultiGCN(input_dim=1600, N_way=args.test_way).cuda()

    model1_select_target = Conv4().cuda()
    model1_select_target.load_state_dict(torch.load(osp.join(args.load, 'proto-max-accCNN.pth')))
    model2_select_target = MultiGCN(input_dim=1600, N_way=args.test_way).cuda()
    model2_select_target.load_state_dict(torch.load(osp.join(args.load, 'proto-max-accGCN.pth')))


    def save_model(name,num):
        torch.save(model_CNN.state_dict(), osp.join(args.save_path, name + str(num) +'CNN.pth'))
        torch.save(model_GCN.state_dict(), osp.join(args.save_path, name + str(num) +'GCN.pth'))


    workbook = xlsxwriter.Workbook('./datasave.xlsx')
    worksheet = workbook.add_worksheet()


    trlog_allDA = {}
    trlog_allDA['max_acc'] = []

    trlog = {}

    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['acc_per_class'] = []
    trlog['mean_acc'] = []

    model_CNN.train()
    model_GCN.train()
    model1_select_target.eval()
    model2_select_target.eval()

    tl = Averager()
    ta = Averager()

    ave_acc = Averager()
    ave_acc_test = Averager()
    vv = Averager_vector(args.test_way2)
    vv_allDA = Averager_vector(args.test_way2)

    label_rem = []
    pred_label_rem = []

    for ii in range(0,args.repeat_number):

        print('train NUM:--------------------------------------------------------------',ii)
        # start training
        model_CNN.load_state_dict(torch.load(osp.join(args.load, 'proto-max-accCNN.pth')))
        model_GCN.load_state_dict(torch.load(osp.join(args.load, 'proto-max-accGCN.pth')))

        optimizer_CNN = torch.optim.Adam(model_CNN.parameters(), lr=0.001)
        lr_scheduler_CNN = torch.optim.lr_scheduler.StepLR(optimizer_CNN, step_size=20, gamma=0.5)
        optimizer_GCN = torch.optim.Adam(model_GCN.parameters(), lr=0.001)
        lr_scheduler_GCN = torch.optim.lr_scheduler.StepLR(optimizer_GCN, step_size=20, gamma=0.5)

        lr_scheduler_CNN.step()
        lr_scheduler_GCN.step()

        target_shot_data, target_shot_label = get_shot_data('./shot_RA/5_shot'+str(ii)+'.csv')
        target_shot_data = target_shot_data.cuda()
        target_shot_label = target_shot_label.cuda()

        valset = QUERY_DATA('./query_RA/5_query'+str(ii)+'.csv')
        val_sampler = CategoriesSampler(valset.label, 50,
                                        args.test_way, args.query)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=8, pin_memory=True)

        model_CNN.train() # phase of train
        model_GCN.train()
#        delete_path(args.save_path)
        trlog['max_acc'] = 0.0
        trlog_allDA['max_acc'] = 0.0
        pred_class_temp = []
        label_temp = []
        target_pred_label_temp = []
        iteration = []
        Loss = []
        Acc = []

        for epoch in range(1, args.max_epoch + 1):
            print('epoch:',epoch)

            for i, (source_batch, target_batch) in enumerate(zip(train_loader, val_loader)):

                source_data, source_label = [_.cuda() for _ in source_batch]
                target_data, target_label = [_.cuda() for _ in target_batch]
                p = args.shot * args.train_way
                source_shot_data, source_query_data = source_data[:p], source_data[p:]

    #======================================== 　Train the network with the source domain data

                source_feature = model_CNN(source_data)
                source_feature_CNN = source_feature
                source_feature = model_GCN(source_feature)
                source_shot_feature = source_feature[:p]
                source_shot_proto = source_shot_feature.reshape(args.shot, args.train_way,-1).mean(dim=0)
        # select same classes index from source
                tt = source_label[p:] < args.same_class_num
                source_label_common_class = np.where(tt.cpu() == True)
                source_query_feature_CNN0 = source_feature_CNN[p:]
                source_label_same_class0 = source_label[p:]
                source_query_feature_CNN = source_query_feature_CNN0[source_label_common_class]
                source_label_same_class = source_label_same_class0[source_label_common_class]

                source_query_feature = source_feature[p:]
                logits = euclidean_metric(source_query_feature, source_shot_proto)
                label = torch.arange(args.train_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)
                loss_all = F.cross_entropy(logits, label)
                loss_s = loss_all
                acc = count_acc(logits, label)

   # ============================== Select data of known classes in the target domain and perform domain adaptation with the source domain data

                target_data_all = torch.cat([target_shot_data,target_data],dim=0)
                p = args.shot*args.test_way
                target_feature = model1_select_target(target_data_all)
                target_feature = model2_select_target(target_feature)
                target_query_feature = target_feature[p:]

                target_proto = target_feature[:p]
                target_shot_proto = target_proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
                logits = euclidean_metric(target_query_feature, target_shot_proto)
                target_pred = F.softmax(logits, dim=1)
                target_pred_label = target_pred.argmax(dim=1)
                # target_label_common_class
                tt = target_pred_label < args.same_class_num
                target_label_common_class = np.where(tt.cpu() == True)
                target_label_private_class = np.where((~tt.cpu()) == True)

                # index_select_know: data with high confidence levels
                index_select_common = find_p_less_index(target_pred,args.same_class_num, 0.8)     #Select data of known classes in the target domain and perform domain adaptation with the source domain data
                index_select_pritvate = find_p_greater_index(target_pred, args.same_class_num, 0.8)

                target_pred_max = target_pred.max(dim = 1)[0]
                target_index_select = index_select_common + index_select_pritvate
                #target_index_select.sort()

                if len(target_index_select) > 0:
                    target_data_all = torch.cat([target_shot_data, target_data[target_index_select]],dim=0)
                    target_feature_CNN = model_CNN(target_data_all)  # Make predictions on target domain data
                    target_feature = model_GCN(target_feature_CNN)
                    target_proto = target_feature[:p]
                    target_shot_proto = target_proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
                    target_query_feature_CNN = target_feature_CNN[p:]
                    target_query_feature = target_feature[p:]

                    logits = euclidean_metric(target_query_feature, target_shot_proto)
                    loss_t = F.cross_entropy(logits, target_pred_label[target_index_select])
                    loss_all = loss_s + 0.1 * loss_t

                target_select_know_feature = target_query_feature[:len(index_select_common)] # easy to make mistake
                target_select_know_feature_CNN = target_query_feature_CNN[:len(index_select_common)]
                target_select_know_label = target_pred_label[index_select_common]


                len_common = min(len(target_select_know_feature_CNN),len(source_query_feature))
                if len_common > 5:
                    coral_w = compute_aifa_W(target_select_know_feature_CNN,
                                              target_select_know_label.cpu(),
                                              source_label_same_class.cpu())
                    target_pred_select_know = target_pred_max[index_select_common]
                    domain_loss_know1 = coral_wBDA(source_query_feature_CNN,
                                            source_label_same_class.cpu(),
                                            target_select_know_feature_CNN,
                                            target_select_know_label.cpu(),
                                            target_pred_select_know,
                                            coral_w)

                    coral_w = compute_aifa_W(target_select_know_feature,
                                              target_select_know_label.cpu(),
                                              source_label_same_class.cpu())
                    domain_loss_know2 = coral_wBDA(source_query_feature[source_label_common_class],
                                            source_label_same_class.cpu(),
                                            target_select_know_feature,
                                            target_select_know_label.cpu(),
                                            target_pred_select_know,
                                            coral_w)



                    loss_all = loss_all  + 1 * domain_loss_know1
                #
                # print('loss_s:',loss_s)
                # print('loss_t:',loss_t)
                # print('domain loss:', domain_loss_know1)



# ==================================To get the classification boundary through the low-density region, the entropy minimization method is used
                target_pred_p = F.softmax(logits, dim=1)
                target_pred_p = target_pred_p[:, :args.same_class_num]
                loss_entropy = EntropyLoss(target_pred_p)
                #loss_all = loss_all + 0.5 * loss_entropy


                optimizer_CNN.zero_grad()
                optimizer_GCN.zero_grad()
                loss_all.backward()
                optimizer_CNN.step()
                optimizer_GCN.step()

                target_pred_label = target_pred.argmax(dim=1)
                acc = count_acc(target_pred, target_label)

                prolog = {}
                prolog['target_pred'] = target_pred
                prolog['target_shot_label'] = target_shot_label
                torch.save(prolog,osp.join(args.save_path,'prolog'))

    #####################################################################################

            with torch.no_grad():
                model1_select_target.eval()
                model2_select_target.eval()
                model_CNN.eval()
                model_GCN.eval()

                vl = Averager()
                va = Averager()
                va_allDA = Averager()

                noDA_acc = Averager()
                allDA_acc = Averager()
                loss_temp = Averager()

                noDA_acc_every_class = Averager_vector(args.test_way2)
                allDA_acc_every_class = Averager_vector(args.test_way2)

                label_every_epoch = []
                target_pred_label_every_epoch = []

                for i, batch in enumerate(val_loader, 1):
                    data, label = [_.cuda() for _ in batch]
                    data_all = torch.cat([target_shot_data, data],dim=0)
                    p = args.shot * args.test_way

                    target_feature = model_CNN(data_all)
                    target_feature = model_GCN(target_feature)

                    target_query_feature = target_feature[p:]
                    target_shot_proto = target_feature[:p]
                    target_shot_proto = target_shot_proto.reshape(args.shot,args.test_way2,-1).mean(dim=0)
                    logits = euclidean_metric(target_query_feature, target_shot_proto)
                    target_pred = F.softmax(logits,dim=1)
                    target_pred_label = target_pred.argmax(dim = 1)
                    loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)
                    acc_every_class = cal_acc2(label.cpu(), target_pred_label.cpu(), args.test_way2)
                    allDA_acc.add(acc)
                    allDA_acc_every_class.add(acc_every_class)

                    ave_acc.add(acc)

                    vl.add(loss.item())
                    va.add(acc)
                    loss_temp.add(loss)

                    label_every_epoch.append(label)
                    target_pred_label_every_epoch.append(target_pred_label)

                    va_allDA.add(acc)

                    proto = None;
                    logits = None;
                    loss = None

                print('all DA acc:', allDA_acc.item())
                print(allDA_acc_every_class.item())

                va_allDA = va_allDA.item()
                vl = vl.item()
                va = va.item()
                print('val acc:{:.2f}'.format(va*100))

                if va_allDA > trlog_allDA['max_acc']:
                    trlog_allDA['max_acc'] = va_allDA
                    pred_class_temp_allDA = allDA_acc_every_class.item()

                if va > trlog['max_acc']:
                    trlog['max_acc'] = va
                    va_temp = va
                    save_model('proto-max-acc',ii)
                    pred_class_temp = allDA_acc_every_class.item()

                    label_temp = label_every_epoch[0]
                    for i in range(1,len(label_every_epoch)):
                        label_temp = torch.cat([label_temp, label_every_epoch[i]])

                    target_pred_label_temp = target_pred_label_every_epoch[0]
                    for i in range(1,len(target_pred_label_every_epoch)):
                        target_pred_label_temp = torch.cat([target_pred_label_temp, target_pred_label_every_epoch[i]])

                trlog['train_loss'].append(tl)
                trlog['train_acc'].append(ta)
                trlog['val_loss'].append(vl)
                trlog['val_acc'].append(va)

                save_model('epoch-last',ii)

                if epoch % args.save_epoch == 0:
                    save_model('epoch-{}'.format(epoch),ii)
                iteration.append(epoch)
                Loss.append(loss_temp.item().cpu())
                Acc.append(round(va_allDA*100,2))

        print('ACC:',Acc)

        trlog['acc_per_class'].append(pred_class_temp)
        trlog['mean_acc'].append(va_temp)
        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        vv.add(pred_class_temp)
        #print('vv:',vv.item())
        label_rem.append(label_temp)
        #print('label_rem:',label_rem)
        pred_label_rem.append(target_pred_label_temp)
        #print('pred_label_rem:',pred_label_rem)
        vv_allDA.add(pred_class_temp_allDA)

pred_per_class = vv_allDA.item()
pred_average = torch.mean(pred_per_class)
print('all DA:', vv_allDA.item())
print('all DA:',pred_average)

pred_per_class = vv.item()
pred_average = torch.mean(pred_per_class)
print(vv.item())
print(pred_average)
torch.save(label_rem,osp.join(args.save_path,'label_rem.pt'))
torch.save(pred_label_rem,osp.join(args.save_path,'pred_label_rem.pt'))
torch.save(vv.item(),osp.join(args.save_path,'class_per_acc.pt'))
torch.save(pred_average,osp.join(args.save_path,'mean_acc.pt'))