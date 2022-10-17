import argparse
import os.path as osp
import time
import os
import torch
import torch.nn.functional as F
import xlsxwriter
from torch.utils.data import DataLoader

from Dataset.Dataset import SOURCE_DATA, TARGET_DATA
from samplers import CategoriesSampler
from model.Multi_gcn import MultiGCN_relation
from model.convert import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric,save_path_step1, save_path_step2, Auxiliary_data, Val_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=60)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=12)
    parser.add_argument('--test-way', type=int, default=19)
    parser.add_argument('--load', default=save_path_step1)
    parser.add_argument('--save-path', default=save_path_step2)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--model', default='Conv4',
                        help='model: Conv{4|6} / ResNet{10|18|34|50|101}')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    torch.manual_seed(3)
    ensure_path(args.save_path)

    trainset = SOURCE_DATA(Auxiliary_data)
    train_sampler = CategoriesSampler(trainset.label, 50,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    valset = TARGET_DATA(Val_data)
    val_sampler = CategoriesSampler(valset.label, 50,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    model_CNN = Convnet().cuda()
    model_GCN = MultiGCN_relation(input_dim=1600, N_way=args.test_way).cuda()

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
        lr_scheduler.step()

        model_CNN.train()
        model_GCN.train()

        start_time = time.time()

        tl = Averager()
        ta = Averager()

        for i, source_batch in enumerate(train_loader):
            data, source_label = [_.cuda() for _ in source_batch]
            p = args.shot * args.train_way

            source_feature = model_CNN(data)
            source_feature = model_GCN(source_feature)
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
