
import argparse
import os.path as osp
import time
import torch
import torch.nn.functional as F
import xlsxwriter
from torch.utils.data import DataLoader
from Dataset.Dataset import SOURCE_DATA, TARGET_DATA
from samplers import CategoriesSampler
from model.convert import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, save_path_step1, Auxiliary_data, Val_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=60)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=12)
    parser.add_argument('--test-way', type=int, default=19)
    parser.add_argument('--save-path', default=save_path_step1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
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
    model = Convnet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    for epoch in range(1, args.max_epoch + 1):

        def save_model(name):
            torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

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

        model.train()

        start_time = time.time()

        tl = Averager()
        ta = Averager()

        for i, source_batch in enumerate(train_loader):
            data, source_label = [_.cuda() for _ in source_batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            query_feature = model(data_query)

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

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            proto = None;
            logits = None;
            loss = None

        end_time = time.time()
        print(end_time-start_time)
        tl = tl.item()
        ta = ta.item()

        model.eval()

        with torch.no_grad():
            vl = Averager()
            va = Averager()

            for i, batch in enumerate(val_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]

                proto = model(data_shot)
                proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)

                logits = euclidean_metric(model(data_query), proto)
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