import os
import shutil
import time
import pprint

import torch
import torch.nn.functional as F

# data_split

filename_shot = './Data/shot_OR'   # shot folder name
filename_query = './Data/query_OR'  # query folder name
shot = 5
save_path_step1 = './save/OR_step1/5shot'  # save path setp1
save_path_step2 = './save/OR_step2/5shot'   # save path step2
save_path_DA = './save/OR_step_DA/5shot'   #save path DA

Auxiliary_data = './Data/OR_12wayOPT'
Val_data = './Data/OR_19wayOPT_val'
Task_data = './Data/OR_12wayRSI'

NUM_shot = 20
Common_class_number = 6  # the number of common class 
# train

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def delete_path(path):
    shutil.rmtree(path)
    os.makedirs(path)



class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Averager_vector():

    def __init__(self,num):
        self.n = 0
        self.v = torch.zeros(num)

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Averager_matrix():

    def __init__(self,M, N):
        self.n = 0
        self.v = torch.zeros(M,N)

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def cos_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    ab = torch.mul(a,b)
    ab = torch.sum(ab, dim=2)
    a_norm = torch.norm(a,dim=2)
    b_norm = torch.norm(b,dim=2)
    ab_norm = torch.mul(a_norm,b_norm)
    logits = ab/ab_norm
    return logits


def K_euclidean_metric(a, b, k, shot):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    #logits_e = torch.exp(logits/100)
    logits_e = logits
    logits_zeros = torch.zeros_like(logits_e)
    _, index = torch.topk(logits, k=k, dim=1, largest=True, sorted=False)
    for num in range(logits_zeros.size(0)):
        logits_zeros[num, index[num,:]] = 1
    logits = torch.mul(logits_e, logits_zeros)

    logits2 = logits.reshape(n,shot,-1).sum(dim=1)
    return logits2


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2


from sklearn.metrics import accuracy_score

NUM_CLASSES = 11

def cal_acc(gt_list, predict_list, num):
    acc_sum = 0
    for n in range(num):
        y = []
        pred_y = []
        for i in range(len(gt_list)):
            gt = gt_list[i]
            predict = predict_list[i]
            if gt == n:
                y.append(gt)
                pred_y.append(predict)
        print ('{}: {:4f}'.format(n if n != (num - 1) else 'Unk', accuracy_score(y, pred_y)))
        if n == (num - 1):
            print ('Known Avg Acc: {:4f}'.format(acc_sum / (num - 1)))
        acc_sum += accuracy_score(y, pred_y)
    print ('Avg Acc: {:4f}'.format(acc_sum / num))
    print ('Overall Acc : {:4f}'.format(accuracy_score(gt_list, predict_list)))


def cal_acc2(gt_list, predict_list, num):
    acc_sum = 0
    class_pred = torch.zeros(num)
    for n in range(num):
        y = []
        pred_y = []
        for i in range(len(gt_list)):
            gt = gt_list[i]
            predict = predict_list[i]
            if gt == n:
                y.append(gt)
                pred_y.append(predict)
        acc = accuracy_score(y, pred_y)
        # print ('{}: {:4f}'.format(n if n != (num - 1) else 'Unk', acc))
        # if n == (num - 1):
        #     print ('Known Avg Acc: {:4f}'.format(acc_sum / (num - 1)))
        class_pred[n] = acc
    # print ('Avg Acc: {:4f}'.format(acc_sum / num))
    return class_pred

def find_index(a, b):
    a_len = len(a)
    index = []
    for i in range(a_len):
        if a[i] < b:
            index.append(i)
    return index

def find_p_less_index(pred, label_eta, pred_eta):
    pred_label = pred.argmax(dim = 1)
    pred_len = len(pred)
    index = []
    for i in range(pred_len):
        position = pred_label[i]
        if pred[i,position] > pred_eta and position < label_eta:
            index.append(i)
    return index



#Find the subscript in pred is greater than or equal to label_eta
def find_p_greater_index(pred, label_eta, pred_eta):
    pred_label = pred.argmax(dim = 1)
    pred_len = len(pred)
    index = []
    for i in range(pred_len):
        position = pred_label[i]
        if pred[i,position] > pred_eta and position >= label_eta:
            index.append(i)
    return index

#input: the probility of softmax, C*N
#       label_eta: class boundary between common classes and private classes
#output: the index of common classes and private classes
def find_know_and_unknow(pred, label_eta):
    index_common = []
    index_peivate = []
    pred_label = pred.argmax(dim = 1)
    pred_len = len(pred)
    for i in range(pred_len):
        if pred_label[i] < label_eta:
            index_common.append(i)
        else:
            index_peivate.append(i)
    return index_common, index_peivate

if __name__=='__main__':
    output = torch.randn(10,6)
    pred = F.softmax(output,dim=1)
    print(pred)
    index_common, index_private = find_know_and_unknow(pred, 3)
    print(index_common)
    print(index_private)
