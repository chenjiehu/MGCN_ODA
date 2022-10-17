#split the shot file and query file
import pandas as pd
import random, csv
import numpy as np
import torch
import os
from utils import shot, Task_data, filename_shot, filename_query, ensure_path
from Dataset.Dataset import SOURCE_DATA
#
def Split_shot_query(filename,file_shot,file_query,shot, num_split = 10):

    filename = filename
    for loop in range(0,num_split):
        filename_shot = os.path.join(file_shot, str(shot)+'_shot'+ str(loop) + '.csv')
        filename_query = os.path.join(file_query, str(shot)+'_query'+ str(loop) + '.csv')
        if not os.path.exists(filename):
            print('csv file dose not exist!')
        else:
            images, labels = [], []
            with open(filename) as f:
                reader = csv.reader(f)
                for row in reader:
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    img, label = row
                    label = int(label)

                    images.append(img)
                    labels.append(label)
                labels = np.array(labels)  # ??
                m_ind = []
                for i in range(max(labels) + 1):  # ??
                    ind = np.argwhere(labels == i).reshape(-1)
                    ind = torch.from_numpy(ind)
                    m_ind.append(ind)
                shot_stack = []
                query_stack = []
                for c in range(max(labels) + 1):
                    l = m_ind[c]
                    # pos1 = torch.randperm(len(l))[:shot]    #pos = torch.randperm(len(l))[:self.n_per]
                    # pos2 = torch.randperm(len(l))[shot:]
                    pos_temp = torch.randperm(len(l))   #pos = torch.randperm(len(l))[:self.n_per]
                    pos1 = pos_temp[:shot]
                    pos2 = pos_temp[shot:]
                    shot_stack.append(l[pos1])        #batch.append(l[pos])
                    query_stack.append(l[pos2])       # batch = torch.stack(batch).t().reshape(-1)
                shot_stack = torch.stack(shot_stack).t().reshape(-1)
                #query_stack = torch.stack(query_stack).t().reshape(-1)
                temp = query_stack[0].t()
                for i in range(1,len(query_stack)):
                    temp = torch.cat([temp, query_stack[i]])
                query_stack = temp
        with open(filename_shot, mode='w', newline='') as f:
            writer = csv.writer(f)
            temp = shot_stack.detach().numpy()
            temp = temp.astype(np.int32)
            for i in range(0,len(temp)):
                writer.writerow([images[temp[i]], labels[temp[i]]])
            print('writen shot csv file:', filename)
        with open(filename_query, mode='w', newline='') as f:
            writer = csv.writer(f)
            temp = query_stack.detach().numpy()
            temp = temp.astype(np.int32)
            for i in range(0,len(temp)):
                writer.writerow([images[temp[i]], labels[temp[i]]])
            print('writen query csv file:', filename)

if __name__=='__main__':
    SOURCE_DATA(os.path.join(Task_data))
    ensure_path(filename_shot)
    ensure_path(filename_query)
    Split_shot_query(os.path.join(Task_data,'source.csv'),filename_shot,filename_query,shot)
    pass
