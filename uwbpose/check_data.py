
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import arguments
from pose_dataset import *
import time

args = arguments.get_arguments()

train_data = PoseDataset(mode='train', args=args)
batch = 250
train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=False, num_workers=16, pin_memory=True)
print(len(train_dataloader)*batch)
li = torch.zeros((len(train_dataloader)*batch, 1948, 3, 3))
print(li.shape)
index = 0
for rf, target_heatmap in tqdm(train_dataloader):
    #rf = rf.reshape(-1, 3, 3)
    li[index*batch:(index+1)*batch] = rf
    index+=1
    #print('mean', torch.mean(rf), 'var', torch.var(rf))
    #print('min', torch.min(rf), 'max', torch.max(rf))
    #print('squre mean', torch.mean(torch.mul(rf, rf)))
    if index > 150 and index < 180:
        time.sleep(0.1)
    #print(index, index*100, (index+1)*100)
print(li.shape)
print('mean', torch.mean(li), 'var', torch.var(li))
print('min', torch.min(li), 'max', torch.max(li))
print('squre mean', torch.mean(torch.mul(li, li)))
