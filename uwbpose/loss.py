# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        # 
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        #print(heatmaps_gt)
        for idx in range(num_joints):
            # batch x (w*h)
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            
            target_weight = (torch.max(heatmap_gt, dim=1)[0] > 0).float().reshape(-1, 1)
            #print("red", heatmap_pred)
            #print(heatmap_gt) 
            #loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            loss += 0.5 * self.criterion(heatmap_pred*target_weight, heatmap_gt)
        return loss / num_joints
        
def dual_loss(output, target):
    loss = 0
    use_loss = 0
    with torch.no_grad():
        batch_size = output.size(0)
        label_size = output.size(1)
        pred = output.reshape((batch_size, -1))
        for i in range(batch_size):
            #temp_loss = torch.tensor([1])
            temp_loss = 1
            temp_target = target[i]
            if pred[i][temp_target] > 0 :
                for j in range(label_size):
                    if j == temp_target or pred[i][j]<=0 :
                        continue
                    else :
                        if pred[i][temp_target] / (pred[i][temp_target] + pred[i][j]) < temp_loss :
                            temp_loss = pred[i][temp_target] / (pred[i][temp_target] + pred[i][j])
            temp_loss = 1 - temp_loss
            if temp_loss >= 0.005:
                loss += temp_loss
                
        loss = loss / batch_size
        if loss != 0:
            use_loss += 1
        
    return loss, use_loss
'''
        _, predicted = torch.max(output, 1)
        for i in range(len(output[0])):
            print(i)
'''
        

if __name__ == '__main__':
    a = JointsMSELoss()
    b = torch.arange(24).reshape(2, 3, 2, 2)
    print(b)
    c = torch.zeros((2, 3, 2, 2))
    c[0][0][0][0] = 1
    d = a(b, c)
    print(d)
