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

if __name__ == '__main__':
    a = JointsMSELoss()
    b = torch.arange(24).reshape(2, 3, 2, 2)
    print(b)
    c = torch.zeros((2, 3, 2, 2))
    c[0][0][0][0] = 1
    d = a(b, c)
    print(d)
