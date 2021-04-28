# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


import numpy as np
from inference import get_max_preds
import torch
import torch.nn as nn


def MSELoss(output, target):

    loss = ((output - target) ** 2).mean()
    return loss


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    # (joint, batch)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                # 두 좌표를 heatmap 크기 고려해서 normalize
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                # l2 distance
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    # -1이 아닌 것만 True. ( -1은 좌표 0, 0 --> 검출x )
    dist_cal = np.not_equal(dists, -1)
    # -1이 아닌 개수 
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        # -1이 아닌 것들 중에 thr 보다 낮은 것들의 개수  / 전체 
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    # 원래 heatmap 크기의 1/10 의 값이 normalize 값
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0: # dist_acc가 -1 이 아닌것만 
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def human_accuracy(output, target):
    i = list(target.size())
    with torch.no_grad():
        _, predicted = torch.max(output, 1) # _는 label개수 중에서 최대인 확률의 값, predicted는 그 label의 index
        acc = 0
        avg_acc = 0
        cnt = 0
        acc += (predicted == target).sum().item()
        cnt += i[0]
        avg_acc = acc/cnt
        return avg_acc, cnt

def dis_accuracy(output, target):
    i = list(target.size())
    with torch.no_grad():
        _, predicted = torch.max(output, 1) # _는 label개수 중에서 최대인 확률의 값, predicted는 그 label의 index
        acc = 0
        avg_acc = 0
        cnt = 0
        for k in range(_.shape[0]):
            if _[k] >= 0.5:
                _[k] = 1
            else :
                _[k] = 0
            #print("output", "target", _[k], target[k][0])
            if _[k] == target[k][0]:
                acc +=1
        #_ = _.unsqueeze(1)
        #print("output", _)
        #print("target", target)
        #acc += (_ == target).sum().item()
        #print("acc", acc)
        cnt += i[0]
        avg_acc = acc/cnt
        return avg_acc, cnt
    
def pck(pred, target):
    keypoints = [ "padding", "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle" ]

    num_img = pred.shape[0]
    num_joint = pred.shape[1]
    true_detect = np.zeros((4, num_joint))
    whole_count = np.zeros((4, num_joint))
    thr = [0.1, 0.2, 0.3, 0.5]
    for i in range(num_img):
        # left shoulder + right_hip,   or   right shoulder + left hip
        #torso = 0
        check = 0
        
        if target[i][2][0] >= 1 and target[i][2][1] >= 1 and target[i][7][0] >= 1 and target[i][7][1] >= 1: # right shoulder + left hip
            check = 1
            torso = np.linalg.norm(target[i][2] - target[i][7])
        
        if check == 0 and target[i][1][0] >= 1 and target[i][1][1] >= 1 and target[i][8][0] >= 1 and target[i][8][1] >= 1: # l_shoulder + r_hip
            check = 1
            torso = np.linalg.norm(target[i][1] - target[i][8])

        if check == 0: # torso diameter 못 구하는 데이터는 스킵.
            continue

        for j in range(num_joint):
            if target[i][j][0] < 1 and target[i][j][1] < 1: # invisible
                continue
            
            dist = np.linalg.norm(target[i][j] - pred[i][j])
            for t in range(len(thr)):
                whole_count[t][j] += 1
                if dist <= thr[t] * torso:
                    true_detect[t][j] += 1

    return true_detect, whole_count
            
    
