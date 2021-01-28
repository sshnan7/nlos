# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import numpy as np
import math

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
    
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    # max값을 찾기 위해서 2차원을 1차원으로 쭉 펼침.
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    # 최고값 index, 값 찾음
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    # batch , joint마다 한개씩 형태로 array 변형
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # (batch, joints, 2) 형태로 바꿈. (idx가 중복되게)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # (batch ,joints) 마다 x, y 두개의 prediction. 행 열이 아닌 x, y
    # cv2에서도 x,y로 그려주면 됨. ( 그대로 좌표 이용 )
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # np.greater -> 최대값이 0보다 작거나 같으면 False. 크면 True
    # tile 이용해서 (batch, joint, 2) 로 preds와 마찬가지로 만들어줌.
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32) # TF에서 1, 0 으로

    # MASK 사용, 즉 최댓값이 0이하인 좌표는 그냥 0,0으로 처리. 
    # 이를 이용하면 pred 안된 것으로 구분 가능할듯.
    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    TEST_POST_PROCESS = True
    if TEST_POST_PROCESS is True:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()
    # transform 안하니까 필요 x  
    # Transform back
    #for i in range(coords.shape[0]):
    #    preds[i] = transform_preds(coords[i], center[i], scale[i],
    #                               [heatmap_width, heatmap_height])

    return preds, maxvals


if __name__ == "__main__":
    sigma = 5
    tmp_size = sigma * 10
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2  # width, height of gaussian points

    # gaussian
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    d = np.zeros((1, 3, 256, 192), np.float32)
    for i in range(1, 4):
        d[0][i-1][30*i:30*i+g.shape[0], 30*i:30*i+g.shape[1]] = g
    get_max_preds(d)
