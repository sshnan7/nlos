from alex_net import AlexNet
from mean_squared_error import mean_squared_error

import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
from scipy.io import loadmat
import glob
import matplotlib.pyplot as plt

class LspDataset(Dataset):
    def __init__(self, gt, root_dir, mode='train', test_ratio=1):

        print("mode : ", mode)
        self.mode = mode
        target = loadmat(gt)['joints'] # 3 x 14 x 2000
        self.root_dir = root_dir     
        # x, y, visiblity ( 3 )   x  joints ( 14 ) x #lsp_dataset
        self.target = target.transpose(2, 1, 0)  # 10000 * 3 * 14
        joints = self.target 
        joints[:, :, 2] = np.logical_not(self.target[:, :, 2]).astype(int)
        self.size_list = []

        print('before ', target.shape, 'after', self.target.shape)
        train_data = []
        crop_size = 256
        transform = transforms.Compose([
            #transforms.Resize((crop_size, crop_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #transforms.Normalize(mean, std),
        ])
        for f in glob.glob(root_dir+'/*.jpg'):
            im = Image.open(f)
            self.size_list.append(crop_size-np.array(im.size))
            #self.size_list.append(crop_size/np.array(im.size))
            #print("im : ", im)
            im = transform(im)
            train_data.append(im)

        self.size_list = np.array(self.size_list)
        for i in range(len(self.size_list)):
            self.target[i][:, :2] += self.size_list[i]/2
            #self.target[i][:, :2] *= self.size_list[i]

        self.train_data = train_data
        self.test_len = int(len(train_data)*test_ratio)
        #print(len(self.train_data))
        #print(self.train_data[0].shape)
        #print(self.train_data[0])
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)-self.test_len
        else:
            return self.test_len

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.train_data[idx], self.target[idx]
        else:
            start = len(self.train_data) - self.test_len
            return self.train_data[start+idx], self.target[start+idx]

def load_dataset(sample):
    images = []
    poses = []
    visibilities = []
    
    for i in range(len(sample[0])):
        img = sample[0][i]
        label = sample[1][i]
        #print("img : ", img.size())
        #print("label : ", label.size())
    images.append(img)
    x = label
    #x = x.view(-1, 3)
    pose = x[:, :2]
    visibility = x[:, 2].clone().view(-1, 1).expand_as(pose)
    #print(pose)
    #print(visibility)
    poses.append(pose)
    visibilities.append(visibility)
    
    return images, poses, visibilities


def get_optimizer(opt, model):
    if opt == 'MomentumSGD':
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.1)
    elif opt == "Adam":
        optimizer = optim.Adam(model.parameters())
    return optimizer

def show_result(sample_batched, output):
    img, label = sample_batched[0], sample_batched[1]
    batch_size = len(img)
    print(img.size(), label.size(), output.size())
    im_size = img.size(2)
    #label = label.transpose(2, 1)
    grid_border_size = 2
    print(img.size(), label.size(), output.size())
    print("output = {} \n{}\n label = {}\n {}\n".format(output.size(), output, label.size(), label))
    grid = utils.make_grid(img)
    plt.imshow(grid.numpy().transpose(1,2,0))

    for i in range(batch_size):
        plt.scatter(label[i, :, 0].numpy() + i * im_size + (i+1) * grid_border_size, 
                        label[i,:,1].numpy() + grid_border_size, s=20, marker ='.', c='r')
        plt.scatter(output[i, :, 0].detach().numpy() + i * im_size + (i+1) * grid_border_size, 
                       output[i,:,1].detach().numpy() + grid_border_size, s=20, marker ='.', c='b')
        plt.title('Batch from dataloader')


if __name__ == '__main__':
    
    PATH = './weights/'
    result_path = './results/'
    print("save_path = ", PATH )

    #model
    model = AlexNet(14)
    optimizer = get_optimizer('Adam', model)
    criterion = nn.MSELoss()
    running_loss=0.0

    #train
    #images, poses, visibilities = load_dataset(path)
    test_data = LspDataset('./lsp_dataset/joints.mat', './lsp_dataset/images', mode='test')
    dataloader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=4)
    
    model.load_state_dict(torch.load(PATH +'deeppose_state_dict.pt')) 
    model.eval()
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched[0].size(), sample_batched[1].size())  
        # torch.Size([1,3,202, 202]) torch.Size([1, 3, 14])
        img = sample_batched[0]
        label = sample_batched[1]
        
        image = img
        pose = label[:, :, :2].type(torch.FloatTensor)
        visibility = label[:, :, 2]
        #print(image[0][0][0])

        output = model(image)
        if i_batch % 10 == 0 :
            #print("image : {}".format(image)
            plt.figure()
            show_result(sample_batched, output)
            plt.axis('off')
            plt.ioff()
            file_name = "result" + str(i_batch) + ".png"
            plt.savefig(result_path + file_name) 

        loss = criterion(output, pose)
        running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' %
                  (0 , i_batch + 1, loss.item()))

        '''
        if i_batch % 100== 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (0 , i_batch + 1, running_loss / 2000))
            running_loss = 0.0
        '''
