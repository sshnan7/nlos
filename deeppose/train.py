from alex_net import AlexNet
import resnet
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
    def __init__(self, gt, root_dir, mode='train', test_ratio=0):

        # lspet data load
        print("mode : ", mode)
        self.mode = mode
        target = loadmat(gt)['joints'] # 3 x 14 x 1000
        self.root_dir = root_dir     
        # x, y, visiblity ( 3 )   x  joints ( 14 ) x #lsp_dataset
        self.target = target.transpose(2, 1, 0)  # 10000 * 3 * 14
        self.target = self.target.transpose(0, 2, 1)  # 10000 * 14 * 3
        self.size_list = []

        print('before ', target.shape, 'after', self.target.shape) # before (14, 3, 10000)  after (10000, 14, 3)
        train_data = []
        crop_size = 256

        transform = transforms.Compose([
            #transforms.Resize((crop_size, crop_size)),          # resize
            transforms.CenterCrop(crop_size),                    # crop
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #transforms.Normalize(mean, std),
        ])
        for f in glob.glob(root_dir+'/*.jpg'):
            im = Image.open(f)
            self.size_list.append(crop_size-np.array(im.size))      # crop
            #self.size_list.append(crop_size/np.array(im.size))     # resize
            #print("im : ", im)
            im = transform(im)
            train_data.append(im)

        self.size_list = np.array(self.size_list)
        for i in range(len(self.size_list)):
            self.target[i][:, :2] += self.size_list[i]/2            # crop
            #self.target[i][:, :2] *= self.size_list[i]             # resize

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


# kepoint를 image 위에 찍어 주는 함수
def show_result(sample_batched, output):
    img, label = sample_batched[0], sample_batched[1]
    batch_size = len(img)
    print(img.size(), label.size(), output.size())
    im_size = img.size(2)
    #label = label.transpose(2, 1)
    grid_border_size = 2
    print(img.size(), label.size(), output.size())
    print(label)
    print(output)
    grid = utils.make_grid(img)
    plt.imshow(grid.numpy().transpose(1,2,0))

    for i in range(batch_size):
        plt.scatter(label[i, :, 0].numpy() + i * im_size + (i+1) * grid_border_size, 
                        label[i,:,1].numpy() + grid_border_size, s=20, marker ='.', c='r')
        plt.scatter(output[i, :, 0].detach().numpy() + i * im_size + (i+1) * grid_border_size, 
                       output[i,:,1].detach().numpy() + grid_border_size, s=20, marker ='.', c='b')
        plt.title('Batch from dataloader')


def get_optimizer(opt, model):
    if opt == 'MomentumSGD':
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.1)
    elif opt == "Adam":
        optimizer = optim.Adam(model.parameters())
    return optimizer


if __name__ == '__main__':
    
    PATH = './weights/'
    print("save_path = ", PATH )

    #model
    model = AlexNet(14)
    #model = resnet.ResNet50()
    optimizer = get_optimizer('Adam', model)
    criterion = nn.MSELoss()
    running_loss=0.0

    #train

    train_data = LspDataset('./lspet/joints.mat', './lspet/images', mode='train')
    #train_data = LspDataset('./lsp_dataset/joints.mat', './lsp_dataset/images', mode='train')
    #test_data = LspDataset('./lsp_dataset/joints.mat', './lsp_dataset/images', mode='test')

    batch = 4
    ### training part 
    #model.load_state_dict(torch.load(PATH +'resnet_state_dict.pt'))  # 불러오기

    # DataLoader 
    #  pytorch 내장 함수, dataset으로 부터 ground trouth data ( img, label ) 을 불러와줌
    dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)  #
    model.train()
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched[0].size(), sample_batched[1].size())  
        # torch.Size([1,3,202, 202]) torch.Size([1, 3, 14])
        img = sample_batched[0]  
        label = sample_batched[1]
        

        # 학습을 위해 array 형을 맞춰줌
        image = img
        pose = label[:, :, :2].type(torch.FloatTensor) 
        visibility = label[:, :, 2].type(torch.FloatTensor)
        visibility = visibility.unsqueeze(2)  # batch_size * 14 * 1
        #print("########", visibility.size())

        #print(image[0][0][0])
        optimizer.zero_grad()
        output = model(image)   # network model 에 image 입력 후 나온 output
        pose = pose.mul(visibility)
        output = output.mul(visibility)

        # progress 출력 -> 결과 image를 저장함.
        if i_batch % 10 ==0 :
            #print("image : {}".format(image))
            print("visibility = {} \n {}\n output = {} \n{}\n pose = {}\n {}\n".format(visibility.size(), visibility, output.size(), output, pose.size(), pose))
            if i_batch > 300:
                plt.figure()
                show_result(sample_batched, output)
                plt.axis('off')
                plt.ioff()
                file_name = "train result2_" + str(i_batch) + ".png"
                plt.savefig('./results/' + file_name) 
      
        loss = criterion(output, pose)  # network output과 image를 비교 (mean squared error)
        # loss 를 바탕으로 network를 업데이트
        loss.backward()   
        optimizer.step()
        running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' %
                  (0 , i_batch + 1, running_loss / (i_batch + 1)))

        '''
        if i_batch % 100== 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (0 , i_batch + 1, running_loss / 2000))
            running_loss = 0.0
        '''
    
    torch.save(model.state_dict(), PATH +'alexnet_state_dict2.pt') # network 결과 저장
    

    

