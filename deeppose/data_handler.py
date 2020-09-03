
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from scipy import io
import glob
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

class LspDataset(Dataset):

    def __init__(self, gt, root_dir, mode='train', test_ratio=0.25):

        print("mode : ", mode)
        self.mode = mode
        mat_file = io.loadmat(gt)
        self.root_dir = root_dir
        print(mat_file)
        # x, y, visiblity ( 3 )   x  joints ( 14 ) x #lsp_dataset
        target = mat_file['joints'] # 3 x 14 x 2000
        
        self.target = target.transpose(2, 1, 0)  # 10000 * 3 * 14
        self.target = self.target.transpose(0, 2, 1)  # 10000 * 14 * 3
        self.size_list = []
        print('before ', target.shape, 'after', self.target.shape)
        train_data = []
        crop_size = 202
        transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),    # resize
            #transforms.CenterCrop(crop_size),            # crop 
            transforms.ToTensor()
            #transforms.Normalize(mean, std),
        ])
        for f in glob.glob(root_dir+'/*.jpg'):
            im = Image.open(f)
            #self.size_list.append(crop_size-np.array(im.size))      # crop
            self.size_list.append(crop_size/np.array(im.size))       # resize
            im = transform(im)
            train_data.append(im)

        self.size_list = np.array(self.size_list)
        for i in range(len(self.size_list)):
            #self.target[i][:, :2] += self.size_list[i]/2    # crop
            self.target[i][:, :2] *= self.size_list[i]     # resize
            
            
        self.train_data = train_data
        self.test_len = int(len(train_data)*test_ratio)
        print(len(self.train_data))
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

def show_label(sample_batched):
    img, label = sample_batched[0], sample_batched[1]
    batch_size = len(img)
    print(img.size(), label.size())
    im_size = img.size(2)
    #label = label.transpose(2, 1)
    grid_border_size = 2
    print(img.size(), label.size())
    print(label)
    grid = utils.make_grid(img)
    plt.imshow(grid.numpy().transpose(1,2,0))

    for i in range(batch_size):
        plt.scatter(label[i, :, 0].numpy() + i * im_size + (i+1) * grid_border_size, 
                        label[i,:,1].numpy() + grid_border_size, s=20, marker ='.', c='r')
        plt.title('Batch from dataloader')


if __name__ == '__main__':
    
    result_path = './data_handle/'
    train_data = LspDataset('./lspet/joints.mat', './lspet/images', mode='train')
    #train_data = LspDataset('./lsp_dataset/joints.mat', './lsp_dataset/images', mode='train')
    #test_data = LspDataset('./lsp_dataset/joints.mat', './lsp_dataset/images', mode='test')

    dataloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)
    radius = 3
    color = [(np.random.randint(256)/255, np.random.randint(256)/255, np.random.randint(256)/255) for i in range(14)]
    thickness = -1
    #print('training data size', len(train_data), 'test data size', len(test_data))

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched[0].size(), sample_batched[1].size())  # torch.Size([1,3,202, 202]) torch.Size([1, 3, 14])
        img = sample_batched[0]
        label = sample_batched[1]
        
        print(img)
        pose = label[:, :, :2].type(torch.FloatTensor)
        print(pose)
        visiblity = label[:, :, 2].type(torch.FloatTensor)
        print(visiblity.size())
        print(visiblity) 
        visiblity = visiblity.transpose(1,0)
        print(visiblity.size())
        print(visiblity) 
        after_pose = pose.mul(visiblity)
        print(after_pose)

        batch_size, c, w, h = img.size()
        print("img info(batch_size, c, w, h) - {} {} {} {}".format(batch_size, c, w, h ))
        if i_batch == 3:
            plt.figure()
            show_label(sample_batched)
            plt.axis('off')
            plt.ioff()
            file_name = "result" + str(i_batch) + ".png"
            plt.savefig(result_path + file_name) 
            break
    '''     
    #print('training data size', len(train_data), 'test data size', len(test_data))
    for img, label in dataloader:
        #print(img)
        #print(label)
        print(img.dtype, label.dtype, img.shape, label.shape)
        _, c, w, h = img.shape
        d = img[0].numpy().reshape(c, w, h).transpose(1, 2, 0)
        d = cv2.cvtColor(d, cv2.COLOR_RGB2BGR)

        coordinate = label.reshape(14, 3)
        for i in range(len(coordinate)):
            z = cv2.circle(d, (coordinate[i][0], coordinate[i][1]), radius, color[i], thickness)
        cv2.imshow('this', z)
        #cv2.imshow('resize', zoom1)
        cv2.waitKey(0)

        #cv2.destroyAllWindows()
    '''
