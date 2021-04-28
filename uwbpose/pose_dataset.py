import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn
import os 
import glob
import numpy as np
import time
import cv2
import random
from matplotlib import pyplot as plt


class PoseDataset(Dataset):
    def __init__(self, args, is_correlation=False, mode='train'):
        
        '''
        dataset 처리
        rf와 이미지의 경우에는 init 할 때부터 읽어와서 메모리에 올리지만 gt는 데이터를 활용할 때마다 load함.
        mode - train : 학습을 위함.  rf, gt, img 다 있는 경우
                test : test를 위함. rf, gt, img 다 있는 경우 
                valid: valid를 위함(demo). rf, img만 있는 경우
        '''
        self.is_correlation = is_correlation
        self.load_img = args.vis
        self.mode = mode
        
        self.is_gaussian = args.gaussian
        self.std = 0.1
        self.mean = 0
        
        self.is_normalize = args.normalize
        self.cutoff = args.cutoff
        
        self.augmentation = args.augment
        self.augmentation_prob = 1
        self.intensity = Intensity(scale=0.05)

        self.flatten = args.flatten
        self.arch = args.arch
        if self.arch =='hrnet':
            self.input_size= 128
        else:
            self.input_size = 120

        data_path = '/data/nlos/save_data_ver2'
        data_path2 = '../../gt'
        #data_path_list = os.listdir(data_path)
        data_path_list = glob.glob(data_path + '/*')
        data_path_list2 = glob.glob(data_path2 + '/*')
        #print("data list", data_path_list)
        data_path_list = sorted(data_path_list)
        data_path_list2 = sorted(data_path_list2)
        #print(data_path_list)
        rf_data = []  # rf data list
        new_rf_data = [] # rf data list non space
        gt_list = []  # ground truth
        img_list = []
        human_label_list = [] #by 승환
        human_label_list_read = []
        real_human_label_list = []
        antena_label_list = []
        human_num_list = [] #by 승환
        print("start - data read")
        #test_dir = [8, 9] # past version - 1
        test_dir = [2, 5, 10, 14, 16, 19] # 승환 수정
        los_dir = [0, 6, 8, 10, 13] #10 ->25 -> 14
        nlos_dir = [5, 12, 15, 17, 18]
        #test_dir = [2, 5, 10] # los
        #test_dir = [14, 16, 19]  #nlos
        #test_dir = [10, 19] # demo - with mask  ,  los , nlos
        remove_dir = [3, 4] 
        #valid_dir = [25, 26, 27]
        #valid_dir = [21]
        # valid_dir = [28, 29] # nlos wall
        valid_dir = [x for x in range(22, 40)]
        valid_dir.remove(25)
        #print(valid_dir)
        
        #valid_dir = [x for x in range(1, 40)]  # Model test
        dir_count = 0

        rf_index = 0
        
        if mode == 'train':
            outlier_list = range(49500, 50000)
            outlier_list = []
        else:
            #outlier_list = range(18000, 19000)
            outlier_list = range(49500, 50000)
            outlier_list = []
        
        
        '''
        if mode == 'train':
            outlier_list = range(2000, 4999)
        else:
            #outlier_list = range(18000, 19000)
            outlier_list = range(0, 2000)
            outlier_list.extend(range(2500, 4999))
        '''
        
        rf_index = -1
        gt_index = -1
        img_index = -1
        human_label_index = -1  #by 승환

        for file, file2 in zip(data_path_list, data_path_list2):
            if dir_count in remove_dir:
                dir_count += 1
                continue
            if mode == 'train' and ((dir_count in valid_dir) or (dir_count not in los_dir and dir_count not in nlos_dir)):
                dir_count += 1
                continue
            elif mode == 'test' and ((dir_count in valid_dir) or (dir_count not in los_dir and dir_count not in nlos_dir)):
                dir_count += 1
                continue
            #elif mode == 'test' and dir_count not in test_dir:
            #    dir_count += 1
            #    continue
            elif mode == 'valid' and (dir_count in test_dir or dir_count in valid_dir):
                dir_count += 1
                continue
                
            '''
            elif mode == 'valid' and dir_count not in valid_dir:
                dir_count += 1
                continue
            '''

            if os.path.isdir(file) is True:
                # 각 폴더 안의 npy 데이터
                rf_file_list = glob.glob(file + '/raw/*.npy')
                
                if mode == 'train':
                    if len(rf_file_list) > 5000:
                        for i in range(6000):
                            rf_file_list.pop()  
                                    
                    elif len(rf_file_list) < 5000:
                        pass
                        #for i in range(500):
                        #    rf_file_list.pop()
                            
                    else:
                        for i in range(1000):
                            rf_file_list.pop()
                            
                if mode == 'test' :
                    if len(rf_file_list) > 5000:
                        for i in range(4000):
                            rf_file_list.pop(0)
                        for i in range(5000):
                            rf_file_list.pop()
                            
                    elif len(rf_file_list) < 5000:
                        pass
                        #for i in range(500):
                        #    rf_file_list.pop(0)
                    
                    else:
                        for i in range(4000):
                            rf_file_list.pop(0)
                    
                            
                rf_file_list = sorted(rf_file_list)
                print('dir(raw):', file, '\t# of data :', len(rf_file_list))
                #print(rf_file_list)
                for rf in rf_file_list:

                    rf_index += 1
                    if rf_index in outlier_list:
                        continue
                    #temp_raw_rf = np.load(rf)[:, :, self.cutoff:] # 3x3 rf
                    #temp_raw_rf = np.load(rf)[:1, :1, self.cutoff:] # 1x1 rf
                    for num_sig1 in range(3):
                        for num_sig2 in range(3):
                            if mode == 'train' and (num_sig1 !=2 or num_sig2 !=2):
                            #if mode == 'train' and (num_sig1 == 0 and num_sig2 == 0):
                                temp_raw_rf = np.load(rf)[num_sig1, num_sig2, self.cutoff:] # 1x1 rf
                                '''
                                plt.plot(temp_raw_rf)
                                plt.savefig('./raw_rf/signal_plot_normal_{}.png'.format(num_sig1*3+num_sig2))
                                plt.clf()
                                '''
                                #temp_raw_rf = np.load(rf)[:2, :2, self.cutoff:] # 2x2 rf
                                #print("raw shape", temp_raw_rf.shape) # 3, 3, 2048 - cutoff
                                
                                #----- normalization ------
                                if self.is_normalize is True:
                                    
                                    ###############for 2d###################
                                    '''
                                    stdev = np.std(temp_raw_rf)
                                    temp_raw_rf = temp_raw_rf/stdev
                                    '''
                                    ##############for 1d#################
                                    stdev = np.std(temp_raw_rf)
                                    temp_raw_rf = temp_raw_rf/stdev
                                    
                                #temp_raw_rf = np.transpose(temp_raw_rf, (2, 1, 0)).transpose(0, 2, 1)
                                #temp_raw_rf = torch.tensor(temp_raw_rf).float()
            
                                
                                if self.flatten:
                                    #print("before flatten ",temp_raw_rf.shape) # 3x3 - [1792, 3, 3] # 1x1 - [1792, 1, 1]
                                    #temp_raw_rf= torch.tensor(temp_raw_rf).float()
                                    #temp_raw_rf= temp_raw_rf.view(1792, 1)
                                    
                                    #temp_raw_rf= temp_raw_rf.view(-1, 1792)
                                    #############################2차원으로 만드려면 다음 코드 사용############################
                                    '''
                                    #temp_raw_rf = temp_raw_rf.view(128, -1) # 128, signal 개수
                                    #f1 = open("test.txt", "w")
                                    
                                    
                                    print("now shape",temp_raw_rf.shape)  # 3x3 - [128, 126] # 1x1 - [128, 14]
                                    temp_raw_rf = temp_raw_rf.unsqueeze(0)
                                    #print("before_resize", temp_raw_rf)
                                    
                                    resize_transform = transforms.Compose(
                                        [transforms.ToPILImage(),
                                         #transforms.Resize((self.input_size, self.input_size)), # for 3*3
                                         transforms.Resize((64, 112)), # 2*2
                                         transforms.ToTensor()]
                                    )
                                    '''
                                    
                                    #temp_raw_rf = resize_transform(temp_raw_rf) 
                                    #print(temp_raw_rf.shape) # 3x3 - [1, input_size, input_size] #1x1 - [1, input_size, input_size]
                                    '''
                                    print("after_resize", temp_raw_rf)
                                    f1.write("after_resize : ")
                                    for i in range(temp_raw_rf.shape[1]):
                                      line = "%d\n" %i
                                      f1.write(line)
                                      for j in range(temp_raw_rf.shape[2]):
                                        if j %5 == 0:
                                          f1.write("\n")
                                        data = "%f    " %temp_raw_rf[0][i][j]
                                        f1.write(data)
                                        
                                    f1.close
                                    '''
                                    ###########################################################################################
                                #print("now shape",temp_raw_rf.shape)
                                rf_data.append(temp_raw_rf)
                                
                            elif mode == 'test' and num_sig1 == 2 and num_sig2 == 2 :
                                temp_raw_rf = np.load(rf)[num_sig1, num_sig2, self.cutoff:] # 1x1 rf
                                #temp_raw_rf = np.load(rf)[:2, :2, self.cutoff:] # 2x2 rf
                                #print("raw shape", temp_raw_rf.shape) # 3, 3, 2048 - cutoff
                                '''
                                #----- normalization for 2d ------
                                if self.is_normalize is True:
                                    for i in range(temp_raw_rf):
                                        for j in range(temp_raw_rf):
                                            stdev = np.std(temp_raw_rf)
                                            temp_raw_rf = temp_raw_rf/stdev
                                '''
                                ##############for 1d#################
                                stdev = np.std(temp_raw_rf)
                                temp_raw_rf = temp_raw_rf/stdev
                                #temp_raw_rf = np.transpose(temp_raw_rf, (2, 1, 0)).transpose(0, 2, 1)
                                #temp_raw_rf = torch.tensor(temp_raw_rf).float()
            
                                
                                if self.flatten:
                                    
                                    #temp_raw_rf= torch.tensor(temp_raw_rf).float()
                                    #temp_raw_rf= temp_raw_rf.view(1792, 1)
                                    #temp_raw_rf= temp_raw_rf.view(-1, 1792)
                                    #############################2차원으로 만드려면 다음 코드 사용############################
                                    '''
                                    #temp_raw_rf = temp_raw_rf.view(128, -1) # 128, signal 개수
                                    #f1 = open("test.txt", "w")
                                    
                                    
                                    print("now shape",temp_raw_rf.shape)  # 3x3 - [128, 126] # 1x1 - [128, 14]
                                    temp_raw_rf = temp_raw_rf.unsqueeze(0)
                                    #print("before_resize", temp_raw_rf)
                                    
                                    resize_transform = transforms.Compose(
                                        [transforms.ToPILImage(),
                                         #transforms.Resize((self.input_size, self.input_size)), # for 3*3
                                         transforms.Resize((64, 112)), # 2*2
                                         transforms.ToTensor()]
                                    )
                                    '''
                                    
                                    #temp_raw_rf = resize_transform(temp_raw_rf) 
                                    #print(temp_raw_rf.shape) # 3x3 - [1, input_size, input_size] #1x1 - [1, input_size, input_size]
                                    '''
                                    print("after_resize", temp_raw_rf)
                                    f1.write("after_resize : ")
                                    for i in range(temp_raw_rf.shape[1]):
                                      line = "%d\n" %i
                                      f1.write(line)
                                      for j in range(temp_raw_rf.shape[2]):
                                        if j %5 == 0:
                                          f1.write("\n")
                                        data = "%f    " %temp_raw_rf[0][i][j]
                                        f1.write(data)
                                        
                                    f1.close
                                    '''
                                    ###########################################################################################
                                #print("now shape",temp_raw_rf.shape)
                                rf_data.append(temp_raw_rf)
                print(len(rf_data))            
                

                #break
                '''
                ground truth data 읽어오기.
                heatmap 형태. 총 데이터의 개수* keypoint * width * height
                '''
                gt_file_list = glob.glob(file2 + '/gt/*')
                gt_file_list = sorted(gt_file_list)
                
                human_labeling = glob.glob(file2 + '/label.txt')
                f = open(human_labeling[0], 'r')
                line = f.readline()
                label = int(line)
                print(label)
                if label in(human_label_list_read):
                    for i in range(len(human_label_list_read)):
                        if human_label_list_read[i] == label:
                            human_label_list.append(human_label_list[i])
                            break
                else:
                    human_label_index += 1
                    human_label_list.append(human_label_index)
                human_label_list_read.append(label)
                f.close
                if human_label_index not in real_human_label_list:
                    real_human_label_list.append(human_label_index)
                    
                #labeling by 승환
                
                if mode == 'train' :
                    if len(gt_file_list)>5000:
                        for i in range(6000):
                            gt_file_list.pop()
                            
                    elif len(gt_file_list) < 5000:
                        pass
                    #    for i in range(500):
                    #        gt_file_list.pop()
                            
                    else:
                        for i in range(1000):
                            gt_file_list.pop()
                            
                if mode == 'test' :
                    if len(gt_file_list) > 5000:
                        for i in range(4000):
                            gt_file_list.pop(0)
                        for i in range(5000):
                            gt_file_list.pop()
                    
                    elif len(gt_file_list) < 5000:
                        pass
                    #    for i in range(500):
                    #        gt_file_list.pop(0)
                            
                    else:
                        for i in range(4000):
                            gt_file_list.pop(0)
                            
                print('dir(gt):', file2, '\t# of data :', len(gt_file_list))
                #np_load_old = np.load
                #np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
                #----- gt 메모리에 올려놓기 -----
                """for gt in gt_file_list:
                    temp_gt = np.load(gt)
                    temp_gt = torch.tensor(temp_gt).float()
                    temp_gt = temp_gt.reshape(13, 120, 120)
                    #print(temp_gt.shape, temp_gt.dtype)
                    gt_list.append(temp_gt)
                """
                #----- gt 파일 이름명만 리스트에 넣어놓기 -----
                for gt in gt_file_list:    #ex of gt_file_list = folder/gt/00001.npy //5000개or 10000개
                    gt_index += 1
                    if gt_index in outlier_list:
                        continue
                    gt_list.append(gt) # 5000~ 10000개의 gt 넣기
                print(human_label_list_read) #승환
                print("real :", real_human_label_list)
                #np.load = np_load_old
                
                if self.load_img is True:
                    img_file_list = glob.glob(file + '/img/*.jpg')
                    img_file_list = sorted(img_file_list)
                    print('dir(img):', file, '\t# of data :', len(img_file_list))
                    for img in img_file_list:
                        img_index += 1
                        if img_index in outlier_list:
                            img_index += 1
                            continue
                        temp_img = cv2.imread(img)
                        img_list.append(temp_img)

          
            dir_count += 1
            
        use_sig = 1
        if mode == 'train' :
            use_sig = int(len(rf_data)/40000)
        self.use_sig = use_sig
        
        people_cnt = (len(nlos_dir) + len(los_dir)) #사람수 x signal 수
        rf_cnt_by_dir = int(len(rf_data)/people_cnt) #폴더당 rf수
        print("rf_cnt_by_dir", rf_cnt_by_dir)
        #for i in range(1700):
            #print((rf_data[0])[0][i] - (rf_data[1])[0][i])
            #print((rf_data[0])[0][i])
            
        ######################for test##########################
        #if mode == "test":
        #    test_rf = rf_data[6000:7000]
        #    for i in range(1000):
        #        rf_data.insert(4000+i, test_rf[i])
        #    for i in range(1000):
        #        rf_data.pop(7000)
            
            
            
        s_stack = 3
        d_stack = 1
        
        ##########################avg_space#######################
        '''
        for i in range(people_cnt): 
            for j in range(rf_cnt_by_dir):
                d_set = []
                if j > rf_cnt_by_dir - (s_stack) :
                    continue
                else :
                    
                        
                    rf_space_average = rf_data[i*rf_cnt_by_dir + j]
                    for k in range(s_stack-1):
                        rf_space_average  += rf_data[i*rf_cnt_by_dir + j + (k+1)]
                    rf_space_average = rf_space_average/s_stack
                    #d_set.append(rf_space_average)
                    #d_set.append(rf_data[i*rf_cnt_by_dir + j])
                    #for l in range(1700):
                    #    print(rf_space_average[0][l])
                    #temp_rf_data = rf_data[i*rf_cnt_by_dir + j] - rf_space_average
                    temp_rf_data = rf_space_average.squeeze()
                    #print(temp_rf_data)
                    temp_rf_data = temp_rf_data.unsqueeze(0)
                    if (i*rf_cnt_by_dir + j) == 2500 :
                        print(temp_rf_data)
                    new_rf_data.append(temp_rf_data)
                    
        '''
        #####################################################
        
        
        
        #########################use d stack#######################
        for i in range(people_cnt): 
            for j in range(rf_cnt_by_dir):
                d_set = []
                s_set = []
                if j > rf_cnt_by_dir - ((s_stack-1)*use_sig+1) :
                    continue
                else :
                   #############avg_space##################
                   for k in range(s_stack):
                       s_set.append((rf_data[i*rf_cnt_by_dir + j + k*use_sig]))
                   
                   avg_s = np.zeros(1792)
                   for k in range(s_stack):
                       avg_s += s_set[k]
                           
                   avg_s = avg_s/s_stack
                   avg_s = torch.tensor(avg_s).float()
                   '''
                   for k in range(d_stack):
                       #d_set.append((s_set[k]+s_set[k+1])/s_stack)
                       #d_set.append((s_set[k+1]-s_set[k])/s_stack)
                       d_set.append(s_set[s_stack-1] - avg_s)
                   #temp_rf_data = torch.cat([d_set[0], d_set[1]], dim = 0)
                   #temp_rf_data = d_set[1]
                   #print(temp_rf_data.shape)
                   ##############normalization#################
                   if self.is_normalize is True: 
                       for d_num in range(len(d_set)):   
                           stdev = np.std(d_set[d_num])
                           d_set[d_num] = d_set[d_num]/stdev
                           d_set[d_num]= (torch.tensor(d_set[d_num]).float()).unsqueeze(0)
                           #d_set[d_num].view(1, 1792)      
                             '''
                   #temp_rf_data = torch.cat([d_set[0].squeeze(), d_set[1].squeeze()], dim = 0) #avg_s, d
                   #print("rf",temp_rf_data)
                   #print(d_set[0])
                   temp_rf_data = avg_s
                   
                   #print(temp_rf_data.size())
                   #temp_rf_data.view([-1, 1792])
                   #print(temp_rf_data.size())
                   #temp_rf_data = d_set[1] #only d
                   new_rf_data.append(temp_rf_data)
                   
                   #if mode == 'train':
                   #    if ((i*rf_cnt_by_dir + j + k) % 3200) == 0 and  (i*rf_cnt_by_dir + j + k) < 20*3200 :
                   #        num = i*rf_cnt_by_dir + j + k
                   #        plt.plot(temp_rf_data)
                   #        plt.savefig('./train_sig/signal_plot_normal_{}.png'.format(num))
                   #        plt.clf()
                   #else :
                   #    if ((i*rf_cnt_by_dir + j + k) % 10) == 0 and  (i*rf_cnt_by_dir + j + k) < 200 :
                   #        num = i*rf_cnt_by_dir + j + k
                   #        plt.plot(temp_rf_data)
                   #        plt.savefig('./test_sig/signal_plot_normal_{}.png'.format(num))
                   #        plt.clf()
                       #for t in temp_rf_data.shape :
                       #    print(temp_rf_data[shape])
                   
                   
                   '''
                   if mode == 'train' :
                       temp_rf_data = torch.cat([d_set[0], -d_set[1]], dim = 0)
                       #temp_rf_data = -d_set[1] 
                       ##############normalization#################
                       new_rf_data.append(temp_rf_data.unsqueeze(0))
                    '''
                        
        #######################################################
        
        
        ################for only 2 stack#######################
        '''
        for i in range(people_cnt): 
            for j in range(rf_cnt_by_dir):
                if j > rf_cnt_by_dir - (s_stack) :
                    continue
                else :
                    rf_space_average = rf_data[i*rf_cnt_by_dir + j + 1]
                    temp_rf_data = rf_data[i*rf_cnt_by_dir + j] - rf_space_average
                    temp_rf_data = temp_rf_data.squeeze()
                    temp_rf_data = temp_rf_data.unsqueeze(0)
                    new_rf_data.append(temp_rf_data)
        '''
      
        ###############for GAN#################
        
        
        ###############################
        
        for i in range(len(rf_data)):
            if mode == 'train':
                if (i % 3200 == 0) and  (i < 20*3200) :
                    plt.plot(rf_data[i])
                    plt.savefig('./train_sig/signal_plot_normal_{}.png'.format(i))
                    plt.clf()
            else :
                if (i % 10 == 0) and  (i < 200) :
                     
                    plt.plot(rf_data[i])
                    plt.savefig('./test_sig/signal_plot_normal_{}.png'.format(i))
                    plt.clf()
        
        people_cnt = (len(nlos_dir) + len(los_dir))
        print("people_cnt", people_cnt)
        for i in range(people_cnt):
            human_num_list.append((i+1)*rf_cnt_by_dir -1)
        '''
        ###########s, d 썻을 때#############
        new_rf_cnt_by_dir = int(len(new_rf_data)/people_cnt) # 폴더당 rf 수
        print("new_rf_cnt_by_dir", new_rf_cnt_by_dir)
        for i in range(people_cnt):
            human_num_list.append((i+1)*(new_rf_cnt_by_dir)-1)
        #print("rf_data_크기 :", len(new_rf_data))
        print(human_num_list)
        #for i in range(1700):
        #    print((new_rf_data[0])[0][i])
        ####################################
        '''
        #############GAN###################
        
        #############################
        

                
        self.rf_data = new_rf_data
        self.gt_list = gt_list
        self.human_num_list = human_num_list #by 승환
        self.human_label_list = human_label_list #by 승환
        self.real_human_label_list = real_human_label_list
        
        #print(len(gt_list))
        
        if self.mode == 'valid' and len(self.gt_list) == 0:
            for i in range(len(self.rf_data)):
                self.gt_list.append(np.zeros((13, 120, 120)))
        self.img_list = img_list
        print("end - data read")
        print("size of dataset", len(self.rf_data))

    def __len__(self):    
        return len(self.rf_data)

    def __getitem__(self, idx):
        #if self.mode == 'none':    #뭔가 이상
        #    gt = np.zeros((13, 120, 120))
        #print(idx)
        if True :
            #gt = np.load(self.gt_list[idx])
            #by 승환
            for i in range(len(self.human_num_list)):
                if idx - self.human_num_list[i] <= 0 :
                    gt_label = self.human_label_list[i]
                    break
                    #
        antena_label = idx%8
        domain = 0
        if idx > 4000*8*5-1 :
            if idx >= 4000*8*6 and idx< 4000*8*7:
                pass
            else:
                domain = 1
            
        #gt = torch.tensor(gt).float()
        #gt = gt.reshape(13, 120, 120)
        gt = 0
        rf = self.rf_data[idx] 
        '''
        for i in range(len(self.real_human_label_list)):
            if gt_label == self.real_human_label_list[i]:
                gt_label = self.real_human_label_list[i]
        '''
        gt_label = torch.tensor(gt_label) #tensor화
        antena_label = torch.tensor(antena_label)
        #print("gt label : ", gt_label) #by 승환
        #---- augmentation  ----#
        random_prob = torch.rand(1)

        if self.mode == 'train' and self.augmentation != 'None' and random_prob < self.augmentation_prob:
            random_target = torch.randint(low=0, high=len(self.rf_data), size=(1,)).item() 
            #print("random target : ", random_target)# by 승환
            #while random_target == idx: # random target이 동일하다면 다시 뽑음.
            #    random_target = torch.randint(low=0, high=len(self.rf_data), size=(1,)).item()
            
            target_gt = np.load(self.gt_list[random_target])
            target_gt = torch.tensor(target_gt).reshape(gt.shape)
            target_rf = self.rf_data[random_target] 
            #by 승환
            for i in range(len(self.human_num_list)):
                if random_target - self.human_num_list[i] <= 0 :
                    target_gt_label = self.human_label_list[i]
                    break
                    #
            #print("augmetatied rf = ", rf.shape)
            #print("augmented gt = ", gt.shape)
            if self.augmentation == 'cutmix':
                rf, gt = cutmix(rf, target_rf, gt, target_gt)
            elif self.augmentation == 'mixup':
                rf, gt = mixup(rf, target_rf, gt, target_gt)
            elif self.augmentation =='intensity':
                rf = self.intensity(rf)
            elif self.augmentation =='all':
                r = np.random.rand(1)
                if r < 0.4:
                    rf, gt = cutmix(rf, target_rf, gt, target_gt)
                #elif r < 0.7:
                    #rf = self.intensity(rf)
                elif r < 0.8:
                    rf, gt = mixup(rf, target_rf, gt, target_gt)
            else:
                print('wrong augmentation')

        if self.load_img is False:
            #gaussian noise
            #print("load_img : false") #by 승환
            if self.mode == 'train' and self.is_gaussian is True:
                gt = gt + torch.randn(gt.size()) * self.std + self.mean
            
            #print("len rf, len gt", len(rf), len(gt)) #by 승환
            return rf, gt, gt_label, antena_label, domain
            # return self.rf_data[idx], self.gt_list[idx]
        else:
            print("load_img : True")
            return rf, gt, self.img_list[idx]

def cutmix(rf, target_rf, gt, target_gt):
    beta = 1.0
    lam = np.random.beta(beta, beta)
    # print("rf.size ", rf.size())
    # print(rf.size()[-2])
    bbx1, bby1, bbx2, bby2 = rand_bbox(rf.size(), lam)
    # print(bbx1, bbx2, bby1, bby2)
    rf[:, bbx1:bbx2, bby1:bby2] = target_rf[:, bbx1:bbx2, bby1:bby2]
    # print((bbx2-bbx1)*(bby2-bby1))
    # print(rf.size()[-1] * rf.size()[-2])
    lam = 1 - (((bbx2 - bbx1) * (bby2 - bby1)) / (rf.size()[-1] * rf.size()[-2]))
    new_rf = rf
    new_gt = lam * gt + (1 - lam) * target_gt
    return new_rf, new_gt

def mixup(rf, target_rf, gt, target_gt):
    '''
    논문에서는 배치 내에서 섞지만, 전체 데이터에서 mixup.
    '''
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)

    new_gt = lam * gt + (1 - lam) * target_gt
    new_rf = lam * rf + (1 - lam) * target_rf
    return new_rf, new_gt

class Intensity(nn.Module):
    def __init__(self, scale=0.05):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, ))
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    

