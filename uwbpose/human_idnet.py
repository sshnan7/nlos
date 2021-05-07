# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# SSH

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1
INPUT_D = 1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, 7, stride, padding=3)
        self.bn1 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 7, stride, padding=3)
        self.bn2 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Class_ResNet(nn.Module):    
    def __init__(self, block, layers):
        self.inplanes = 64
        self.deconv_with_bias = False
        print("---------------flatten pose net---------------")
        super(Class_ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
        )
        self.layer1 = self._make_layer(block, 64, 2,stride=1)  # Bottleneck 3
        #self.do1 = nn.Dropout(0.5)
        self.layer2 = self._make_layer(block, 96, 2, stride=2)
        #self.do2 = nn.Dropout(0.5)
        self.layer3 = self._make_layer(block, 128, 2, stride=2)  # Bottleneck 6
        #self.do3 = nn.Dropout(0.5)
        self.layer4 = self._make_layer(block, 192, 2, stride=2)  # Bottlenect 3
        #self.do4 = nn.Dropout(0.5)
        self.layer5 = self._make_layer(block, 256, 2, stride=2)
        self.linear = nn.Linear(256*7, 256*7)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256*7, 10)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        #x = self.do1(x)
        x = self.layer2(x)
        #x = self.do2(x)
        x = self.layer3(x)
        #x = self.do3(x)
        x = self.layer4(x)
        #x = self.do4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        t = x
        x = self.linear2(x)
        return x
   

class Generator(nn.Module):
#################UNet########################

    def __init__(self, block, layers):
        self.inplanes = 64
        self.deconv_with_bias = False
        super(Generator, self).__init__()
        '''
        def conv1(in_c, out_c, kernel_size = 9, stride = 1, padding = 4, bias = False):
            layers = []
            layers += [nn.Conv1d(in_channels = in_c, out_channels = out_c, kernel_size = kernel_size, stride = 1, padding = 4, bias= bias)]
            layers += [nn.BatchNorm1d(num_features = out_c)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout()]
            
            return nn.Sequential(*layers)
            
        self.encoding1_1 = conv1(1, 64)
        #self.encoding1_2 = conv1(64, 64)
        self.downto2 = nn.MaxPool1d(kernel_size=2)
        
        self.encoding2_1 = conv1(64, 128)
        #self.encoding2_2 = conv1(128, 128)
        self.downto3 = nn.MaxPool1d(kernel_size=2)
        
        self.encoding3_1 = conv1(128, 256)
        #self.encoding3_2 = conv1(256, 256)
        self.downto4 = nn.MaxPool1d(kernel_size = 2)
        
        self.encoding4_1 = conv1(256, 512)
        #self.encoding4_2 = conv1(512, 512)
        self.downto5 = nn.MaxPool1d(kernel_size = 2)
        
        self.encoding5_1 = conv1(512, 1024)
        #self.encoding5_2 = conv1(1024, 1024)
        self.upto4 = conv1(1024, 512)
        
        self.unpool4 = nn.ConvTranspose1d(in_channels = 512, out_channels = 512, kernel_size = 10, stride = 2, padding = 4) # (512, 224)
        self.decoding4_1 = conv1(2*512, 512)
        self.upto3 = conv1(512, 256)
        
        self.unpool3 = nn.ConvTranspose1d(in_channels = 256, out_channels = 256, kernel_size = 10, stride = 2, padding = 4) # (256, 448)
        self.decoding3_1 = conv1(2*256, 256)
        self.upto2 = conv1(256, 128)
        
        self.unpool2 = nn.ConvTranspose1d(in_channels = 128, out_channels = 128, kernel_size = 10, stride = 2, padding = 4) # (128, 896)
        self.decoding2_1 = conv1(2*128, 128)
        self.upto1 = conv1(128, 64)
        
        self.unpool1 = nn.ConvTranspose1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2, padding = 4) # (64, 1792)
        self.decoding1_1 = conv1(2*64, 64) #(64, 1792)
        self.out = nn.Conv1d(64, 1, 9, 1, 4, bias = False)
    
    def forward(self, x):
        encoding1 = self.encoding1_1(x) #(1, 1792) -> (64, 1784)
        #encoding1 = self.encoding1_2(x)
        #print("layer1", encoding1.shape)
        x = self.downto2(encoding1) #(64, 1776) -> (64, 888)
        
        encoding2 = self.encoding2_1(x) #(64, 888) -> (128, 880)
        #encoding2 = self.encoding2_2(x)
        #print("layer2", encoding2.shape)
        x = self.downto3(encoding2) #(128, 872) -> (128, 436)
        
        encoding3 = self.encoding3_1(x) #(128, 436) -> (256, 428)
        #encoding3 = self.encoding3_2(x)
        #print("layer3", encoding3.shape)
        x = self.downto4(encoding3) #(256, 420) -> (256, 210)
        
        encoding4 = self.encoding4_1(x) #(256, 210) -> (512, 202)
        #encoding4 = self.encoding4_2(x)
        #print("layer4", encoding4.shape)
        x = self.downto5(encoding4) #(512, 194) -> (512, 97)
        
        x = self.encoding5_1(x) #(512, 97) -> (1024, 89)
        #x = self.encoding5_2(x)
        x = self.upto4(x) 
        
        x = self.unpool4(x) #(512, 73) -> (512, 146)
        #print(x.shape)
        #encoding4 = encoding4[:,: , 24:170]
        x = torch.cat([encoding4, x], dim = 1)
        #print("layer up 4", x.shape)
        x = self.decoding4_1(x)
        x = self.upto3(x)
        
        x = self.unpool3(x) #(256, 130) -> (256, 260)
        #encoding3 = encoding3[:, :, 80:340]
        x = torch.cat([encoding3, x], dim = 1)
        x = self.decoding3_1(x)
        x = self.upto2(x)
        
        x = self.unpool2(x) #(128, 244) -> (128, 488)
        #encoding2 = encoding2[:,: , 92 : 580]
        x = torch.cat([encoding2, x], dim = 1)
        x = self.decoding2_1(x)
        x = self.upto1(x)
        
        x = self.unpool1(x) #(64, 472) -> (64, 944)
        #encoding1 = encoding1[:, :, 416:1360]
        x = torch.cat([encoding1, x], dim = 1)
        #print("layer up 1", x.shape)
        x = self.decoding1_1(x)
        x = self.out(x) #(1, 928)
        
        return x


###############일반적 generator###############

    
        '''
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            #nn.Dropout(),
            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            #nn.Dropout(),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            #nn.Dropout(),
            nn.Conv1d(256, 256, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Conv1d(256, 512, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            #nn.Dropout(),
            nn.Conv1d(512, 1024, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(1024, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            #nn.Dropout(),
            nn.Conv1d(1024, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(1, momentum=BN_MOMENTUM),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            
        )
        
        #self.layer1 = self._make_layer(block, 64, 2,stride=1)  # Bottleneck 3
        #self.do1 = nn.Dropout(0.5)
        #self.layer2 = self._make_layer(block, 96, 2, stride=2)
        #self.do2 = nn.Dropout(0.5)
        #self.layer3 = self._make_layer(block, 128, 2, stride=2)  # Bottleneck 6
        #self.do3 = nn.Dropout(0.5)
        #self.layer4 = self._make_layer(block, 192, 2, stride=2)  # Bottlenect 3
        #self.do4 = nn.Dropout(0.5)
        #self.layer5 = self._make_layer(block, 256, 2, stride=2)
        
        self.linear = nn.Sequential(
            nn.Linear(56, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),
            #nn.Linear(64, 64),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(),
            #nn.Linear(64, 64),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(),
            nn.Linear(64, 1792),
        )
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        #print("initial", x.shape)
        #x = self.conv1(x)
        
        #x = x.view(x.size(0), -1)
        
        x = self.linear(x)
        #print("conv1", x.shape)
        #x = self.layer1(x)
        #print("layer1", x.shape)
        #x = self.layer2(x)
        #print("layer2", x.shape)
        #x = self.layer3(x)
        #print("layer3", x.shape)
        #x = self.layer4(x)
        #print("layer4", x.shape)
        #x = self.layer5(x)
        #print("layer5", x.shape)
        #x = self.layer5(x)
        
        #x = x.view(x.size(0), -1)
        #x = self.linear(x)
        #x = self.linear2(x)
        x = x.unsqueeze(1)
        x = x.view(-1, 1,  1792)
        #print(x.shape)
        return x

class Discriminator2(nn.Module):
    def __init__(self, block, layers) :
        self.inplanes = 64
        self.deconv_with_bias = False
        super(Discriminator2, self).__init__()
        print("---------------flatten pose net---------------")
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3, bias=False), #st = 3, padding = 3
            nn.BatchNorm1d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3, bias=False),#st = 3, padding = 3
            nn.BatchNorm1d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(256, 512, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 3
            nn.BatchNorm1d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(512, 1024, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.BatchNorm1d(1024, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            
            #nn.BatchNorm1d(1024, momentum=BN_MOMENTUM),
            #nn.ReLU(inplace=True),
            #nn.Conv1d(1024, 1, kernel_size=6, stride=1, padding=0, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.BatchNorm1d(1024, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(1024, 1024, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.BatchNorm1d(1024, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(1024, 2048, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.BatchNorm1d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(2048, 1, kernel_size=7, stride=1, padding=0, bias=False), #st = 5, padding = 1
            nn.Sigmoid()
        )
        self.linear = nn.Sequential(
            nn.Conv1d(1024, 1, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.Linear(56,128),
            nn.Dropout(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        '''
        self.layer1 = self._make_layer(block, 64, 2,stride=1)  # Bottleneck 3
        #self.do1 = nn.Dropout(0.5)
        self.layer2 = self._make_layer(block, 96, 2, stride=2)
        #self.do2 = nn.Dropout(0.5)
        self.layer3 = self._make_layer(block, 128, 2, stride=2)  # Bottleneck 6
        #self.do3 = nn.Dropout(0.5)
        self.layer4 = self._make_layer(block, 192, 2, stride=2)  # Bottlenect 3
        #self.do4 = nn.Dropout(0.5)
        self.layer5 = self._make_layer(block, 256, 2, stride=2)
        self.linear = nn.Linear(256*7, 256*7)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256*7, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        '''
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.layer1(x)
        #x = self.do1(x)
        #x = self.layer2(x)
        #x = self.do2(x)
        #x = self.layer3(x)
        #x = self.do3(x)
        #x = self.layer4(x)
        #x = self.do4(x)
        #x = self.layer5(x)
        #x = x.view(x.size(0), -1)
        #x = self.linear(x)
        #x = self.relu(x)
        #t = x
        '''
        x = 0.05*x
        for i in range(x.shape[0]):
            if x[i][0] >= 0.999:
                x[i][0] = 0.999
            if x[i][0] <= 0.001:
                x[i][0] = 0.001
                '''
        #x = self.linear2(x)
        #x = self.sigmoid(x)
        x = x.view(x.size(0), -1)
        return x
        
class Discriminator(nn.Module):
    def __init__(self, block, layers) :
        self.inplanes = 64
        self.deconv_with_bias = False
        super(Discriminator, self).__init__()
        print("---------------flatten pose net---------------")
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            #nn.Dropout(),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3, bias=False), #st = 3, padding = 3
            nn.Dropout(),
            nn.BatchNorm1d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            #nn.Dropout(),
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3, bias=False),#st = 3, padding = 3
            nn.Dropout(),
            nn.BatchNorm1d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            #nn.Dropout(),
            nn.Conv1d(256, 512, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 3
            nn.Dropout(),
            nn.BatchNorm1d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            #nn.Dropout(),
            nn.Conv1d(512, 1024, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.Dropout(),
            nn.BatchNorm1d(1024, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            #nn.Dropout(),
            
            #nn.BatchNorm1d(1024, momentum=BN_MOMENTUM),
            #nn.ReLU(inplace=True),
            #nn.Conv1d(1024, 1, kernel_size=6, stride=1, padding=0, bias=False),
        )
        '''
        self.conv2 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.BatchNorm1d(1024, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(1024, 1024, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.BatchNorm1d(1024, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(1024, 2048, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.BatchNorm1d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(),
            nn.Conv1d(2048, 1, kernel_size=7, stride=1, padding=0, bias=False), #st = 5, padding = 1
            nn.Sigmoid()
        )
        '''
        self.linear = nn.Sequential(
            nn.Conv1d(1024, 1, kernel_size=7, stride=1, padding=3, bias=False), #st = 5, padding = 1
            nn.Dropout(),
            nn.BatchNorm1d(1, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
           # nn.Dropout(),
            nn.Linear(56,128),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        '''
        self.layer1 = self._make_layer(block, 64, 2,stride=1)  # Bottleneck 3
        #self.do1 = nn.Dropout(0.5)
        self.layer2 = self._make_layer(block, 96, 2, stride=2)
        #self.do2 = nn.Dropout(0.5)
        self.layer3 = self._make_layer(block, 128, 2, stride=2)  # Bottleneck 6
        #self.do3 = nn.Dropout(0.5)
        self.layer4 = self._make_layer(block, 192, 2, stride=2)  # Bottlenect 3
        #self.do4 = nn.Dropout(0.5)
        self.layer5 = self._make_layer(block, 256, 2, stride=2)
        self.linear = nn.Linear(256*7, 256*7)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256*7, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        '''
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.layer1(x)
        #x = self.do1(x)
        #x = self.layer2(x)
        #x = self.do2(x)
        #x = self.layer3(x)
        #x = self.do3(x)
        #x = self.layer4(x)
        #x = self.do4(x)
        #x = self.layer5(x)
        #x = x.view(x.size(0), -1)
        x = self.linear(x)
        #x = self.relu(x)
        #t = x
        '''
        x = 0.05*x
        for i in range(x.shape[0]):
            if x[i][0] >= 0.999:
                x[i][0] = 0.999
            if x[i][0] <= 0.001:
                x[i][0] = 0.001
                '''
        #x = self.linear2(x)
        #x = self.sigmoid(x)
        x = x.view(x.size(0), -1)
        return x

class PoseResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        self.deconv_with_bias = False
        print("---------------flatten pose net---------------")
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1)      # Bottleneck 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)      # Bottleneck 6
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer5 = nn.Linear(128*25, 2048)
        
        self.deconv_layer = self._make_deconv_layer(
            4,  # NUM_DECONV_LAYERS
            [256,256,256,256],  # NUM_DECONV_FILTERS
            [3,4,4,4],  # NUM_DECONV_KERNERLS
        )
        self.final_layer = nn.Conv2d(
            in_channels=256,  # NUM_DECONV_FILTERS[-1]
            out_channels=13,  # NUM_JOINTS,
            kernel_size=1,  # FINAL_CONV_KERNEL
            stride=1,
            padding=0  # if FINAL_CONV_KERNEL = 3 else 1
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 5:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):  
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        s = 2
        self.inplanes = 512
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            if i==0:
                s=3
            else:
                s=2
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=s,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)
  
        x = self.layer4(x)
    
        
        x = x.view(x.size(0),512,5,5)
        x = self.deconv_layer(x)
 
        x = self.final_layer(x)
     
        
        return x
'''
class PoseResNet(nn.Module):
    def __init__(self, block, layers):
        super(PoseResNet, self).__init__()
        self.a1 = PoseResNet_element(block,layers)
        self.a2 = PoseResNet_element(block,layers)
        self.a3 = PoseResNet_element(block,layers)
        self.a4 = PoseResNet_element(block,layers)
        self.a5 = PoseResNet_element(block,layers)
        self.a6 = PoseResNet_element(block,layers)
        self.a7 = PoseResNet_element(block,layers)
        self.a8 = PoseResNet_element(block,layers)
        self.a9 = PoseResNet_element(block,layers)
        self.a10 = PoseResNet_element(block,layers)
        self.a11 = PoseResNet_element(block,layers)
        self.a12 = PoseResNet_element(block,layers)
        self.a13 = PoseResNet_element(block,layers)
    def forward(self, x):
        x1 = self.a1(x)
        x2 = self.a2(x)
        x3 = self.a3(x)
        x4 = self.a4(x)
        x5 = self.a5(x)
        x6 = self.a6(x)
        x7 = self.a7(x)
        x8 = self.a8(x)
        x9 = self.a9(x)
        x10 = self.a10(x)
        x11 = self.a11(x)
        x12 = self.a12(x)
        x13 = self.a13(x)
        out = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13),1)
        return out
'''
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
              }

def get_human_id_net(num_layer):
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]
    model = Class_ResNet(block_class,layers)
    return model

def get_generator(num_layer):
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]
    model = Generator(block_class,layers)
    return model
    
def get_discriminator(num_layer):
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]
    model = Discriminator(block_class,layers)
    return model

def get_discriminator2(num_layer):
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]
    model = Discriminator2(block_class,layers)
    return model
    
def get_2d_pose_net(num_layer, input_depth):
    global INPUT_D
    INPUT_D = input_depth
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]

    # model = PoseResNet(block_class, layers, cfg, **kwargs)

    model = PoseResNet(block_class, layers)

    # if is_train and cfg.MODEL.INIT_WEIGHTS:
    #    model.init_weights(cfg.MODEL.PRETRAINED)
    # model.init_weights('models/imagenet/resnet50-19c8e357.pth')
    return model