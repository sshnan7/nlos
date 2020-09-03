import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """ The AlexNet :
    'A. Krizhevsky, I. Sutskever, and G. Hinton.
    Imagenet clas-sification with deep convolutional neural networks. InNIPS , 2012'
    Args:
        Nj (int): Size of joints.
    """

    def __init__(self, Nj):
        super(AlexNet, self).__init__()
        self.Nj = Nj
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, Nj*2),
        )

    
    def forward(self, x):
        x = self.features(x)
        #print(x)
        x = self.avgpool(x)
        #print(x)
        x = torch.flatten(x, 1)
        #print(x)
        x = self.classifier(x)
        #print(x)
        x = x.view(-1, self.Nj, 2)
        return x
        '''
        # layer1
        h = F.relu(self.conv1(x))
        print(h)
        h = F.max_pool2d(h, 3, stride=2)
        print(h)
        # layer2
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 3, stride=2)
        print(h)
        # layer3-5
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        print(h)
        h = F.max_pool2d(h, 3, stride=2)
        h = h.view(-1, 256*6*6)
        print(h)
        # layer6-8
        h = F.dropout(F.relu(self.fc6(h)), training=self.training)
        h = F.dropout(F.relu(self.fc7(h)), training=self.training)
        h = self.fc8(h)
        print(h)
        return h.view(-1, self.Nj, 2)
        '''