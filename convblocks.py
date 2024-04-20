import torch 
import torch.nn as nn
import torch.nn.functional as F

class CAM(nn.Module):
    
    def __init__(self, in_channel, reduction_factor=16, bn_momentum=0.999, l2=0.0001):
        super(CAM, self).__init__()
        self.in_channel = in_channel
        self.reduction_factor = reduction_factor
        self.bn_momentum = bn_momentum
        self.l2 = l2
        
        self.pool = torch.nn.MaxPool2d(kernel_size=7,
                                       stride=1,
                                       padding='SAME')
        
        self.squeeze = torch.nn.Conv2d(in_channels=self.in_channel,
                                       out_channels=(self.in_channel//self.reduction_factor),
                                       kernel_size=1,
                                       stride=1,
                                       padding='SAME'
                                       )
        
        self.squeeze_bn = torch.nn.BatchNorm2d(num_features=(self.in_channel//self.reduction_factor), momentum=self.bn_momentum)
        
        self.excite = torch.nn.Conv2d(in_channels=(self.in_channel//self.reduction_factor),
                                       out_channels=self.in_channel,
                                       kernel_size=1,
                                       stride=1,
                                       padding='SAME'
                                       )
        
        self.excite_bn = torch.nn.BatchNorm2d(num_features=self.in_channel, momentum=self.bn_momentum)
    
    def forward(self, inputs, training=False):
        pool = self.pool(inputs)
        squeeze = F.ReLU(self.squeeze_bn(self.squeeze(pool)))
        excite = F.sigmoid(self.excite_bn(self.excite(squeeze)))
        return inputs*excite
        

class FIRE(nn.Module):
    
    def __init__(self, sq1x1_planes, ex1x1_planes, ex3x3_planes, bn_momentum=0.999, l2=0.0001):
        super(FIRE, self).__init__()
        self.sq1x1_planes = sq1x1_planes
        self.ex1x1_planes = ex1x1_planes
        self.ex3x3_planes = ex3x3_planes
        self.bn_momentum = bn_momentum
        self.l2 = l2
        
        self.squeeze = torch.nn.Conv2d(in_channels=self.sq1x1_planes,
                                       out_channels=self.sq1x1_planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding='SAME'
                                       )
        
        self.squeeze_bn = torch.nn.BatchNorm2d(num_features=self.sq1x1_planes, momentum=self.bn_momentum)
        
        self.expand1x1 = torch.nn.Conv2d(in_channels=sq1x1_planes,
                                       out_channels=self.ex1x1_planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding='SAME'
                                       )
        
        self.expand1x1_bn = torch.nn.BatchNorm2d(num_features=self.ex1x1_planes, momentum=self.bn_momentum)
        
        self.expand3x3 = torch.nn.Conv2d(in_channels=self.ex1x1_planes,
                                       out_channels=self.ex3x3_planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding='SAME'
                                       )
        
        self.expand3x3_bn = torch.nn.BatchNorm2d(num_features=self.ex3x3_planes, momentum=self.bn_momentum)
    
    def forward(self, inputs, training=False):
        squeeze = F.relu(self.squeeze_bn(self.squeeze(inputs), training))
        ex1x1 = F.relu(self.expand1x1_bn(self.expand1x1(squeeze), training))
        ex3x3 = F.relu(self.expand3x3_bn(self.expand3x3(squeeze), training))
        return torch.concat([ex1x1, ex3x3], dim=3)


class FIREUP(nn.Module):

    def __init__(self, sq1x1_planes, ex1x1_planes, ex3x3_planes, stride, bn_momentum=0.99, l2=0.0001):
        super(FIREUP, self).__init__()
        self.sq1x1_planes = sq1x1_planes
        self.ex1x1_planes = ex1x1_planes
        self.ex3x3_planes = ex3x3_planes
        self.stride = stride
        self.bn_momentum = bn_momentum
        self.l2 = l2

        self.squeeze = nn.Conv2d(
            in_channels=self.sq1x1_planes,
            out_channels=self.sq1x1_planes,
            kernel_size=1,
            stride=1,
            padding=0,  # no padding
            bias=False
        )
        self.squeeze_bn = nn.BatchNorm2d(self.sq1x1_planes, momentum=self.bn_momentum)

        if self.stride == 2:
            self.upconv = nn.ConvTranspose2d(
                in_channels=self.sq1x1_planes,
                out_channels=self.sq1x1_planes,
                kernel_size=(1, 4),
                stride=(1, 2),
                padding=(0, 1),  # same padding
                bias=False
            )

        self.expand1x1 = nn.Conv2d(
            in_channels=self.sq1x1_planes,
            out_channels=self.ex1x1_planes,
            kernel_size=1,
            stride=1,
            padding=0,  # no padding
            bias=False
        )
        self.expand1x1_bn = nn.BatchNorm2d(self.ex1x1_planes, momentum=self.bn_momentum)

        self.expand3x3 = nn.Conv2d(
            in_channels=self.sq1x1_planes,
            out_channels=self.ex3x3_planes,
            kernel_size=3,
            stride=1,
            padding=1,  # same padding
            bias=False
        )
        self.expand3x3_bn = nn.BatchNorm2d(self.ex3x3_planes, momentum=self.bn_momentum)

    def forward(self, inputs):
        squeeze = F.relu(self.squeeze_bn(self.squeeze(inputs)))
        if self.stride == 2:
            upconv = F.relu(self.upconv(squeeze))
        else:
            upconv = squeeze
        expand1x1 = F.relu(self.expand1x1_bn(self.expand1x1(upconv)))
        expand3x3 = F.relu(self.expand3x3_bn(self.expand3x3(upconv)))
        return torch.cat([expand1x1, expand3x3], dim=3)       
            