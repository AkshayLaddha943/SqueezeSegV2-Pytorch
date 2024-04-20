import torch
import torch.nn as nn
from encoding import cfg
from convblocks import CAM, FIRE, FIREUP
import torch.nn.functional as F

class defconfig(object):
    ZENITH_LEVEL = 32
    AZIMUTH_LEVEL = 240
    NUM_FEATURES = 6
    L2_WEIGHT_DECAY = 0.05
    DROP_RATE = 0.1
    BN_MOMENTUM = 0.9
    REDUCTION = 16

config = defconfig()

class SqueezeSegV2(nn.Module):
    """SqueezeSegV2 Model as custom PyTorch Module"""

    def __init__(self, cfg):
        super(SqueezeSegV2, self).__init__()
        self.NUM_CLASS = cfg.NUM_CLASS
        self.CLASSES = cfg.CLASSES

        # input shape
        self.ZENITH_LEVEL = config.ZENITH_LEVEL
        self.AZIMUTH_LEVEL = config.AZIMUTH_LEVEL
        self.NUM_FEATURES = config.NUM_FEATURES

        # regularization
        self.drop_rate = config.DROP_RATE
        self.l2 = config.L2_WEIGHT_DECAY
        self.bn_momentum = config.BN_MOMENTUM

        # Metrics
        self.miou_tracker = None  # PyTorch doesn't have a built-in MeanIoU metric
        self.loss_tracker = None  # PyTorch doesn't have a built-in loss tracker

        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

        # Layers

        # Encoder
        self.conv1 = nn.Conv2d(
            in_channels=self.NUM_FEATURES,
            out_channels=64,
            kernel_size=3,
            stride=(1, 2),
            padding=(1, 0),  # same padding
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.cam1 = CAM(in_channels=64, bn_momentum=self.bn_momentum, l2=self.l2)

        self.conv1_skip = nn.Conv2d(
            in_channels=self.NUM_FEATURES,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn1_skip = nn.BatchNorm2d(64, momentum=self.bn_momentum)

        self.fire2 = FIRE(sq1x1_planes=16, ex1x1_planes=64, ex3x3_planes=64, bn_momentum=self.bn_momentum, l2=self.l2)
        self.cam2 = CAM(in_channels=128, bn_momentum=self.bn_momentum, l2=self.l2)
        self.fire3 = FIRE(sq1x1_planes=16, ex1x1_planes=64, ex3x3_planes=64, bn_momentum=self.bn_momentum, l2=self.l2)
        self.cam3 = CAM(in_channels=128, bn_momentum=self.bn_momentum, l2=self.l2)

        self.fire4 = FIRE(sq1x1_planes=32, ex1x1_planes=128, ex3x3_planes=128, bn_momentum=self.bn_momentum, l2=self.l2)
        self.fire5 = FIRE(sq1x1_planes=32, ex1x1_planes=128, ex3x3_planes=128, bn_momentum=self.bn_momentum, l2=self.l2)

        self.fire6 = FIRE(sq1x1_planes=48, ex1x1_planes=192, ex3x3_planes=192, bn_momentum=self.bn_momentum, l2=self.l2)
        self.fire7 = FIRE(sq1x1_planes=48, ex1x1_planes=192, ex3x3_planes=192, bn_momentum=self.bn_momentum, l2=self.l2)
        self.fire8 = FIRE(sq1x1_planes=64, ex1x1_planes=256, ex3x3_planes=256, bn_momentum=self.bn_momentum, l2=self.l2)
        self.fire9 = FIRE(sq1x1_planes=64, ex1x1_planes=256, ex3x3_planes=256, bn_momentum=self.bn_momentum, l2=self.l2)

        # Decoder
        self.fire10 = FIREUP(sq1x1_planes=64, ex1x1_planes=128, ex3x3_planes=128, stride=2, bn_momentum=self.bn_momentum,
                             l2=self.l2)
        self.fire11 = FIREUP(sq1x1_planes=32, ex1x1_planes=64, ex3x3_planes=64, stride=2, bn_momentum=self.bn_momentum,
                             l2=self.l2)
        self.fire12 = FIREUP(sq1x1_planes=16, ex1x1_planes=32, ex3x3_planes=32, stride=2, bn_momentum=self.bn_momentum,
                             l2=self.l2)
        self.fire13 = FIREUP(sq1x1_planes=16, ex1x1_planes=32, ex3x3_planes=32, stride=2, bn_momentum=self.bn_momentum,
                             l2=self.l2)

        self.conv14 = nn.Conv2d(
            in_channels=16,
            out_channels=self.NUM_CLASS,
            kernel_size=3,
            stride=1,
            padding=1,  # same padding
            bias=False
        )
        self.dropout = nn.Dropout(self.drop_rate)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        lidar_input, lidar_mask = inputs[0], inputs[1]

        # Encoder
        x = F.relu(self.bn1(self.conv1(lidar_input)))

        cam1_output = self.cam1(x)

        conv1_skip = self.bn1_skip(self.conv1_skip(lidar_input))

        x = F.max_pool2d(cam1_output, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        x = self.fire2(x)
        x = self.cam2(x)
        x = self.fire3(x)
        cam3_output = self.cam3(x)

        x = F.max_pool2d(cam3_output, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        x = self.fire4(x)
        fire5_output = self.fire5(x)

        x = F.max_pool2d(fire5_output, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        fire9_output = self.fire9(x)

        # Decoder
        x = self.fire10(fire9_output)
        x = torch.add(x, fire5_output)
        x = self.fire11(x)
        x = torch.add(x, cam3_output)
        x = self.fire12(x)
        x = torch.add(x, cam1_output)
        x = self.fire13(x)
        x = torch.add(x, conv1_skip)

        x = self.dropout(x)

        logits = self.conv14(x)

        probabilities, predictions = self.segmentation_head(logits, lidar_mask)

        return probabilities, predictions

    def segmentation_head(self, logits, lidar_mask):
        
        probabilities = self.softmax(logits)

        predictions = torch.argmax(probabilities, dim=1, keepdim=False)

        # set predictions to the "None" class where no points are present
        predictions = torch.where(lidar_mask.squeeze(),
                                  predictions,
                                  torch.ones_like(predictions) * self.CLASSES.index("None")
                                  )
        return probabilities, predictions
            