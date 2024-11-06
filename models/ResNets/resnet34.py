import torch
from torch import nn
from nn.modules.resnet_block import ResNetBlock


class ResNet34(nn.Module):
    def __init__(self, channels, kernel_size, convs_per_block, num_blocks, activation, norm, in_channel, pool):
        super(ResNet34, self).__init__()
        self.channels = []
        for i, channel in enumerate(channels):
            self.channels += [channel] * (convs_per_block * num_blocks[i])
        self.network = [nn.Conv2d(in_channel, channels[0], kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(channels[0]),
                        nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        last_channel = self.channels[0]
        
        for i in range(0, len(self.channels), convs_per_block): # generates res blocks
            in_channels = self.channels[i : i+convs_per_block]
            in_channels += [in_channels[-1]]
            downsample = False
            if in_channels[0] != last_channel:
                tmp_channels = [last_channel] + in_channels[1:]
                last_channel = in_channels[0]
                in_channels = tmp_channels
                downsample = True
            self.network += [ResNetBlock(in_channels, activation, norm, pool, downsample)]
        self.network += [nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, 1000)]
        self.network = nn.Sequential(* self.network)

    def forward(self, x):
        x = self.network(x)
        return x    