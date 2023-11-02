import torch
from tqdm import tqdm
import time
import wandb
import numpy as np
import argparse
import itertools
import parmap
import os
import pandas as pd
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, roc_curve, auc
from matplotlib.gridspec import GridSpec
from datetime import datetime 
import sys
import traceback

def create_dir( dirPath ):
    try:
        if not(os.path.isdir(dirPath)):
            os.makedirs(os.path.join(dirPath))
            #print( dirPath )
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create output directory.")
            raise

def my_collate(batch):
    x_batch = []
    y_batch = []
    for i, data_path in enumerate(batch):
        x, y = torch.load(data_path) 
        x_batch.append(x)
        y_batch.append(y)

    x_batch = torch.cat(x_batch, dim=0)
    y_batch = torch.cat(y_batch, dim=0)
    return x_batch, y_batch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


#ResNet
class ResNet(nn.Module):
    def __init__(self, block, img_size, in_channel, out_channels, layers, fc_neurons, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = in_channel
        self.conv = conv3x3(in_channel, in_channel)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, out_channels[0], layers[0])
        self.layer2 = self.make_layer(block, out_channels[1], layers[1], 1)
        self.layer3 = self.make_layer(block, out_channels[2], layers[2], 1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)

        num_neurons = (img_size-2)**2 * out_channels[2]

        self.fc = nn.Sequential(\
            nn.Linear(num_neurons, fc_neurons[0]),\
            nn.ReLU(inplace=True),\
            nn.Linear(fc_neurons[0], fc_neurons[1]),\
            nn.ReLU(inplace=True),\
            nn.Linear(fc_neurons[1], num_classes))

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential( conv3x3(self.in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


