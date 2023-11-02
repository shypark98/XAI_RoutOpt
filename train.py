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


def get_arguments():
    deviceCount = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=range(deviceCount), help='number of gpu to use')
    parser.add_argument('--opt', type=str, choices=['Adam', 'RMSprop', 'SGD'], default='SGD', help='Optimizer')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--threshold', type=int, choices=range(0,11), default=2)
    parser.add_argument('--randseed', type=int, default=777)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--early_stop', type=int, choices=range(3,11), default=5)
    parser.add_argument('--b1', type=int, default=2)
    parser.add_argument('--b2', type=int, default=3)
    parser.add_argument('--b3', type=int, default=3)
    parser.add_argument('--c1', type=int, default=60)
    parser.add_argument('--c2', type=int, default=90)
    parser.add_argument('--c3', type=int, default=120)
    parser.add_argument('--f1', type=int, default=300)
    parser.add_argument('--f2', type=int, default=30)
    parser.add_argument('--fold', type=int, choices=range(3,11), default=5)
    parser.add_argument('--hop', type=int, choices=[5,7], default=15)
    parser.add_argument('--gcell_size', type=int, choices=range(5,11), default=15)
    parser.add_argument('--project', type=str, default='DAC-LBR-Test') 
    parser.add_argument('--in_channel', type=int, default=49)
    parser.add_argument('--data_dir', type=str, default='./TORCH/batch')
    parser.add_argument('--weight', type=float, default=0.5) 
    parser.add_argument('--checkpoint_dir', type=str, default='./MODEL')
    arg = parser.parse_args()
    return arg


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
        #self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        #num_neurons = img_size**2 * out_channels[2]
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

def train(config, cur_time, model_name):

    
    print(config)
    img_size = 2*config.hop+1
    device = torch.device('cuda:%d' % config.gpu)
    
    

    
    out_channels = [ config.c1, config.c2, config.c3 ]
    num_blocks = [ config.b1, config.b2, config.b3 ]
    num_neurons = [ config.f1, config.f2 ]
    
    print('[LOG] ',out_channels)
    print('[LOG] ',num_blocks)
    print('[LOG] ',num_neurons)
    
    checkpoint_dir = "%s/%s" % (config.checkpoint_dir, model_name)
    create_dir(checkpoint_dir)
    print('[LOG] ',cur_time)
    print('[LOG] ',model_name)
    print('[LOG] ',checkpoint_dir)

    model = ResNet(ResidualBlock, img_size, config.in_channel, out_channels, num_blocks, num_neurons, 1)
    model = model.to(device)

    print('[LOG] ',device)

    train_dir = "%s/x%d/train" % (config.data_dir, config.gcell_size)
    test_dir = "%s/x%d/test" % (config.data_dir, config.gcell_size)
    
    train_files = []
    test_files = []
    for file_name in os.listdir(train_dir):
        #print(file_name)
        train_files.append(os.path.join(train_dir, file_name))
    for file_name in os.listdir(test_dir):
        #print(file_name)
        test_files.append(os.path.join(test_dir, file_name))


    print('[LOG] ',len(train_files))
    print('[LOG] ',len(test_files))

    model_config = {}
    model_config['gcell_size'] = config.gcell_size
    model_config['img_size'] = img_size
    model_config['in_channel'] = config.in_channel
    model_config['out_channels'] = out_channels
    model_config['num_blocks'] = num_blocks
    model_config['num_neurons'] = num_neurons


    criterion = nn.BCELoss()
    
    if config.opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    elif config.opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.opt == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    #return

    torch.autograd.set_detect_anomaly(True)

    step = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=config.randseed)
    early_stopping_rounds = 5
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_files)):
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        data_loader = {}
        data_loader['train'] = torch.utils.data.DataLoader(\
                train_files, batch_size=config.batch_size, \
                sampler=train_sampler, num_workers=2, collate_fn=my_collate)
        data_loader['val'] = torch.utils.data.DataLoader(\
                train_files, batch_size=config.batch_size, \
                sampler=val_sampler, num_workers=2, collate_fn=my_collate)
        data_loader['test'] = torch.utils.data.DataLoader(\
                test_files, batch_size=config.batch_size, collate_fn=my_collate)

        # For early stopping
        min_val_loss = 1e+30
        early_stop = False
        patient = 0
        
        for epoch in range(config.epochs):
            step += 1
            losses = {}
            # Train, Validation
            for phase in ['train', 'val', 'test']:
                if phase == 'test':#and step % 5 != 4:
                    continue
                
                losses[phase] = AverageMeter()
                if phase == 'train':
                    model.train()
                else:
                    model.train(False)
                    model.eval()

                print("Total # of steps for %s %d"  % (phase, len(data_loader[phase]))) 
   
                trues = []
                preds = []
                for i, (x, y) in enumerate(tqdm(data_loader[phase])):
                    x = x.squeeze().to(device)
                    y = y.squeeze().to(device)
                    y = (y > config.threshold).float()
                    y_p = model(x)
                    y_p = torch.sigmoid(y_p)
                    loss = criterion(y_p.reshape((-1,1)), y.reshape((-1,1)))
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    losses[phase].update(loss.cpu().item(), y.shape[0])
                    trues.extend(y.squeeze().cpu().tolist())
                    preds.extend(y_p.squeeze().cpu().tolist())     
                    del x, y, y_p

               
                preds = np.array(preds, dtype=float)
                trues = np.array(trues, dtype=np.int64)

                fpr, tpr, th = roc_curve(trues, preds, pos_label=1)
                auroc = auc(fpr, tpr)
                
                preds = np.array(preds>0.5, dtype=np.int64)
                f1 = f1_score(trues, preds, pos_label=1, zero_division=0)
                recall = recall_score(trues, preds, pos_label=1, zero_division=0)
                precision = precision_score(trues, preds, pos_label=1, zero_division=0)
                acc = accuracy_score(trues, preds)
                       
                conf = conf = confusion_matrix(trues, preds, labels=[0,1])



                summary = {}
                summary['loss/%s' % phase] = losses[phase].avg
                summary['f1/%s' % phase] = f1 #.item()
                summary['acc/%s' % phase] = acc #.item()
                summary['auroc/%s' % phase] = auroc #.item()
                summary['recall/%s' % phase] = recall #.item()
                summary['precision/%s' % phase] = precision #.item()
                summary['confusion_matrix/%s' % phase] = \
                    wandb.plot.confusion_matrix(probs=None, y_true=trues, preds=preds, class_names=['clean','dirty']) 

                wandb.log(summary, step=step)
                log = "[{fold:d}] Epoch {epoch:d} Step {step:d} {phase:s} loss {loss:.3f} acc {acc:.3f} f1 {f1:.3f} auroc {auroc:.3f} recall {recall:.3f} precision {precision:.3f}"\
                       .format(fold=fold, epoch=epoch, step=step, phase=phase,loss=losses[phase].avg, \
                               acc=acc, f1=f1, auroc=auroc, recall=recall, precision=precision)
                print(log)
                print(conf)
                del preds, trues
            model_path = "%s/step_%d" % (checkpoint_dir, step)
            torch.save({'step': step,\
                        'epoch': epoch,\
                        'fold': fold,\
                        'model_state_dict': model.state_dict(),\
                        'model_config': model_config,\
                        'optimizer_state_dict': optimizer.state_dict()}, model_path)


            if min_val_loss > losses['val'].avg:
                min_val_loss = losses['val'].avg
            else:
                patient += 1
                if patient > early_stopping_rounds:
                    print("EarlyStopping")
                    early_stop = True
            if early_stop:
                break


wandb_dir = './WANDB'
create_dir(wandb_dir)

def main():
    arg = get_arguments()
    cur_time = datetime.now().strftime("%m-%d-%H-%M")
    model_name = "model_%s" % cur_time
    # Logging in Weight & Bias
    wandb.init(\
            settings=wandb.Settings(start_method='fork'),\
            project=arg.project, name=model_name, config=arg)
    train(arg, cur_time, model_name)


def run_sweep(config=None):
    cur_time = datetime.now().strftime("%m-%d-%H-%M")
    model_name = "model_%s" % cur_time
    run = wandb.init(config=config, name=model_name, dir=wandb_dir)
    #run = wandb.init(config=config)
    config = wandb.config
    print("config : ", config)
    #train_bk(config)
    train_simple(config, cur_time, model_name)
    #train_simple(w_config, cur_time, model_name)
    #wandb.finish()
    run.finish()

#train_simple(config, cur_time, model_name)
#sweep_id = wandb.sweep(sweep_config, project="DAC-LBR-project")
#wandb.agent(sweep_id, run_sweep, count=15)

if __name__ == '__main__':
    main()




