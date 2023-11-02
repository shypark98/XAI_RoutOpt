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
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_curve, auc
from matplotlib.gridspec import GridSpec
from datetime import datetime 



inputs = [ 'cell_den', 'pin_den', 'rudy', 'lnet_rudy', 'gnet_rudy',\
        'snet_rudy', 'wire_den_rsmt', 'lnet_den_rsmt', 'gnet_den_rsmt',\
        'chan_den_rsmt', 'chan_den_v_rsmt', 'chan_den_h_rsmt', 'wire_den_egr1',\
        'wire_den_egr2', 'wire_den_egr3', 'wire_den_egr4', 'wire_den_egr5',\
        'wire_den_egr6', 'wire_den_egr7', 'wire_den_egr8', 'chan_den_egr1',\
        'chan_den_egr2', 'chan_den_egr3', 'chan_den_egr4', 'chan_den_egr5',\
        'chan_den_egr6', 'chan_den_egr7', 'chan_den_egr8', 'via_den_egr1',\
        'via_den_egr2', 'via_den_egr3', 'via_den_egr4', 'via_den_egr5',\
        'via_den_egr6', 'via_den_egr7', 'lnet_den_egr', 'gnet_den_egr',\
        'chan_den_egr', 'chan_den_v_egr', 'chan_den_h_egr','avg_terms',\
        'num_insts', 'num_terms', 'num_nets','num_gnets', 'num_lnets',\
        'clk_ratio', 'wns', 'tns']

output = 'num_drvs'

print("length of inputs : ", len(inputs))
def read_csv(fileName):
    start = time.time()
    chunk = pd.read_csv(fileName, index_col=False, chunksize=100000, engine="c")
    df = pd.concat(chunk)
    end = time.time()
    return df

def create_dir( dirPath ):
    try:
        if not(os.path.isdir(dirPath)):
            os.makedirs(os.path.join(dirPath))
            #print( dirPath )
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create output directory.")
            raise

def randseed(seed):
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    np.random.seed(seed)
    return g

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_layout_images(file_list, save_home, gcell_size):
    layout_dir = '%s/layout/x%d' % (save_home, gcell_size)
    flatten_dir = '%s/flatten/x%d' % (save_home, gcell_size) 
    create_dir(layout_dir)
    create_dir(flatten_dir)

    for file_path in tqdm(file_list):
        #if "fake_ldpc3_util_0.70" in tar_dir or\
        #"fake_nova2_util_0.60" in tar_dir:
        #    continue
        if ".csv" not in file_path:
            continue
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df = df.dropna()

        df = df.sort_values(by=['col','row'])
        total_num_drvs = np.sum(df['num_drvs'].values)
        #print(file_path, total_num_drvs)

        # file exception 처리 할것
        width, height = int(np.max(df['col'].values)+1), int(np.max(df['row'].values)+1)
        channel = len(inputs)
        x = df[inputs].values.reshape((width, height, channel))
        y = df[output].values.reshape((width,height))
  
        x = torch.FloatTensor(x).permute(2,0,1)
        y = torch.FloatTensor(y)

        #layout data
        data = x, y
        #print(tar_dir, x.shape) #print(y.shape)

        file_name = file_path.split('/')[-1].strip('.csv')
        data_path = "%s/%s" % (layout_dir, file_name)
        torch.save(data, data_path)

def create_clip_images(save_home, gcell_size, hop):
    layout_dir = '%s/layout/x%d' % (save_home, gcell_size)
    flatten_dir = '%s/flatten/x%d' % (save_home, gcell_size)

    for design in tqdm(os.listdir(layout_dir)):
        data_path = os.path.join(layout_dir, design)
        x, y = torch.load(data_path)
        pad = (hop, hop, hop, hop)
        x_pad = torch.nn.functional.pad(x,pad, 'constant', 0)
        channel, width, height = x.size()
        #print(channel, width, height)
        img_size = 2*hop+1
        x_flatten = torch.zeros((width*height, channel,img_size, img_size))
        y_flatten = torch.zeros((width*height))
        offset = hop

        for b, (i,j) in enumerate(itertools.product(list(range(width)), list(range(height)))):
            i1 = offset + i-hop
            i2 = offset + i+hop+1
            j1 = offset + j-hop
            j2 = offset + j+hop+1
            x_flatten[b,:,:,:] = x_pad[:,i1:i2, j1:j2]
            y_flatten[b] = y[i,j]

        data = x_flatten, y_flatten, width, height
        data_path = "%s/%s" % (flatten_dir, design)
        torch.save(data, data_path)

            # 검증용 코드
            #x_flatten = x_flatten[:,:,5,5]
            #x_flatten = x_flatten.squeeze()
            #print(x_flatten.shape)
            #x_reshape = x_flatten.reshape((width, height,channel))
            #print(x_reshape.size())
            #x_reshape = x_reshape.permute((2,0,1))
            #is_same = (x == x_reshape)
            #print(is_same)
            #torch.argwhere(is_same==False)

class Database(object):
    def __init__(self, data_paths, batch_size):
        self.batch_size = batch_size
        self.files = []
        self.offset = {}

        print("Initialize database")
        index_iter = 0
        for data_path in tqdm(data_paths):
            x, y, w, h = torch.load(data_path)
            samples = len(x)
            num_batches = int(samples / self.batch_size)
            self.offset[data_path] = index_iter + num_batches
            index_iter += num_batches
            for _ in range(num_batches):
                self.files.append(data_path)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x, y, w, h = torch.load(self.files[i])
        start = i - self.offset[self.files[i]]
        end = start + self.batch_size
        return x[start:end,:,:,:], y[start:end]

def write_minmax_tensor(target_dir, save_dir, gcell_size):
    x_min = 10000*torch.ones(len(inputs), dtype=torch.float32)
    x_max = -10000*torch.ones(len(inputs),dtype=torch.float32)
    #print(target_dir, save_dir)
    # get min-max stats
    print("Get min-max stats")
    for file_name in tqdm(os.listdir(target_dir)):
        file_path = os.path.join(target_dir, file_name)
    #print(file_path)
        x, y, w, h = torch.load(file_path)

        batch, channel, width, height = x.size()
        cx = int(width/2)
        cy = int(height/2)

        x = x[:,:,cx,cy]
        _min = torch.min(x, dim=0)[0]
        _max = torch.max(x, dim=0)[0]
        x_min = torch.minimum(x_min, _min)
        x_max = torch.maximum(x_max, _max)

    torch.save(x_min, "%s/x_min.pt" % save_dir)
    torch.save(x_max, "%s/x_max.pt" % save_dir)

    print("x_min : ", x_min)
    print("x_max : ", x_max)

def create_batches(file_paths, save_dir, x_min, delim, batch_size):

    batch_idx = 0
    for file_path in tqdm(file_paths):
        x, y, w, h = torch.load(file_path)
        samples = len(x)
        num_batches = int(samples / batch_size)
        # norm
        x = torch.FloatTensor((x-x_min)/delim)
        x = torch.nan_to_num(x)
        #print(num_batches)
        for i in tqdm(range(num_batches)):
            start = batch_size*i
            end = start + batch_size
            x_mini = x[start:end,:,:,:].clone()
            y_mini = y[start:end].clone()
                #print(x_mini.shape, start, end)
            data = x_mini, y_mini
            save_path = "%s/batch_%d.pt" % (save_dir, batch_idx)
            torch.save(data, save_path)
            batch_idx+=1
            print(save_path)

def main():

    data_home = "./CSV"
    save_home = "./TORCH"
    data_config = "%s/data.config" % (data_home)
    designs = []
    
    test_list = [ ]
    data_Set = os.listdir(data_home)
    train_list = []
    for data in data_Set:
        if data not in test_list:
            train_list.append(data)
    print(train_list)

    gcell_size = 7
    hop = 7
    batch_size = 50
    file_list = []
    
    for file_name in train_list:
        file_list.append(os.path.join(data_home, file_name))
    create_layout_images(file_list, save_home, gcell_size)

    file_list = []

    for file_name in test_list:
        file_list.append(os.path.join(data_home, file_name))
    create_layout_images(file_list, save_home, gcell_size)

    create_clip_images(save_home, gcell_size, hop)

    # Write normalize vector
    target_dir = '%s/flatten/x%d' % (save_home, gcell_size)
    save_dir = '%s/minmax/x%d' % (save_home, gcell_size)
    create_dir(save_dir)
    write_minmax_tensor(target_dir, save_dir, gcell_size)
    flatten_home = "%s/flatten/x%d" % (save_home, gcell_size)

    x_min = torch.load("%s/minmax/x%d/x_min.pt" % (save_home, gcell_size))
    x_max = torch.load("%s/minmax/x%d/x_max.pt" % (save_home, gcell_size))
    img_size = 2*hop+1
    delim = x_max - x_min
    delim[delim==0] = 1
    x_min = x_min.repeat(img_size**2,1).transpose(1,0).reshape((len(inputs),img_size,img_size))
    delim = delim.repeat(img_size**2,1).transpose(1,0).reshape((len(inputs),img_size,img_size))

    # 학습가속화를 위해 data preprocessing 필요
    # Create mini batch (train)
    batch_index = 0
    train = []
    batch_home = "%s/batch/x%d/train" % (save_home, gcell_size)
    create_dir(batch_home)
    for file_name in train_list:
        file_name = file_name.strip('.csv')
        file_path = os.path.join(flatten_home, file_name)
        print(file_path)
        train.append(file_path)
    create_batches(train, batch_home, x_min, delim, batch_size)

    # Create mini batch (test)
    test = []
    batch_home = "%s/batch/x%d/test" % (save_home, gcell_size)
    create_dir(batch_home)
    for file_name in test_list:
        file_name = file_name.strip('.csv')
        file_path = os.path.join(flatten_home, file_name)
        print(file_path)
        test.append(file_path)
    create_batches(test,batch_home, x_min, delim, batch_size)


if __name__ == '__main__':
    main()

