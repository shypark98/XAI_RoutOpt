from util import *
import scipy
import argparse
import torch


deviceCount = torch.cuda.device_count()
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, choices=range(deviceCount), help='number of gpu to use')
arg = parser.parse_args()
# Set device
if arg.gpu == None:
    device = torch.device('cpu')
else:
    device = torch.device("cuda:%d" % arg.gpu)



#model_checkpoint = "/project/cad-team/DAC-LBR2/MODEL/model_11-23-01-39/step_6"
model_checkpoint = "./MODEL/model_02-02-14-04/step_46"
#device = torch.device('cuda')
checkpoint = torch.load(model_checkpoint, map_location=device)

model_config = checkpoint['model_config']
gcell_size = model_config['gcell_size']
img_size = model_config['img_size']
in_channel = model_config['in_channel']
out_channels = model_config['out_channels']
num_blocks = model_config['num_blocks']
num_neurons = model_config['num_neurons']

model = ResNet(ResidualBlock, img_size, in_channel, out_channels, num_blocks, num_neurons, 1)
model.load_state_dict(checkpoint['model_state_dict'])


data_home = "./TORCH"
flatten_dir = "%s/flatten/x%d" % (data_home, gcell_size)
x_min = torch.load('%s/minmax/x%d/x_min.pt' % (data_home,gcell_size))
x_max = torch.load('%s/minmax/x%d/x_max.pt' % (data_home,gcell_size))
delim = (x_max-x_min)
delim[delim==0]=1
x_min = x_min.repeat(img_size**2,1).transpose(1,0).reshape((in_channel,img_size,img_size))
delim = delim.repeat(img_size**2,1).transpose(1,0).reshape((in_channel,img_size,img_size))

#train_list = []
#test_list = []
file_list = os.listdir(flatten_dir)
#for name in file_list:
#    if "fake_ldpc2" in name:
#        train_list.append(name)
#    elif "fake_nova1" in name:
#        train_list.append(name)
#    elif "fake_nova9" in name:
#        train_list.append(name)
#test_list = [x for x in file_list if x not in train_list]


fig_dir = "fig"
create_dir(fig_dir)


model = model.to(device)
DRVth = 3

for file_name in tqdm(file_list):
    design = file_name
    
    data_path = os.path.join(flatten_dir, file_name)
    (x,y,w,h) = torch.load(data_path)
    x = (x-x_min) / delim
    x = torch.nan_to_num(x)
    x = x.to(device)
    y = y.to(device)

    batch_size = 100

    y_t = np.zeros((w*h))
    y_p = np.zeros((w*h))

    for start in range(0, len(x), batch_size):
        end = min(len(x), start+batch_size)
        x_part = x[start:end]
        y_part = y[start:end]

        #
        y_t[start:end] = y_part.cpu().detach().numpy()
        y_p[start:end] = model(x_part).squeeze().cpu().detach().numpy()
        

    trues = np.array((y_t > DRVth), dtype=np.int32)
    preds = np.array((y_p > 0.5), dtype=np.int32)
    f1 = f1_score(trues, preds, pos_label=1, zero_division=0)
    rec = recall_score(trues, preds, pos_label=1, zero_division=0)
    pre = precision_score(trues, preds, pos_label=1, zero_division=0)
    acc = accuracy_score(trues, preds)
    conf = confusion_matrix(trues, preds, labels=[0,1])
    tn, fp, fn, tp = conf.ravel()
    num_drvs = torch.sum(y).item()

    log = "\n%s #drvs %d\n" % (design, num_drvs)
    log += " - acc %.3f f1 %.3f rec %.3f pre %.3f tn %d fp %d fn %d tp %d\n" % (acc, f1, rec, pre, tn, fp, fn, tp)
    print(log)


    h_zoom = 8.0
    w_zoom = 8.0

    nrows = 1
    ncols = 2
    gs = GridSpec(nrows=nrows, ncols=ncols, width_ratios=[1,1], height_ratios=[1])
    fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])


    img_true = y_t.reshape((w,h))
    img_pred = y_p.reshape((w,h))

    img_true = scipy.ndimage.interpolation.zoom(img_true, [w_zoom, h_zoom])
    img_pred = scipy.ndimage.interpolation.zoom(img_pred, [w_zoom, h_zoom])

    im1 = ax1.imshow(img_true, interpolation='nearest', cmap='jet', vmin=0, vmax=3.0)
    im2 = ax2.imshow(img_pred, interpolation='nearest', cmap='jet', vmin=0, vmax=3.0)

    #fig.suptitle(key)
    ax1.set_title("True")
    ax2.set_title("Pred")

    fig_path = "%s/%s.png" % (fig_dir, design)

    bgcolor = 'white'
    ax1.patch.set_facecolor(bgcolor)
    ax2.patch.set_facecolor(bgcolor)        
    fig.savefig(fig_path, dpi=150)

    #plt.show(block=False)
    _ = input()
    plt.clf()
    plt.close()



