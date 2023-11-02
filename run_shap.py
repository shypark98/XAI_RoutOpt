from util import *
import scipy
import argparse
import torch
import shap

print(shap.__version__)

deviceCount = torch.cuda.device_count()
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, choices=range(deviceCount), help='number of gpu to use')
arg = parser.parse_args()
# Set device
if arg.gpu == None:
    device = torch.device('cpu')
else:
    device = torch.device("cuda:%d" % arg.gpu)


model_checkpoint = "./MODEL/model_05-06-17-27/step_59"
checkpoint = torch.load(model_checkpoint, map_location=device)

model_config = checkpoint['model_config']
gcell_size = model_config['gcell_size']
img_size = model_config['img_size']
in_channel = model_config['in_channel']
out_channels = model_config['out_channels']
num_blocks = model_config['num_blocks']
num_neurons = model_config['num_neurons']
hop = int((img_size-1)/2)
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

#file_list = list(os.listdir(flatten_dir))
file_list = [\
    'GCELL_x7_mpeg2_top_util_0.90_ar_1.0_rmax_6_pdn_sparse_clk_0',\
    'GCELL_x7_nova_util_0.80_ar_1.0_rmax_7_pdn_sparse_clk_0',\
    'GCELL_x7_tate_pairing_util_0.70_ar_1.0_rmax_7_pdn_sparse_clk_0',\
    'GCELL_x7_wb_conmax_top_util_0.80_ar_0.5_rmax_7_pdn_sparse_clk_0'\
    ]


fig_dir = "fig"
create_dir(fig_dir)


model = model.to(device)
DRVth = 3

batch_home = "%s/batch/x%d/train" % (data_home, gcell_size)
batches = os.listdir(batch_home)
rand_indices = np.random.choice(len(batches), 10)

print("rand_indicies : ", rand_indices)
x_ref = []
for i in rand_indices:
    file_name = batches[i]
    path = os.path.join(batch_home, file_name)
    x, y = torch.load(path)
    #print(x.shape)
    #print(y.shape)
    x_ref.append(x)

x_ref = torch.cat(x_ref, dim=0).to(device)

print(x_ref.shape)

e = shap.DeepExplainer(model, x_ref) 

for file_name in file_list:#tqdm(file_list):
    design = file_name
    print("Start to calculate SHAP ", file_name)
    data_path = os.path.join(flatten_dir, file_name)
    (x,y,w,h) = torch.load(data_path)
    print(x.shape)
    #print("x ", x)
    #print("y ", y)
    print("w ", w)
    print("h ", h)

    x = (x-x_min) / delim # x_max - x_min
    x = torch.nan_to_num(x)
    x = x.to(device)
    y = y.to(device)

    batch_size = 50

    #y_t = np.zeros((w*h))
    #y_p = np.zeros((w*h))
    y_t = y.clone().to(device) #golden data
    y_p = torch.zeros((w*h), dtype=torch.float).to(device)
    
    
    for start in range(0, len(x), batch_size):
        end = min(len(x), start+batch_size)
        x_part = x[start:end].detach().clone().to(device)
        #y_ = y[start:end].detach().clone().to(device)
        y_p[start:end] = torch.sigmoid(model(x_part)).detach().squeeze() #예측 결과를 sigmoid
        del x_part

    mask = (y_p > 0.5).bool() #예측 결과가 DRV hotspot인 지점 masking
    x_masked = x[mask,:,:,:].detach().clone().to(device)
    
    print(mask)
    print(mask.shape)
    #print("here", len(x_masked))
    #shap_values = e.shap_values(x_masked)
    #print(shap_values)
    c_map = torch.zeros((49,w,h), dtype=torch.float)#.to(device)
    for i, val in enumerate(tqdm(mask)):
        if val:
            x_mini = x[i,:,:,:].detach().clone().to(device)
            x_mini = x_mini.unsqueeze(0)
            print("x_mini : ", x_mini.shape)
            shap_values = e.shap_values(x_mini)
            print("hop : ", hop)
            #w_ = min(w, h)
            r = i % h
            c = int(i / h)
            c1 = max(0, c-hop)
            c2 = min(w, c+hop+1)
            r1 = max(0, r-hop)
            r2 = min(h, r+hop+1)
            #contribution[:,c1:c2,r1:r2] += shap_values[i,:,c-c1:c2-c,r-r1:r2-r]
            #c_map[:,c1:c2,r1:r2] += shap_values[0,:,hop-(c-c1):c2-c1,r-r1:r2-r1]

            i1 = hop - (c-c1)
            i2 = hop + (c2-c)
            j1 = hop - (r-r1)
            j2 = hop + (r2-r)
            print(i)
            print(r, c)
            print(i1, i2, j1, j2)
            print(c1, c2, r1, r2)
            
            print(c_map[:,c1:c2,r1:r2].shape)
            print(shap_values[0,:,i1:i2,j1:j2].shape)
            c_map[:,c1:c2,r1:r2] += shap_values[0,:,i1:i2,j1:j2]
            

    y_t = y_t.cpu()
    y_p = y_p.cpu()
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
    
    torch.save(y_t.reshape((w,h)), "DUMP_test/%s.true" % file_name)
    torch.save(y_p.reshape((w,h)), "DUMP_test/%s.pred" % file_name)
    torch.save(c_map, "DUMP_test/%s.map" % file_name)


    '''
    h_zoom = 8.0
    w_zoom = 8.0

    nrows = 1
    ncols = 3
    gs = GridSpec(nrows=nrows, ncols=ncols, width_ratios=[1,1,1], height_ratios=[1])
    fig = plt.figure(figsize=(21,7))
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    img_true = y_t.reshape((w,h))
    img_pred = y_p.reshape((w,h))
    img_cmap = c_map[0:,:,:]

    img_true = scipy.ndimage.interpolation.zoom(img_true, [w_zoom, h_zoom])
    img_pred = scipy.ndimage.interpolation.zoom(img_pred, [w_zoom, h_zoom])
    img_cmap = scipy.ndimage.interpolation.zoom(img_cmap, [w_zoom, h_zoom])

    im1 = ax1.imshow(img_true, interpolation='nearest', cmap='jet', vmin=0, vmax=3.0)
    im2 = ax2.imshow(img_pred, interpolation='nearest', cmap='jet', vmin=0, vmax=3.0)
    im3 = ax3.imshow(img_cmap, interpolation='nearest', cmap='jet', vmin=0, vmax=3.0)

    #fig.suptitle(key)
    ax1.set_title("True")
    ax2.set_title("Pred")
    ax3.set_title("Cmap")

    fig_path = "%s/%s.png" % (fig_dir, design)

    bgcolor = 'white'
    ax1.patch.set_facecolor(bgcolor)
    ax2.patch.set_facecolor(bgcolor)        
    ax3.patch.set_facecolor(bgcolor)        
    fig.savefig(fig_path, dpi=150)

    plt.show(block=False)
    #_ = input()
    #plt.clf()
    #plt.close()
    '''

