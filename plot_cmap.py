from util import *
import scipy
import argparse
import torch
import shap
import csv
import os

def plot_img(val, zoom, vmin, vmax, title, savedir='.'):
    img = scipy.ndimage.interpolation.zoom(val, [zoom, zoom])
    plt.imshow(img, interpolation='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.show(block=False)

    file_name = "%s/%s.png" % (savedir, title)
    plt.savefig(file_name, dpi=200)



inputs = [ 'cell_den', 'pin_den', 'rudy', 'lnet_rudy', 'gnet_rudy',\
        'snet_rudy', 'wire_den_rsmt', 'lnet_den_rsmt', 'gnet_den_rsmt',\
        'chan_den_rsmt', 'chan_den_v_rsmt', 'chan_den_h_rsmt','wire_den_egr1',\
        'wire_den_egr2', 'wire_den_egr3', 'wire_den_egr4', 'wire_den_egr5',\
        'wire_den_egr6', 'wire_den_egr7', 'wire_den_egr8','chan_den_egr1',\
        'chan_den_egr2', 'chan_den_egr3', 'chan_den_egr4', 'chan_den_egr5',\
        'chan_den_egr6', 'chan_den_egr7', 'chan_den_egr8', 'via_den_egr1',\
        'via_den_egr2', 'via_den_egr3', 'via_den_egr4', 'via_den_egr5',\
        'via_den_egr6', 'via_den_egr7', 'lnet_den_egr', 'gnet_den_egr',\
        'chan_den_egr', 'chan_den_v_egr', 'chan_den_h_egr','avg_terms',\
        'num_insts', 'num_terms', 'num_nets','num_gnets', 'num_lnets',\
        'clk_ratio', 'wns', 'tns']

files = os.listdir("DUMP/")
file_list = []
for file_ in files:
    if "prev" in file_:
        continue
    temp_name = file_.split("clk")[0]
    file_list.append(temp_name+"clk_0")

print(file_list)

'''
file_list = [\

    'GCELL_x7_fake_nova_12_25_26_10_util_0.50_ar_1.0_rmax_6_pdn_sparse_clk_0', \
    'GCELL_x7_fake_nova_12_25_28_10_util_0.50_ar_1.0_rmax_6_pdn_sparse_clk_0', \
    'GCELL_x7_fake_nova_12_25_30_09_util_0.45_ar_1.0_rmax_6_pdn_sparse_clk_0', \
    'GCELL_x7_fake_nova_12_35_27_06_util_0.50_ar_1.0_rmax_6_pdn_sparse_clk_0', \
    'GCELL_x7_fake_nova_12_35_28_09_util_0.50_ar_1.0_rmax_7_pdn_sparse_clk_0', \
    'GCELL_x7_fake_nova_12_40_30_07_util_0.60_ar_1.0_rmax_6_pdn_sparse_clk_0', \
    'GCELL_x7_fake_nova_12_40_30_08_util_0.55_ar_1.0_rmax_6_pdn_sparse_clk_0', \
    'GCELL_x7_fake_nova_test6_util_0.46_ar_1.0_rmax_6_pdn_sparse_clk_0', \
    'GCELL_x7_fake_nova_test6_util_0.50_ar_1.0_rmax_6_pdn_sparse_clk_0', \
    'GCELL_x7_fake_nova_test6_util_0.50_ar_1.0_rmax_7_pdn_sparse_clk_0'\
]
'''

for file_name in file_list:
    #file_name = file_list[1]
    y_true = torch.load("DUMP/%s.true" % file_name)
    y_pred = torch.load("DUMP/%s.pred" % file_name)
    y_cmap = torch.load("DUMP/%s.map" % file_name)
    #print(y_cmap.size())
    c, w, h = y_cmap.size()
    y_true = y_true.reshape((w,h))
    y_pred = y_pred.reshape((w,h))


    save_dir = "IMG/%s" % (file_name)
    create_dir(save_dir)
    
    print("true\n")
    print(y_true.shape)
    print("pred\n")
    print(y_pred.shape)

    plot_img(y_true, 8.0, 0, 3, 'true', save_dir)
    #_ = input()
    plot_img(y_pred, 8.0, 0, 3, 'pred', save_dir)
    #_ = input()

    summary = open("%s/summary.txt" % save_dir, 'w')

    summary.write("feature max avg\n")

    for i,title in enumerate(inputs):
        plot_img(y_cmap[i], 8.0, 0, 2000, title, save_dir)
        #_ = input()
        summary.write("%s %.1f %.1f\n" % (title, y_cmap[i].max(), y_cmap[i].mean()))
        print("%s max %.1f avg %.1f" % (title, y_cmap[i].max(), y_cmap[i].mean()))

    summary.close()

