import torch
import numpy as np
from sklearn.manifold import TSNE
import sklearn
import matplotlib.pyplot as plt
import os
from loda_data import ECGDataset
from tqdm import tqdm
import torch.nn as nn
import matplotlib as mpl

os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def colormap():
    color = ['#20B2AA', '#9370DB', '#87cefa', '#ffcc00', '#ffe5b4', '#ff4d40', '#d2691e', '#ccff00', '#ffb6c1']
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', color, 256)


def list_to_np(list):
    """"将list转化为numpy数组"""
    for i in range(len(list)):
        if i == 0:
            b = list[i].cpu().numpy()
        else:
            b = np.append(b, list[i].cpu().numpy(), axis=0)
    return b


def out_feature(x, y, path):
    data_set = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=10, shuffle=False)
    net = torch.load(path)
    net.eval()
    classifier = nn.Sequential()
    net.GTN.output_linear = classifier
    test_data = tqdm(enumerate(data_loader), total=len(data_loader), disable=True)
    model_layer = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
    out_put = []
    out = []
    with torch.no_grad():
        for i, (x_test, label_test) in test_data:
            x_test = x_test.cuda()
            x_out = net(x_test, 'test')
            out_put.append(x_out)
            for name, layer in net.named_modules():
                if name in model_layer:
                    x_test = layer(x_test)
            x_out = x_test.view(-1, 120320)
            out.append(x_out)
    out_put = list_to_np(out_put)
    out = list_to_np(out)
    return out_put, out


model_name = 'GTN_RE_NOblance'
# path = '/home/ubuntu/liuyuanlin/data/ECG/example'
path = '/home/ubuntu/liuyuanlin/data/ECG/500'
model_path = '/home/ubuntu/liuyuanlin/code/ECG/best_model/GTN_RE_NOblance_3_3_1_model.pt'
ECG = ECGDataset(path, frequency=250, time=30, exchange=False)
# x, y = ECG.test_loder()
x, y, x_val, y_val = ECG.data_loader(val_size=0.2)
# x1 = x.view(-1, 90000)
feature, x1 = out_feature(x, y, model_path)

X1_tsne = TSNE(n_components=2, random_state=33).fit_transform(x1)
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(feature)
cm = colormap()
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], s=10, c=y, cmap=cm)
plt.xlabel('(a)')
plt.subplot(212)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10, c=y, cmap=cm)
plt.xlabel('(b)')
plt.savefig(f'/home/ubuntu/liuyuanlin/code/ECG/plot/{model_name}_tsne.png', dpi=300)
plt.show()
