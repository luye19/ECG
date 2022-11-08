import torch
import numpy as np
from loda_data import ECGDataset
from tqdm import tqdm
import os
import torch.nn as nn

os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""对Resnet-1d提取到的特征进行可视化"""

model_path = '/home/ubuntu/liuyuanlin/code/ECG/best_model/GTN_RE_NOblance_3_3_1_model.pt'
path = '/home/ubuntu/liuyuanlin/data/ECG/example'
ECG = ECGDataset(path, frequency=250, time=30, exchange=False)
x_test, y_test = ECG.test_loder()
# # x_test = x_test.unsqueeze(1)
test_set = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
net = torch.load(model_path)
net.eval()
test_data = tqdm(enumerate(test_loader), total=len(test_loader), disable=True)
output = dict()
model_layer = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
with torch.no_grad():
    for i, (x, label_test) in test_data:
        x = x.cuda()
        for name, layer in net.named_modules():
            if name in model_layer:
                print(name)
                x = layer(x)

test_data.close()
x1 = x.cpu().numpy()
x1 = np.squeeze(x1, 0)
np.savetxt('feature.txt', x1)

