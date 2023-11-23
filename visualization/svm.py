import os

import numpy as np
import sklearn.svm as svm
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score


from loda_data import ECGDataset
os.environ['CUDA_LAUNCH_BLOCKING'] = '5'

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    out_put = torch.tensor(out_put)
    return out


def metrics_cal(test_loader, net):
    #net.eval()
    pre_output = []
    label = []
    y_score = []
    test_data = tqdm(enumerate(test_loader), total=len(test_loader), disable=True)
    with torch.no_grad():
        for i, (x_test, label_test) in test_data:
            # x_test = x_test.cuda()
            pre_test = net.predict(x_test)
            pre_test = torch.tensor(pre_test)
            pre_output.append(pre_test)

    f1 = f1_score(label, pre_output, average='macro')
    acc = accuracy_score(label, pre_output)
    pre = precision_score(label, pre_output, average='macro')
    recall = recall_score(label, pre_output, average='macro')
    return f1, acc, pre, recall, auc


model_name = 'GT_RE_NOblance'
# path = '/home/ubuntu/liuyuanlin/data/ECG/example'
# path = '/home/ubuntu/liuyuanlin/data/ECG/500'
path = '/home/ubuntu/liuyuanlin/data/ECG/500_original'
model_path = '/home/ubuntu/liuyuanlin/code/ECG/best_model/GT_RE_NOblance_3_2_model.pt'

ECG = ECGDataset(path, frequency=250, time=30, exchange=False)
x, y, x_val, y_val = ECG.data_loader(val_size=0.2)
feature_train = out_feature(x, y, model_path)
feature_val = out_feature(x_val, y_val, model_path)

val_set = torch.utils.data.TensorDataset(feature_val, y_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=8)

model = svm.SVC(kernel="linear", decision_function_shape="ovo")
model.fit(feature_train, y)
pre_test = model.predict(feature_val)
f1, acc, pre, recall, auc = metrics_cal(val_loader, model)
