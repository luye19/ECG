import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from loda_data import ECGDataset
from tqdm import tqdm
from itertools import cycle
from units import list_to_np
from scipy.interpolate import interpn
import scipy
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

"""计算每个模型的fpr、tpr、roc_auc并存储在csv文件中"""

# load data
#path = '/home/ubuntu/liuyuanlin/data/ECG/500'
path = '/home/ubuntu/liuyuanlin/data/ECG/500_original'
# path = '/home/ubuntu/liuyuanlin/data/ECG/example'
ECG = ECGDataset(path, frequency=250, time=30)
# x_test, y_test = ECG.test_loder()
# test_set = torch.utils.data.TensorDataset(x_test, y_test)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
#
# x_test_1 = x_test.unsqueeze(1)
# test_set_1 = torch.utils.data.TensorDataset(x_test_1, y_test)
# test_loader_1 = torch.utils.data.DataLoader(test_set_1, batch_size=32, shuffle=False)


# x_train, y_train, x_val, y_val = ECG.data_loader(val_size=0.8, seed=11)
# x_train = x_train.unsqueeze(1)
# x_val = x_val.unsqueeze(1)
# train_set = torch.utils.data.TensorDataset(x_train, y_train)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
#
# val_set = torch.utils.data.TensorDataset(x_val, y_val)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)


def macro_auc(model_name, test_loader, num_classes):
    y_score = []
    y_label = []
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    model_path = '/home/ubuntu/liuyuanlin/code/ECG/best_model/%s_%d_model.pt' % (model_name, 1)
    net = torch.load(model_path)
    test_data = tqdm(enumerate(test_loader), total=len(test_loader), disable=True)
    with torch.no_grad():
        for i, (x_test, x_label) in test_data:
            x_test = x_test.cuda()
            x_label = x_label.cuda()
            # pre_test = net(x_test)
            pre_test = net(x_test, 'test')
            pre_test = torch.softmax(pre_test, 1)
            y_score.append(pre_test.cpu().numpy())
            x_label = torch.nn.functional.one_hot(x_label, num_classes=9)
            y_label.append(x_label.cpu().numpy())
    test_data.close()
    y_score = np.vstack(y_score)
    y_label = np.vstack(y_label)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    fp = all_fpr
    tp = mean_tpr
    roc = auc(fp, tp)
    roc = round(roc, 4)

    return fp, tp, roc


def muti_macro_auc(model_name, test_loader, num_classes, repeat_num):
    fpr = 0
    tpr = 0
    ROC = 0
    for i in range(repeat_num):
        model_path = '/home/ubuntu/liuyuanlin/code/AF/best_model/%s_%d_model.pt' % (model_name, i + 1)
        fp, tp, roc = macro_auc(model_name, test_loader, num_classes, model_path)
        fpr += fp
        tpr += tp
        ROC += roc
    fpr = fpr / repeat_num
    tpr = tpr / repeat_num
    ROC = ROC / repeat_num
    return fpr, tpr, ROC


# model_name = ['Res1d-18', 'lstm_7','GTN_RE_NOblance']
# model_name_1 = ['Resnet10', 'VGG16', 'Inception_2']
fpr = dict()
tpr = dict()
roc_auc = dict()
# for model_id in model_name:
#     fpr[model_id], tpr[model_id], roc_auc[model_id] = macro_auc(model_id, test_loader, num_classes=9)
#     dataframe = pd.DataFrame({'fpr': fpr[model_id], 'tpr': tpr[model_id], 'roc_auc': roc_auc[model_id]})
#     path = "/home/ubuntu/liuyuanlin/code/AF/ROC_AUC/%s.csv" % model_id
#     dataframe.to_csv(path, index=False, sep=',')
# for model_id in model_name_1:
#     fpr[model_id], tpr[model_id], roc_auc[model_id] = macro_auc(model_id, test_loader_1, num_classes=9)
#     dataframe = pd.DataFrame({'fpr': fpr[model_id], 'tpr': tpr[model_id], 'roc_auc': roc_auc[model_id]})
#     path = "/home/ubuntu/liuyuanlin/code/AF/ROC_AUC/%s.csv" % model_id
#     dataframe.to_csv(path, index=False, sep=',')


# ECG = ECGDataset(path, frequency=250, time=30, exchange=False)
# x_test, y_test = ECG.test_loder()
# test_set = torch.utils.data.TensorDataset(x_test, y_test)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)


x_train, y_train, x_val, y_val = ECG.data_loader(val_size=0.11)
val_set = torch.utils.data.TensorDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
model_id = "GTN_RE_NOblance_3_3"
fpr[model_id], tpr[model_id], roc_auc[model_id] = macro_auc(model_id, val_loader, num_classes=9)
dataframe = pd.DataFrame({'fpr': fpr[model_id], 'tpr': tpr[model_id], 'roc_auc': roc_auc[model_id]})
path = "/home/ubuntu/liuyuanlin/code/AF/ROC_AUC/%s.csv" % model_id
dataframe.to_csv(path, index=False, sep=',')
