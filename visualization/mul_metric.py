import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from loda_data import ECGDataset
from tqdm import tqdm
import os
import pandas as pd

"""计算每一个类别的相应指标"""

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_to_np(list):
    """"将list转化为numpy数组"""
    for i in range(len(list)):
        if i == 0:
            b = np.array(list[i])
        else:
            b = np.append(b, np.array(list[i]))
    return b


def label_cal(model_name=None, data_loader=None, n=1):
    model_path = '/home/ubuntu/liuyuanlin/code/ECG/best_model/%s_%d_model.pt' % (model_name, n)
    net = torch.load(model_path)
    net.eval()
    pre_output = []
    label = []
    y_score = []
    test_data = tqdm(enumerate(data_loader), total=len(data_loader), disable=True)
    with torch.no_grad():
        for i, (x_test, label_test) in test_data:
            x_test = x_test.cuda()
            pre_test = net(x_test, 'test')
            # pre_test = net(x_test)
            pre_test = torch.softmax(pre_test, 1)
            y_score.append(pre_test.cpu().numpy())
            pre_test_label = pre_test.argmax(1)
            label.append(label_test.cpu().numpy())
            pre_output.append(pre_test_label.cpu().numpy())
    test_data.close()
    y_score = np.vstack(y_score)
    # label = np.vstack(label).flatten()
    label = list_to_np(label)
    # pre_output = np.vstack(pre_output).flatten()
    pre_output = list_to_np(pre_output)
    return label, pre_output, y_score


def class_change(data_class=0, ture_label=None, pre_label=None, score=None):
    label_class = np.zeros(len(ture_label))
    pre_label_class = np.zeros(len(pre_label))
    a = np.where(ture_label == data_class)
    b = np.where(pre_label == data_class)
    y_score = np.ones([score.shape[0], 2])
    y_score[:, 1] = score[..., data_class]
    y_score[..., 0] = np.ones_like(score.shape[0]) - score[..., data_class]
    for i in range(len(a)):
        label_class[a[i]] = 1
        pre_label_class[b[i]] = 1
    f1 = f1_score(label_class, pre_label_class)
    acc = accuracy_score(label_class, pre_label_class)
    pre = precision_score(label_class, pre_label_class)
    recall = recall_score(label_class, pre_label_class)
    auc = roc_auc_score(label_class, pre_label_class)
    return f1, acc, pre, recall, auc


def cal_avg_sd(mean, sd):
    mean_sd = []
    for i in range(len(mean)):
        mean_sd.append(f"{mean[i]}±{sd[i]}")
    return mean_sd


def metric_cal_avg_sd(metric=None):
    metric_all = []
    for i in metric.keys():
        mean = round(np.mean(metric[i]), 3)
        std = round(np.std(metric[i]), 3)
        mean_sd = f"{mean}±{std}"
        metric_all.append(mean_sd)
    return metric_all


# load data
# path = '/home/ubuntu/liuyuanlin/data/ECG/500'
#path = '/home/ubuntu/liuyuanlin/data/ECG/500_original'
path = '/home/ubuntu/liuyuanlin/data/ECG/example'
ECG = ECGDataset(path, frequency=250, time=30)

# x_test, y_test = ECG.test_loder()
# test_set = torch.utils.data.TensorDataset(x_test, y_test)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

x_train, y_train, x_val, y_val = ECG.data_loader(val_size=0.05, seed=11)
# x_train = x_train.unsqueeze(1)
# x_val = x_val.unsqueeze(1)
train_set = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

val_set = torch.utils.data.TensorDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
f1 = dict()
acc = dict()
pre = dict()
recall = dict()
auc = dict()

for j in range(9):
    f1[j] = []
    acc[j] = []
    pre[j] = []
    recall[j] = []
    auc[j] = []
for i in range(3):
    for j in range(9):
        label, pre_output, y_score = label_cal(model_name='GT_RE_NOblance_3', data_loader=val_loader, n=i + 1)
        f_score, accuracy, precison, re, auc_roc = class_change(data_class=j, ture_label=label, pre_label=pre_output,
                                                                score=y_score)
        f1[j].append(f_score)
        acc[j].append(accuracy)
        pre[j].append(precison)
        recall[j].append(re)
        auc[j].append(auc_roc)
f_score = metric_cal_avg_sd(f1)
accuracy = metric_cal_avg_sd(acc)
precison = metric_cal_avg_sd(pre)
re = metric_cal_avg_sd(recall)
auc_roc = metric_cal_avg_sd(auc)
calss_name = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
# saving metrics
res = {'class': calss_name,
       'acc': accuracy,
       'pre': precison,
       'recall': re,
       'f1': f_score,
       'auc': auc_roc,
       }
df = pd.DataFrame(res)
df.to_csv('/home/ubuntu/liuyuanlin/code/ECG/calss_metric_1.csv', index=False)
