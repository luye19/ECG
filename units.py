import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import numpy as np
from tqdm import tqdm
import torch


def plot_figure(X, p_label, path):
    plt.figure(dpi=150)
    plt.plot(X, 'b', label=p_label)
    plt.ylabel(p_label)
    plt.xlabel('iter_num')
    plt.legend()
    plt.savefig(path + p_label + '.jpg')


def metrics_cal(test_loader, net):
    net.eval()
    pre_output = []
    label = []
    y_score = []
    test_data = tqdm(enumerate(test_loader), total=len(test_loader), disable=True)
    with torch.no_grad():
        for i, (x_test, label_test) in test_data:
            x_test = x_test.cuda()
            pre_test = net(x_test)
            # pre_test = net(x_test, 'test')
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
    f1 = f1_score(label, pre_output, average='macro')
    acc = accuracy_score(label, pre_output)
    pre = precision_score(label, pre_output, average='macro')
    recall = recall_score(label, pre_output, average='macro')
    auc = roc_auc_score(label, y_score, average='macro', multi_class='ovo')
    return f1, acc, pre, recall, auc


def list_to_np(list):
    """"将list转化为打平后的numpy数组"""
    for i in range(len(list)):
        if i == 0:
            b = np.array(list[i])
        else:
            b = np.append(b, np.array(list[i]))
    return b
