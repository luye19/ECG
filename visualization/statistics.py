import numpy as np
import pandas as pd
import os

"""计算每个模型的每个指标的均值和方差"""


def del_repeat(file_name, del_str, front=True):
    if front:
        for (i, name) in zip(range(len(file_name)), file_name):
            file_name[i] = name.lstrip(del_str)
    else:
        for (i, name) in zip(range(len(file_name)), file_name):
            file_name[i] = name.rstrip(del_str)
    return file_name


def cal_avg_sd(mean, sd):
    mean_sd = []
    for i in range(len(mean)):
        mean_sd.append(f"{mean[i]}±{sd[i]}")
    return mean_sd


f1_mean = []
acc_mean = []
pre_mean = []
recall_mean = []
auc_mean = []
f1_sd = []
acc_sd = []
pre_sd = []
recall_sd = []
auc_sd = []
model = []
path = '/home/ubuntu/liuyuanlin/code/ECG/result'
file_name = os.listdir(path)
file_name = del_repeat(file_name, '_log.csv', front=False)
file_name = del_repeat(file_name, 'metrics_', front=True)
file_name = del_repeat(file_name, '.csv', front=False)
file_name = set(file_name)
#file_name = ['GTN_15']
for model_name in file_name:
    model.append(model_name)
    fname = 'metrics_' + model_name + '.csv'
    model_path = os.path.join(path, fname)
    inform = pd.read_csv(model_path, header=0, encoding='gbk')
    f1_mean.append(round(np.mean(np.array(inform['f1'])), 3))
    acc_mean.append(round(np.mean(np.array(inform['acc'])), 3))
    pre_mean.append(round(np.mean(np.array(inform['pre'])), 3))
    recall_mean.append(round(np.mean(np.array(inform['recall'])), 3))
    auc_mean.append(round(np.mean(np.array(inform['auc'])), 3))
    f1_sd.append(round(np.std(np.array(inform['f1'])), 3))
    acc_sd.append(round(np.std(np.array(inform['acc'])), 3))
    pre_sd.append(round(np.std(np.array(inform['pre'])), 3))
    recall_sd.append(round(np.std(np.array(inform['recall'])), 3))
    auc_sd.append(round(np.std(np.array(inform['auc'])), 3))
f1 = cal_avg_sd(f1_mean, f1_sd)
acc = cal_avg_sd(acc_mean, acc_sd)
pre = cal_avg_sd(pre_mean, pre_sd)
recall = cal_avg_sd(recall_mean, recall_sd)
auc = cal_avg_sd(auc_mean, auc_sd)
# saving metrics
# res = {'model': model,
#        'f1_mean': f1_mean,
#        'f1_sd': f1_sd,
#        'acc_mean': acc_mean,
#        'acc_sd': acc_sd,
#        'pre_mean': pre_mean,
#        'pre_sd': pre_sd,
#        'recall_mean': recall_mean,
#        'recall_sd': recall_sd,
#        'auc_mean': auc_mean,
#        'auc_sd': auc_sd,
#        }
# df = pd.DataFrame(res)
# df.to_csv('/home/ubuntu/liuyuanlin/code/ECG/statistics.csv', index=False)


# saving metrics
res = {'model': model,
       'acc': acc,
       'pre': pre,
       'recall': recall,
       'f1': f1,
       'auc': auc,
       }
df = pd.DataFrame(res)
df.to_csv('/home/ubuntu/liuyuanlin/code/ECG/statistics.csv', index=False)
