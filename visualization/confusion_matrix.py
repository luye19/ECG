from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
from loda_data import ECGDataset
from tqdm import tqdm
import numpy as np
import itertools
import os
import matplotlib

matplotlib.use('TkAgg')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

path = '/home/ubuntu/liuyuanlin/data/ECG/500'
# path = '/home/ubuntu/liuyuanlin/data/ECG/500_original'
# path = '/home/ubuntu/liuyuanlin/data/ECG/example'
ECG = ECGDataset(path, frequency=250, time=30)
x_test, y_test = ECG.test_loder()
test_set = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

x_test_1 = x_test.unsqueeze(1)
test_set_1 = torch.utils.data.TensorDataset(x_test_1, y_test)
test_loader_1 = torch.utils.data.DataLoader(test_set_1, batch_size=32, shuffle=False)


def con_matrix(model_name, data):
    y_score = []
    y_label = []
    model_path = '/home/ubuntu/liuyuanlin/code/ECG/best_model/%s_%d_model.pt' % (model_name, 1)
    net = torch.load(model_path)
    test_data = tqdm(enumerate(data), total=len(data), disable=True)
    with torch.no_grad():
        for i, (x_test, x_label) in test_data:
            x_test = x_test.cuda()
            x_label = x_label.cuda()
            #pre_test = net(x_test)
            pre_test = net(x_test, 'test')
            pre_test = torch.softmax(pre_test, 1)
            pre_test_label = pre_test.argmax(dim=1)
            y_score.append(pre_test_label.cpu().numpy())
            y_label.append(x_label.cpu().numpy())
    test_data.close()
    y_score = list_to_np(y_score)
    y_label = list_to_np(y_label)
    return y_score, y_label


def list_to_np(list):
    """"将list转化为打平后的numpy数组"""
    for i in range(len(list)):
        if i == 0:
            b = np.array(list[i])
        else:
            b = np.append(b, np.array(list[i]))
    return b


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap='Blues',  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True, model_name='Res1d-34'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 6), dpi=300)
    #    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=15)
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), size=15)
    plt.xlabel('Predicted label', size=15)
    plt.savefig(f'/home/ubuntu/liuyuanlin/code/ECG/plot/{model_name}.png', format='png', bbox_inches='tight')
    plt.show()


# model_name = 'VGG16'
# y_pred, y_true = con_matrix(model_name, test_loader_1)
# C = confusion_matrix(y_true, y_pred)
# plot_confusion_matrix(C, normalize=False, target_names=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
#                       title='Confusion Matrix', model_name=model_name)

# model_name = ['Res1d-18', 'lstm_7']
# model_name_1 = ['Resnet10', 'VGG16', 'Inception_2']

model_name = ['GT_RE_NOblance_3']

for model_id in model_name:
    y_pred, y_true = con_matrix(model_id, test_loader)
    C = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(C, normalize=False, target_names=['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE'],
                          title='Confusion Matrix', model_name=model_id)
# for model_id in model_name_1:
#     y_pred, y_true = con_matrix(model_id, test_loader_1)
#     C = confusion_matrix(y_true, y_pred)
#     plot_confusion_matrix(C, normalize=False, target_names=['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE'],
#                           title='Confusion Matrix', model_name=model_id)
