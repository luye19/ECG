import numpy as np
import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from Model.resnet import basic_block, Resnet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score, auc
from Model.VGG import MLP, Vgg16_net
from loda_data import ECGDataset
import random
from GTN.RE_GTN import Re_GTN, BasicBlock1d
from GTN.DGTN_RE import DGTN_RE
from GTN.GTN_RE import GTN_RE
from Model.lstm import LSTM
from Model.Inception import inception
from Model.RESNET import resnet18
from units import plot_figure, metrics_cal
from Model import utils
from GTN.transformer import Transformer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=11, help='Random seed.')
parser.add_argument('--name', type=str, default='test', help='name of model.')
parser.add_argument('--bachsize', type=int, default=32, help='Number of bachsize.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--repeat_num', type=int, default=1, help='number of randomized experiments')
parser.add_argument('--interval', type=int, default=50, help='Iterates over the displayed spacing')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    acc = 0
    losses = 0
    val_data = tqdm(enumerate(val_loader), total=len(val_loader), disable=True)
    with torch.no_grad():
        for i, (x_val, label_val) in val_data:
            x_val = x_val.cuda()
            label_val = label_val.cuda()

            # pre_val = model(x_val)
            pre_val = model(x_val, 'test')
            pre_val_label = pre_val.argmax(1)

            loss = criterion(pre_val, label_val.long())
            # label_val_hot = torch.nn.functional.one_hot(label_val, num_classes=9)
            # loss = criterion(pre_val, label_val_hot.float())

            losses += loss.cpu().item()
            acc += accuracy_score(label_val.cpu().numpy(), pre_val_label.cpu().numpy())
    val_data.close()
    acc_val = acc / len(val_loader)
    acc_loss = losses / len(val_loader)

    return acc_val, acc_loss


def train(model, train_loader, optimizer, val_loader, epoch):
    model.train()
    losses = 0.
    loss_total = 0.
    correct = 0.
    total = 0.
    model.train()
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.BCEWithLogitsLoss().cuda()
    train_data = tqdm(enumerate(train_loader), total=len(train_loader), disable=True)
    for i, (x, label) in train_data:
        x = x.cuda()
        label = label.cuda()
        # pre_train = model(x)
        pre_train = model(x, 'train')
        loss = criterion(pre_train, label.long())
        # label_hot = torch.nn.functional.one_hot(label, num_classes=9)
        # loss = criterion(pre_train, label_hot.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.cpu().item()
        loss_total += loss.cpu().item()
        pre_train = pre_train.argmax(dim=1)
        a = (pre_train == label).squeeze().sum().cpu()
        correct += a.numpy()
        total += label.size(0)
        if (i + 1) % args.interval == 0:
            loss_mean = losses / args.interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] train_loss: {:.4f} train_acc:{:.4f}".format(
                epoch + 1,
                args.epochs,
                i, len(
                    train_loader), loss_mean, correct / total))
            losses = 0
    train_data.close()
    acc_train = correct / total
    loss_train = loss_total / len(train_loader)
    acc_val, loss_val = validate(val_loader, model, criterion)
    # pre_val = model(x_val)
    # pre_val_label = pre_val.argmax(1)
    # acc_val = accuracy_score(y_val.cpu().numpy(), pre_val_label.cpu().numpy())
    # loss_val = criterion(pre_val, y_val)
    return acc_train, loss_train, acc_val, loss_val


##### metrics save#####
num = []
acc_list = []
pre_list = []
recall_list = []
f1_list = []
auc_list = []


def main():
    # load data
    path = '/home/ubuntu/liuyuanlin/data/ECG/500'
    # path = '/home/ubuntu/liuyuanlin/data/ECG/500_original'
    path = '/home/ubuntu/liuyuanlin/data/ECG/example'
    ECG = ECGDataset(path, frequency=500, time=60, exchange=False)
    x_test, y_test = ECG.test_loader(seg=True)
    # x_test = x_test.unsqueeze(1)
    test_set = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bachsize, shuffle=False)
    log = pd.DataFrame(index=[], columns=[
        'experiment', 'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'
    ])
    # 10 randomized experiments
    for i in range(1, args.repeat_num + 1):
        best_acc = 0
        x_train, y_train, x_val, y_val = ECG.data_loader(val_size=0.3, seed=i + 10, seg=True)
        # x_train = x_train.unsqueeze(1)
        # x_val = x_val.unsqueeze(1)
        train_set = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bachsize, shuffle=True)

        val_set = torch.utils.data.TensorDataset(x_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.bachsize, shuffle=False)
        # model = Vgg16_net()
        # resblock = basic_block
        # model = Resnet(resblock, blockNums=[1, 1, 1, 1], nb_classes=9)
        # model = LSTM(embedding_dim=7500, hidden_size=512, num_classes=9, num_layers=1, bidirectional=False)
        # model = resnet18()
        # model = inception()
        # model = Transformer(d_model=256, d_input=3000, d_channel=12, d_output=9, d_hidden=512, q=1, v=1, h=1, N=1, dropout=args.dropout, pe=True, mask=True, device=DEVICE)
        # model = Re_GTN(BasicBlock1d, [2, 2, 2, 2], device=DEVICE)
        # model = GTN_RE(BasicBlock1d, [2, 2, 2, 2], device=DEVICE)
        model = DGTN_RE(BasicBlock1d, [2, 2, 2, 2], device=DEVICE)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        loss_tr = []
        loss_v = []
        acc_tr = []
        acc_v = []

        for epoch in range(args.epochs):
            print('Experiment [%d/%d] Epoch [%d/%d]' % (i, args.repeat_num, epoch + 1, args.epochs))
            acc_train, loss_train, acc_val, loss_val = train(model, train_loader, optimizer, val_loader, epoch)
            print('train_loss %.4f - train_acc %.4f - val_loss %.4f - val_acc %.4f'
                  % (loss_train, acc_train, loss_val, acc_val))

            # save data as csv file

            tmp = pd.Series([i, epoch + 1, args.lr, loss_train, acc_train, loss_val, acc_val],
                            index=['experiment', 'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])

            log = log.append(tmp, ignore_index=True)
            log.to_csv('/home/ubuntu/liuyuanlin/code/ECG/result/%s_log.csv' % args.name, index=False)

            if acc_val > best_acc:
                torch.save(model, '/home/ubuntu/liuyuanlin/code/ECG/best_model/%s_%d_model.pt' % (args.name, i))
                best_model_path = str('/home/ubuntu/liuyuanlin/code/ECG/best_model/%s_%d_model.pt' % (args.name, i))
                best_acc = acc_val
                print("=> saved best model for experiment %d" % i)

            loss_tr.append(loss_train)
            acc_tr.append(acc_train)
            loss_v.append(loss_val)
            acc_v.append(acc_val)
        best_model_path = '/home/ubuntu/liuyuanlin/code/ECG/best_model/test_1_model.pt'
        best_net = torch.load(best_model_path)
        f1, acc, pre, recall, auc = metrics_cal(test_loader, best_net)
        num.append(i)
        f1_list.append(f1)
        acc_list.append(acc)
        pre_list.append(pre)
        recall_list.append(recall)
        auc_list.append(auc)

        print("Experiment%d The best val_acc: %.4f" % (i, best_acc))
        # path = '/home/ubuntu/liuyuanlin/code/ECG/plot/' + args.name + '_ex' + str(i) + '_'
        # plot_figure(loss_tr, 'loss_train', path)
        # plot_figure(loss_v, 'loss_val', path)
        # plot_figure(acc_tr, 'acc_train', path)
        # plot_figure(acc_v, 'acc_val', path)

    # saving metrics
    res = {'num': num,
           'f1': f1_list,
           'acc': acc_list,
           'pre': pre_list,
           'recall': recall_list,
           'auc': auc_list, }
    df = pd.DataFrame(res)
    df.to_csv('/home/ubuntu/liuyuanlin/code/ECG/result/metrics_%s.csv' % args.name, index=False)


if __name__ == '__main__':
    main()
