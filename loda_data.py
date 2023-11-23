import numpy as np
import os
import pandas as pd
import torch
from scipy import signal

np.random.seed(11)


def load_data(path):
    x = []
    # path = '/home/ubuntu/liuyuanlin/code/ECG/Dataset_exa'
    txt_name = os.listdir(path)
    txt_name.sort()
    for txt_item in txt_name:
        if txt_item.endswith('.txt') and (not txt_item.startswith('._')):
            print(txt_item)

            record_path = os.path.join(path, txt_item)
            # x = sio.loadmat(record_path)['ECG'][0][0][2][:, -72000:].T
            data = np.loadtxt(record_path)
            if data.shape[1] < 15000:
                data = np.pad(data, ((0, 0), (0, 15000 - data.shape[1])), 'constant',
                              constant_values=(0, 0))  # 用0补齐记录时间不足30s的数据
                data = data.astype(np.float16)
                x.append(data)
            else:
                data_1 = data
                data = np.random.rand(12, 15000)
                for i in range(data.shape[0]):
                    data[i] = np.delete(data_1[i], np.arange(15000, data_1.shape[1]), axis=0)
                x.append(data)
    x = np.asanyarray(x)
    # x = x[::-1].copy()
    x = torch.tensor(x)
    x = x.to(torch.float32)
    inform = pd.read_csv('/home/ubuntu/liuyuanlin/code/ECG/label_ecg.CSV', header=0, encoding='gbk')
    label = inform['label']
    label = np.array(label)
    label = torch.tensor(label)
    return x, label


def load_data_exa(path):
    x = []
    txt_name = os.listdir(path)
    txt_name.sort()
    # path = '/home/ubuntu/liuyuanlin/code/ECG/Dataset_exa'
    for txt_item in txt_name:
        if txt_item.endswith('.txt') and (not txt_item.startswith('._')):
            print(txt_item)

            record_path = os.path.join(path, txt_item)
            # x = sio.loadmat(record_path)['ECG'][0][0][2][:, -72000:].T
            data = np.loadtxt(record_path)
            if data.shape[1] < 7500:
                data = np.pad(data, ((0, 0), (0, 7500 - data.shape[1])), 'constant',
                              constant_values=(0, 0))  # 用0补齐记录时间不足30s的数据
                # data = data.astype(np.float16)
                x.append(data)
            else:
                data_1 = data
                data = np.random.rand(12, 7500)
                for i in range(data.shape[0]):
                    data[i] = np.delete(data_1[i], np.arange(15000, data_1.shape[1]), axis=0)
                x.append(data)
    x = np.asanyarray(x)
    # x = x[::-1].copy()
    x = torch.tensor(x)
    x = x.to(torch.float32)
    # x = x.type(torch.LongTensor)
    inform = pd.read_csv('/home/ubuntu/liuyuanlin/code/ECG/label_ecg_exa.CSV', header=0, encoding='gbk')
    label = inform['label']
    label = np.array(label)
    label = torch.tensor(label)
    return x, label


def data_split():
    path = '/home/ubuntu/liuyuanlin/code/ECG/Dataset'
    np.random.seed(11)
    # range_num = np.random.randint(0, high=6377, size=[637])
    range_num = np.random.choice(6377, size=637, replace=False)
    inform = pd.read_csv('/home/ubuntu/liuyuanlin/code/ECG/label_ecg.CSV', header=0, encoding='gbk')
    label_train = []
    label_test = []
    test_name = []
    train_name = []
    i = 0
    txt_name = os.listdir(path)
    txt_name.sort()
    for txt_item in txt_name:
        if txt_item.endswith('.txt') and (not txt_item.startswith('._')):
            print(txt_item)
            record_path = os.path.join(path, txt_item)
            data = np.loadtxt(record_path)
            if i in range_num:
                path_test = f'/home/ubuntu/liuyuanlin/data/ECG/500/testset/{txt_item}'
                np.savetxt(path_test, data, fmt='%f')
                label_test.append(inform.label.loc[i])
                test_name.append(txt_item)
            else:
                path_train = f'/home/ubuntu/liuyuanlin/data/ECG/500/trainset/{txt_item}'
                np.savetxt(path_train, data, fmt='%f')
                label_train.append(inform.label.loc[i])
                train_name.append(txt_item)
        i = i + 1
    dataframe = pd.DataFrame({'ID': train_name, 'label': label_train})
    dataframe.to_csv("/home/ubuntu/liuyuanlin/data/ECG/500/train_label.csv", index=False, sep=',')
    dataframe = pd.DataFrame({'ID': test_name, 'label': label_test})
    dataframe.to_csv("/home/ubuntu/liuyuanlin/data/ECG/500/test_label.csv", index=False, sep=',')


class ECGDataset:
    def __init__(self, path, frequency=500, time=60, exchange=False):
        self.path = path
        self.frequency = frequency  # 目标采样频率
        self.time = time
        self.HZ = 500  # 数据当前的频率
        self.exchange = exchange

    def test_loader(self, seg=False):
        """load test dataset"""
        x = []
        label = []
        path = os.path.join(self.path, 'testset')
        txt_name = os.listdir(path)
        txt_name.sort()
        inform = pd.read_csv(os.path.join(self.path, 'test_label.csv'), header=0, encoding='gbk')
        for (i, txt_item) in zip(range(len(txt_name)), txt_name):
            if txt_item.endswith('.txt') and (not txt_item.startswith('._')):
                print(txt_item)
                record_path = os.path.join(path, txt_item)
                # x = sio.loadmat(record_path)['ECG'][0][0][2][:, -72000:].T
                data = np.loadtxt(record_path)
                if seg:
                    # data = self._resample_(data)
                    data = self.segment(data)
                    label_data = np.ones(data.shape[0]) * inform.label.loc[i]
                else:
                    data = self._resample_(data)  # 数据重采样
                    data = self._unified_length_(data)  # 统一数据长度
                    label_data = inform.label.loc[i]
            x.append(data)
            label.append(label_data)
        x = np.asanyarray(x, dtype=float)
        if self.exchange:
            x = np.swapaxes(x, 1, 2)
        if seg:
            x = torch.tensor(self._list_to_np(x))
            label = torch.tensor(self._list_to_np(label))
        else:
            x = torch.tensor(x)
            label = torch.tensor(np.asanyarray(label))
        x = x.to(torch.float32)
        return x, label

    def data_loader(self, val_size=0.1, seed=11, seg=False):
        """load train and validation dataset"""
        label_train = []
        label_val = []
        x_train = []
        x_val = []
        i = 0
        path = os.path.join(self.path, 'trainset')
        txt_name = os.listdir(path)
        txt_name.sort()
        np.random.seed(seed)
        range_num = np.random.choice(len(txt_name), size=int(len(txt_name) * val_size), replace=False)
        inform = pd.read_csv(os.path.join(self.path, 'train_label.csv'), header=0, encoding='gbk')
        for txt_item in txt_name:
            if txt_item.endswith('.txt') and (not txt_item.startswith('._')):
                print(txt_item)
                record_path = os.path.join(path, txt_item)
                # x = sio.loadmat(record_path)['ECG'][0][0][2][:, -72000:].T
                data = np.loadtxt(record_path)
                # a = self._test_()
                if i in range_num:
                    if seg:
                        # data = self._resample_(data)
                        data = self.segment(data)
                        label_data = np.ones(data.shape[0]) * inform.label.loc[i]
                    else:
                        data = self._resample_(data)
                        data = self._unified_length_(data)
                        label_data = np.ones(1) * inform.label.loc[i]
                    x_val.append(data)
                    label_val.append(label_data)
                else:
                    if seg:
                        # data = self._resample_(data)
                        # data = self._transform_(data)
                        data = self.segment(data)
                        label_data = np.ones(data.shape[0]) * inform.label.loc[i]
                    else:
                        data = self._resample_(data)
                        data = self._transform_(data)
                        data = self._unified_length_(data)
                        label_data = np.ones(1) * inform.label.loc[i]
                    x_train.append(data)
                    label_train.append(label_data)
            i = i + 1

        if self.exchange:
            # x_train = torch.tensor(np.swapaxes(np.asanyarray(x_train), 1, 2))
            # x_val = torch.tensor(np.swapaxes(np.asanyarray(x_val), 1, 2))
            x_train = torch.tensor(np.swapaxes(self._list_to_np(x_train), 1, 2))
            x_val = torch.tensor(np.swapaxes(self._list_to_np(x_val), 1, 2))
        else:
            x_train = torch.tensor(np.asanyarray(x_train))
            x_val = torch.tensor(np.asanyarray(x_val))
            # x_train = torch.tensor(self._list_to_np(x_train))
            # x_val = torch.tensor(self._list_to_np(x_val))
        x_train = x_train.to(torch.float32)
        x_val = x_val.to(torch.float32)
        label_train = torch.tensor(self._list_to_np(label_train))
        label_val = torch.tensor(self._list_to_np(label_val))
        if val_size == 0:
            return x_train, label_train
        else:
            return x_train, label_train, x_val, label_val

    def classdata_loader(self, class_name=None, lead=None, seg=False):
        """class_name = 0 represent the first class, lead = 0 represent the first lead"""
        x = []
        label = []
        path = os.path.join(self.path, 'trainset')
        txt_name = os.listdir(path)
        txt_name.sort()
        inform = pd.read_csv(os.path.join(self.path, 'train_label.csv'), header=0, encoding='gbk')
        for j in class_name:
            class_index = inform[inform.label == j].index.tolist()
            # class_label = np.ones(len(class_index))*j
            # label_1.append(class_label)
            for i in class_index:
                txt_item = txt_name[i]
                print(txt_item)
                record_path = os.path.join(path, txt_item)
                data = np.loadtxt(record_path)
                data = data[lead]
                if seg:
                    data = self.segment(data)
                    label_data = np.ones(data.shape[0]) * j
                else:
                    data = self._resample_(data)
                    data = self._unified_length_(data, lead_num=len(lead))
                    label_data = np.ones(data.shape[0]) * j
                # self.normalization(data, txt_item) # 检查中存不存在某些导联全部为零的数据
                x.append(data)
                label.append(label_data)
        if seg:
            x = torch.tensor(self._list_to_np(x))
            label = torch.tensor(self._list_to_np(label))
        else:
            x = torch.tensor(np.asanyarray(x))
            label = torch.tensor(np.asanyarray(label))
        # x = x.to(torch.float32)
        return x, label

    def _unified_length_(self, data, lead_num=12):
        #N = self.frequency * self.time
        N = 9216
        if data.shape[1] <= N:
            data = np.pad(data, ((0, 0), (0, N - data.shape[1])), 'constant',
                          constant_values=(0, 0))  # 用0补齐记录时间不足30s的数据
        else:
            data_1 = data
            data = np.empty([lead_num, N])
            for i in range(data.shape[0]):
                data[i] = np.delete(data_1[i], np.arange(N, data_1.shape[1]), axis=0)
        return data

    def _scaling_(self, X, sigma=0.1):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
        myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
        return X * myNoise

    def _shift_(self, sig, interval=20):
        for col in range(sig.shape[1]):
            offset = np.random.choice(range(-interval, interval))
            sig[:, col] += offset / 1000
        return sig

    def _transform_(self, sig):
        if np.random.randn() > 0.7:
            sig = self._scaling_(sig)
        if np.random.randn() > 0.7:
            sig = self._shift_(sig)
        return sig

    def _resample_(self, sig):
        if self.frequency < self.HZ:
            sig = signal.resample_poly(sig, self.frequency, self.HZ, axis=1)
        return sig

    def _list_to_np(self, list):
        """"将list转化为numpy数组"""
        global b
        for i in range(len(list)):
            if i == 0:
                b = np.array(list[i])
            else:
                b = np.append(b, np.array(list[i]), axis=0)
        return b

    def normalization(self, x, txt):
        max = np.max(x, axis=1)
        min = np.min(x, axis=1)
        # max = np.expand_dims(max, axis=1)
        # min = np.expand_dims(min, axis=1)
        a = (max - min)
        if np.argmax(a == 0) != 0:
            print(txt)  # 筛查某一导联全为零的异常数据

    def segment(self, data):
        "分割长度=1s X 采样频率"
        x = []
        time_step = data.shape[1]
        N = time_step // self.frequency
        # print(N)
        for i in range(N):
            n = 0 + i * self.frequency
            m = self.frequency + i * self.frequency
            x.append(data[..., n:m])
        x = np.asanyarray(x)
        return x

    def _test_(self):
        a = np.random.randn()
        return a

# data_split()
