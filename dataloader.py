import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import scipy.io
import torch
import random

from torchvision.datasets import MNIST

mm = MinMaxScaler()

class UCI(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'UCI.mat')

        self.Y = data['truelabel'][0, 0].astype(np.int64).flatten()

        self.V1 = data['data'][0, 0].T.astype(np.float32)
        self.V2 = data['data'][0, 1].T.astype(np.float32)
        self.V3 = data['data'][0, 2].T.astype(np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        y = self.Y[idx]

        return [torch.from_numpy(x1),
                torch.from_numpy(x2),
                torch.from_numpy(x3)], y, torch.tensor(idx, dtype=torch.long)

class BDGP(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'BDGP.mat')
        self.Y = data['Y'].astype(np.int32).flatten()
        self.V1 = data['X1'].astype(np.float32)
        self.V2 = data['X2'].astype(np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        y = self.Y[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], y, torch.tensor(idx, dtype=torch.long)

class Synthetic3D(Dataset):
    def __init__(self, path):

        data = scipy.io.loadmat(path + 'synthetic3d.mat')

        self.views = []
        X = data['X']
        if X.shape == (1, 3):
            for i in range(3):
                view_data = X[0][i].astype(np.float32)  # (N, D_i)
                self.views.append(view_data)
        elif X.shape == (3, 1):
            for i in range(3):
                view_data = X[i][0].astype(np.float32)
                self.views.append(view_data)
        else:
            raise ValueError(f"Unexpected shape of data['X']: {X.shape}")
        Y = data['Y']
        if Y.ndim == 2:
            if Y.shape[0] == 1:
                self.Y = Y.flatten().astype(np.int32)
            elif Y.shape[1] == 1:
                self.Y = Y.flatten().astype(np.int32)
            else:
                self.Y = Y.astype(np.int32).flatten()
        else:
            self.Y = Y.astype(np.int32)

        self.num_samples = self.views[0].shape[0]

        for i, v in enumerate(self.views):
            assert v.shape[
                       0] == self.num_samples, f"View {i} has mismatched samples: {v.shape[0]} vs {self.num_samples}"

        assert len(self.Y) == self.num_samples, f"Label count {len(self.Y)} != sample count {self.num_samples}"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        items = [torch.from_numpy(v[idx]) for v in self.views]
        label = self.Y[idx]
        index = torch.tensor(idx).long()
        return items, label, index

class Caltech101_20(Dataset):
    def __init__(self, path):
        """
        path: directory of dataset
        default file: path/6V_Caltech101_20.mat
        """
        data = scipy.io.loadmat(path + 'Caltech101_20.mat')

        # Load labels
        self.Y = data['Y'].astype(np.int32).flatten()

        # Load six views and flatten into vectors
        self.V1 = data['X1'].astype(np.float32).reshape(len(self.Y), -1)
        self.V2 = data['X2'].astype(np.float32).reshape(len(self.Y), -1)
        self.V3 = data['X3'].astype(np.float32).reshape(len(self.Y), -1)
        self.V4 = data['X4'].astype(np.float32).reshape(len(self.Y), -1)
        self.V5 = data['X5'].astype(np.float32).reshape(len(self.Y), -1)
        self.V6 = data['X6'].astype(np.float32).reshape(len(self.Y), -1)

    def __len__(self):
        """Return total number of samples"""
        return len(self.Y)

    def __getitem__(self, idx):
        """
        Return:
        [view1, view2, view3, view4, view5, view6], label, index
        """
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        y = self.Y[idx]

        return [
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            torch.from_numpy(x4),
            torch.from_numpy(x5),
            torch.from_numpy(x6)
        ], y, torch.tensor(idx, dtype=torch.long)

class ALOI100(Dataset):
    def __init__(self, path):
        """
               path: directory of dataset
               default file: path/3V_Fashion_MV.mat
               """
        data = scipy.io.loadmat(path + 'ALOI_100.mat')

        # Load labels
        self.Y = data['Y'].astype(np.int32).flatten()

        # Load three views and flatten into vectors
        self.V1 = mm.fit_transform(data['colorsim'].astype(np.float32).reshape(len(self.Y), -1))
        self.V2 = mm.fit_transform(data['haralick'].astype(np.float32).reshape(len(self.Y), -1))
        self.V3 = mm.fit_transform(data['HSB'].astype(np.float32).reshape(len(self.Y), -1))
        self.V4 = mm.fit_transform(data['RGB'].astype(np.float32).reshape(len(self.Y), -1))

    def __len__(self):
        """Return total number of samples"""
        return len(self.Y)

    def __getitem__(self, idx):
        """
        Return:
        [view1, view2], label, index
        """
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        y = self.Y[idx]

        return [
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.from_numpy(x3),
            torch.from_numpy(x4),
        ], y, torch.tensor(idx, dtype=torch.long)

class MNIST_USPS(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + "MNIST_USPS.mat")

        self.V1 = data['X1'].astype(np.float32)
        self.V2 = data['X2'].astype(np.float32)
        self.Y = data['Y'].astype(np.int64).flatten()

        # 拉平成二维特征
        self.V1 = self.V1.reshape(self.V1.shape[0], -1)  # (5000, 784)
        self.V2 = self.V2.reshape(self.V2.shape[0], -1)  # (5000, 28)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x1 = torch.from_numpy(self.V1[idx])
        x2 = torch.from_numpy(self.V2[idx])
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        idx = torch.tensor(idx, dtype=torch.long)

        return [x1, x2], y, idx

class BBC(Dataset):
    def __init__(self, path):

        data = scipy.io.loadmat(path + "BBC.mat")


        self.Y = data['truth'].astype(np.int64).flatten()


        self.V1 = data['data1'].toarray().T.astype(np.float32)
        self.V2 = data['data2'].toarray().T.astype(np.float32)
        self.V3 = data['data3'].toarray().T.astype(np.float32)
        self.V4 = data['data4'].toarray().T.astype(np.float32)


        print(self.V1.shape)  # (2000, 4659)
        print(self.V2.shape)  # (2000, 4633)
        print(self.V3.shape)  # (2000, 4665)
        print(self.V4.shape)  # (2000, 4684)
        print(self.Y.shape)   # (2000,)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        y = self.Y[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], y, torch.tensor(idx, dtype=torch.long)

class HandWritten(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+"HandWritten.mat")

        self.Y = data['Y'].flatten()

        X = data['X']

        self.V1 = mm.fit_transform(X[0,0].astype(np.float32))
        self.V2 = mm.fit_transform(X[0,1].astype(np.float32))
        self.V3 = mm.fit_transform(X[0,2].astype(np.float32))
        self.V4 = mm.fit_transform(X[0,3].astype(np.float32))
        self.V5 = mm.fit_transform(X[0,4].astype(np.float32))
        self.V6 = mm.fit_transform(X[0,5].astype(np.float32))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.V1[idx]),
            torch.from_numpy(self.V2[idx]),
            torch.from_numpy(self.V3[idx]),
            torch.from_numpy(self.V4[idx]),
            torch.from_numpy(self.V5[idx]),
            torch.from_numpy(self.V6[idx]),
        ], self.Y[idx], idx

class NGs(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'NGs.mat')
        self.Y = data['Y'].astype(np.int32).flatten()  # 展平为一维
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class CIFAR10(Dataset):
    def __init__(self, path, train=True):
        """
        加载经过处理的 cifar10.mat（含多视图和缺失信息）
        """
        data = scipy.io.loadmat(path + 'cifar10.mat')

        # === 1. 加载三个视图的特征（转置为 N×D）===
        self.views = []
        for i in range(3):
            feat_mat = data['data'][i, 0]  # shape: (D, N)
            if feat_mat.ndim != 2:
                raise ValueError(f"View {i} must be 2D, got {feat_mat.shape}")
            view_data = feat_mat.T.astype(np.float32)  # (N, D)
            self.views.append(view_data)

        N = self.views[0].shape[0]

        # === 2. 加载标签（取第一个视图）===
        labels_raw = data['truelabel'][0, 0]
        if labels_raw.ndim == 2:
            if labels_raw.shape[0] == 1:
                labels = labels_raw[0, :]
            elif labels_raw.shape[1] == 1:
                labels = labels_raw[:, 0]
            else:
                labels = labels_raw.flatten()
        else:
            labels = labels_raw

        if len(labels) != N:
            raise ValueError(f"Label count {len(labels)} != sample count {N}")
        self.labels = labels.astype(np.int32)

        # === 3. 加载缺失状态 ===
        # missing_status = data['MissingStatus']
        # if missing_status.shape != (N, 3):
        #     raise ValueError(f"MissingStatus shape {missing_status.shape} != ({N}, 3)")
        # self.missing_status = missing_status.astype(np.uint8)

        self.num_samples = N
        self.dims = [v.shape[1] for v in self.views]  # 如 [512, 512, 512]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        items = []
        # for i in range(3):
        #     if self.missing_status[idx, i] == 1:
        #         # 存在：返回真实特征
        #         feat = self.views[i][idx]  # (D,)
        #         items.append(torch.from_numpy(feat))
        #     else:
        #         # 缺失：返回零向量（关键！避免 None）
        #         D = self.views[i].shape[1]
        #         items.append(torch.zeros(D, dtype=torch.float32))

        for i in range(3):
            # 存在：返回真实特征
            feat = self.views[i][idx]  # (D,)
            items.append(torch.from_numpy(feat))

        label = int(self.labels[idx])  # 转为 Python int
        index = torch.tensor(idx, dtype=torch.long)

        # 返回三项，匹配 train.py 中的: for batch_idx, (xs, _, _) in enumerate(data_loader)
        return items, label, index

mv_path = "./data/"

def load_data(dataset):
    if dataset == "UCI":
        dataset = UCI(mv_path)
        dims = [240, 76, 6]
        view = 3
        data_size = len(dataset)
        class_num = 10

    elif dataset == "BBC":
        dataset = BBC(mv_path)
        dims = [4659, 4633, 4665,4684]
        view = 4
        data_size = len(dataset)
        class_num = 5
    elif dataset == "cifar10":
        dataset = CIFAR10(mv_path)  # 自动加载 ./data/cifar10.mat
        dims = dataset.dims  # 自动获取各视图维度，如 [512, 512, 512]
        view = 3
        data_size = len(dataset)  # 更安全的方式
        class_num = 10  # CIFAR-10 固定 10 类

    elif dataset == "MNIST_USPS":
        dataset = MNIST_USPS(mv_path)
        dims = [784,784]
        view = 2
        data_size = len(dataset)
        class_num = 10

    elif dataset == "Caltech-5V":
        view = 5
        dataset = Caltech5V(mv_path + "Caltech-5V",view)
        dims = [
            40,  # X1
            254,  # X2
            928,  # X3
            512,  # X4
            1984  # X5
        ]

        data_size = len(dataset)
        class_num = 7

    elif dataset == "synthetic3d":
        dataset = Synthetic3D(mv_path)
        dims = [3,3,3]
        view = 3
        data_size = 600
        class_num = 3

    elif dataset == "HandWritten":
        dataset = HandWritten(mv_path)
        dims = [240,76,216,47,64,6]
        view = 6
        data_size = len(dataset)
        class_num = 10


    elif dataset == "BDGP":
        dataset = BDGP(mv_path)
        dims = [1750,79]
        view = 2
        data_size = 2500
        class_num = 5

    elif dataset == 'Caltech101_20':  # Caltech101_20
        dataset = Caltech101_20(mv_path)
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = len(dataset)
        class_num = 20

    elif dataset == 'ALOI_100':
        dataset = ALOI100(mv_path)
        dims = [77, 13, 64, 125]
        view = 4
        data_size = len(dataset)
        class_num = 100
    elif dataset == "NGs":
        dataset = NGs(mv_path)
        dims = [2000, 2000, 2000]
        view = 3
        data_size = 500
        class_num = 5
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    return dataset, dims, view, data_size, class_num

class Caltech5V(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()