import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Water_transfer(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='water.npy',
                 target='PH', scale=True, timeenc=0, freq='w'):
        # scale 表示是否归一化
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 52
            self.label_len = 26
            self.pred_len = 8
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.split = 2  #目标域数据集训练测试划分,可取{1.5，2，3，4，5}

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  # 均值标准差归一化
        df_raw = np.load(os.path.join(self.root_path,
                                          self.data_path))
        # df_raw [110,120,6]
        # df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))
        # 训练集、验证集、测试集分界线 比例为 6:2:2
        border1s = [0, 90, 90]
        border2s = [90, 120, 120]
        '''
        border1s = [0, 75, 75] #new_water
        border2s = [75, 95, 95] #new_water
        '''
        # border1s = [0, 91, 91]
        # border2s = [90, 92, 92] #绘制曲线图

        '''
        # 湖泊（太湖30：36，鄱阳湖47：51）➡湖泊（115）
        print(df_raw.shape)
        df_raw[:,[36,37,38,39,47,48,49,50],:]=df_raw[:,[47,48,49,50,36,37,38,39],:]
        border1s = [30, 115, 115]
        border2s = [40, 116, 116]
        # '''
        '''
        # 河流（淮河70：80）➡湖泊（115）
        border1s = [70, 115, 115]
        border2s = [80, 116, 116]
        # '''
        '''
        # 湖泊（太湖30：36，鄱阳湖47：51）➡河流（淮河95）
        df_raw[:, [36, 37, 38, 39, 47, 48, 49, 50], :] = df_raw[:, [47, 48, 49, 50, 36, 37, 38, 39], :]
        border1s = [30, 95, 95]
        border2s = [40, 96, 96]
        # '''
        '''
        # 河流（淮河70：80）➡河流（淮河95）
        border1s = [70, 95, 95]
        border2s = [80, 96, 96]
        # '''
        '''
        # 湖泊（太湖30：36，鄱阳湖47：51）➡河流（淮河94）
        df_raw[:, [36, 37, 38, 39, 47, 48, 49, 50], :] = df_raw[:, [47, 48, 49, 50, 36, 37, 38, 39], :]
        border1s = [30, 94, 94]
        border2s = [40, 95, 95]
        # '''
        '''
        # 河流（淮河70：80）➡河流（淮河91）
        border1s = [70, 91, 91]
        border2s = [80, 92, 92]
        # '''
        '''
        # 湖泊（太湖30：36，鄱阳湖47：51）➡河流（淮河93,94,95）
        df_raw[:, [36, 37, 38, 39, 47, 48, 49, 50], :] = df_raw[:, [47, 48, 49, 50, 36, 37, 38, 39], :]
        border1s = [30, 93, 93]
        border2s = [40, 96, 96]
        # '''
        '''
        # 湖泊（太湖30：36，鄱阳湖47：51）➡湖泊（115，116，117）
        print(df_raw.shape)
        df_raw[:,[36,37,38,39,47,48,49,50],:]=df_raw[:,[47,48,49,50,36,37,38,39],:]
        border1s = [30, 115, 115]
        border2s = [40, 118, 118]
        # '''
        '''
        # 河流（淮河70：80）➡湖泊（115，116，117）
        border1s = [70, 115, 115]
        border2s = [80, 118, 118]
        # '''
        '''
        # 河流（淮河70：80）➡河流（淮河93,94,95）
        border1s = [70, 93, 93]
        border2s = [80, 96, 96]
        # '''
        '''
        # 河流（淮河70：80）➡河流（海河103,104,105）
        border1s = [70, 103, 103]
        border2s = [80, 106, 106]
        # '''
        '''
        # 湖泊（太湖30：36，鄱阳湖47：51）➡河流（海河103,104,105）
        df_raw[:, [36, 37, 38, 39, 47, 48, 49, 50], :] = df_raw[:, [47, 48, 49, 50, 36, 37, 38, 39], :]
        border1s = [30, 103, 103]
        border2s = [40, 106, 106]
        # '''


        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[:, :, 0:4]  # 取PH、DO、COD、NH3-N [110,120,4]
            # print(cols_data)
            # df_data = df_raw[cols_data]
            # df_data.shape: (17420, 7)
            # print(df_data.shape)
        elif self.features == 'S':
            target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
            set_target = target_map[self.target]
            df_data = np.expand_dims(df_raw[:, :, set_target], 2)

        # 对三维时空数据进行均值标准差归一化
        if self.scale:
            # train_data = df_data[:, border1s[0]:border2s[0], :]
            data_tp = df_data.transpose(1, 0, 2)  # data_tp[120,110,4]
            trans_data = []
            '''
            新水质数据（95，110，4）的标准化
            for i in np.arange(95):  # new_water_point
                self.scaler.fit(data_tp[i])
                trans_data.append(self.scaler.transform(data_tp[i]).tolist())
            data = np.array(trans_data).transpose(1, 0, 2)  # data [110,120,4]
            np.save('.\data\ETT\\new_water_norm.npy', data)  # new_water
            '''
            # 老水质数据集（120，110，4）的标准化
            for i in np.arange(120):
                self.scaler.fit(data_tp[i])
                trans_data.append(self.scaler.transform(data_tp[i]).tolist())
            data = np.array(trans_data).transpose(1, 0, 2)  # data [110,120,4]
            np.save('.\data\ETT\water_norm.npy', data)
            # print(train_data)
            # 使用训练集的均值和标准差对整个数据集归一化
            # self.scaler.fit(train_data)
            # data = self.scaler.transform(df_data)
            # print(data)
        else:
            data = df_data

        # df_stamp = df_raw[['date']][border1:border2]
        # print(df_stamp)
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # print(df_stamp)
        # print(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            # 时间戳编码
            # print(df_stamp['date'].values)
            # timestamp = time_features(pd.to_datetime(np.arange(0,110),unit='W', origin=pd.Timestamp('2013-01-05')), freq='w')
            timestamp = time_features(pd.to_datetime(np.arange(0, 293), unit='W', origin=pd.Timestamp('2012-07-05')),
                                      freq='w')  # new_water
            timestamp = timestamp.transpose(1, 0)
            data_stamp = []
            for i in np.arange(120):
                data_stamp.append(timestamp.tolist())
            data_stamp = np.array(data_stamp).transpose(1, 0, 2)  # data_stamp[110,120,2]
            # data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            # data_stamp = data_stamp.transpose(1, 0)
            # print(data_stamp)
            # print(data_stamp.shape)

        self.data_x = data[:, self.border1:self.border2, :]
        self.data_y = data[:, self.border1:self.border2, :]
        self.data_stamp = data_stamp[:, self.border1:self.border2, :]

    def __getitem__(self, index):
        if self.set_type == 0:
            point_i = int(index / (len(self.data_x) - self.seq_len - self.pred_len + 1))
            s_begin = index % (len(self.data_x) - self.seq_len - self.pred_len + 1)
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            # seq_x为seq_len；seq_y为label_len + pred_len；mark表示时间编码
            data_x = self.data_x.transpose(1, 0, 2)  # 空间节点，时间节点，水质指标
            data_y = self.data_y.transpose(1, 0, 2)  # 空间节点，时间节点，水质指标
            data_stamp = self.data_stamp.transpose(1, 0, 2)
            seq_x = data_x[point_i][s_begin:s_end]
            seq_y = data_y[point_i][r_begin:r_end]
            seq_x_mark = data_stamp[point_i][s_begin:s_end]
            seq_y_mark = data_stamp[point_i][r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark

        elif self.set_type == 1:
            point_i = int(index / int((len(self.data_x) - self.seq_len - self.pred_len + 1)/self.split))
            s_begin = index % int((len(self.data_x) - self.seq_len - self.pred_len + 1)/self.split)
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            # seq_x为seq_len；seq_y为label_len + pred_len；mark表示时间编码
            data_x = self.data_x.transpose(1, 0, 2)  # 空间节点，时间节点，水质指标
            data_y = self.data_y.transpose(1, 0, 2)  # 空间节点，时间节点，水质指标
            data_stamp = self.data_stamp.transpose(1, 0, 2)
            seq_x = data_x[point_i][s_begin:s_end]
            seq_y = data_y[point_i][r_begin:r_end]
            seq_x_mark = data_stamp[point_i][s_begin:s_end]
            seq_y_mark = data_stamp[point_i][r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark

        elif self.set_type == 2:
            point_i = int(index / int((len(self.data_x) - self.seq_len - self.pred_len + 1) * (1-1/self.split)))
            s_begin = index % int((len(self.data_x) - self.seq_len - self.pred_len + 1) * (1-1/self.split)) + int((len(self.data_x) - self.seq_len - self.pred_len + 1) / self.split)
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            # seq_x为seq_len；seq_y为label_len + pred_len；mark表示时间编码
            data_x = self.data_x.transpose(1, 0, 2)  # 空间节点，时间节点，水质指标
            data_y = self.data_y.transpose(1, 0, 2)  # 空间节点，时间节点，水质指标
            data_stamp = self.data_stamp.transpose(1, 0, 2)
            seq_x = data_x[point_i][s_begin:s_end]
            seq_y = data_y[point_i][r_begin:r_end]
            seq_x_mark = data_stamp[point_i][s_begin:s_end]
            seq_y_mark = data_stamp[point_i][r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.seq_len - self.pred_len + 1)*(self.border2 - self.border1)
        elif self.set_type == 1:
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1)/self.split) * (self.border2 - self.border1)
        elif self.set_type == 2:
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1) * (1-1/self.split)) * (self.border2 - self.border1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Water(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='water.npy',
                 target='PH', scale=True, timeenc=0, freq='w'):
        # scale 表示是否归一化
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 52
            self.label_len = 26
            self.pred_len = 8
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  # 均值标准差归一化
        df_raw = np.load(os.path.join(self.root_path,
                                          self.data_path))
        # df_raw [110,120,6]
        # df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))
        # 训练集、验证集、测试集分界线 比例为 6:2:2
        border1s = [0, 80, 100]
        border2s = [80, 100, 120]
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[:, :, 0:4]  # 取PH、DO、COD、NH3-N [110,120,4]
            # print(cols_data)
            # df_data = df_raw[cols_data]
            # df_data.shape: (17420, 7)
            # print(df_data.shape)
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 对三维时空数据进行均值标准差归一化
        if self.scale:
            # train_data = df_data[:, border1s[0]:border2s[0], :]
            data_tp = df_data.transpose(1, 0, 2)  # data_tp[120,110,4]
            trans_data = []
            for i in np.arange(120):
                self.scaler.fit(data_tp[i])
                trans_data.append(self.scaler.transform(data_tp[i]).tolist())
            data = np.array(trans_data).transpose(1,0,2)  # data [110,120,4]
            # print(train_data)
            # 使用训练集的均值和标准差对整个数据集归一化
            # self.scaler.fit(train_data)
            # data = self.scaler.transform(df_data)
            # print(data)
        else:
            data = df_data

        # df_stamp = df_raw[['date']][border1:border2]
        # print(df_stamp)
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # print(df_stamp)
        # print(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            # 时间戳编码
            # print(df_stamp['date'].values)
            timestamp = time_features(pd.to_datetime(np.arange(0,110),unit='W', origin=pd.Timestamp('2013-01-05')), freq='w')
            timestamp = timestamp.transpose(1, 0)
            data_stamp = []
            for i in np.arange(120):
                data_stamp.append(timestamp.tolist())
            data_stamp = np.array(data_stamp).transpose(1, 0, 2)  # data_stamp[110,120,2]
            # data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            # data_stamp = data_stamp.transpose(1, 0)
            # print(data_stamp)
            # print(data_stamp.shape)

        self.data_x = data[:, self.border1:self.border2, :]
        self.data_y = data[:, self.border1:self.border2, :]
        self.data_stamp = data_stamp[:, self.border1:self.border2, :]

    def __getitem__(self, index):
        point_i = int(index / (len(self.data_x) - self.seq_len - self.pred_len + 1))
        s_begin = index % (len(self.data_x) - self.seq_len - self.pred_len + 1)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # seq_x为seq_len；seq_y为label_len + pred_len；mark表示时间编码
        data_x = self.data_x.transpose(1, 0, 2)
        data_y = self.data_y.transpose(1, 0, 2)
        data_stamp = self.data_stamp.transpose(1, 0, 2)
        seq_x = data_x[point_i][s_begin:s_end]
        seq_y = data_y[point_i][r_begin:r_end]
        seq_x_mark = data_stamp[point_i][s_begin:s_end]
        seq_y_mark = data_stamp[point_i][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1)*(self.border2 - self.border1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # scale 表示是否归一化
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  # 均值标准差归一化
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # 训练集、验证集、测试集分界线 比例为 6:2:2
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            # print(cols_data)
            df_data = df_raw[cols_data]
            # df_data.shape: (17420, 7)
            # print(df_data.shape)
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            # print(train_data)
            # 使用训练集的均值和标准差对整个数据集归一化
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            # print(data)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        # print(df_stamp)
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # print(df_stamp)
        # print(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            # 时间戳编码
            # print(df_stamp['date'].values)
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            # print(data_stamp)
            # print(data_stamp.shape)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # seq_x为seq_len；seq_y为label_len + pred_len；mark表示时间编码
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
