from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, LSTM, CNN, MLP, BiLSTM, CLA
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


# 传统迁移学习
# class Exp_Main_Transfer(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Main_Transfer, self).__init__(args)
#
#     def _build_model(self):
#         model_dict = {
#             'Autoformer': Autoformer,
#             'Transformer': Transformer,
#             'Informer': Informer,
#             'LSTM': LSTM,
#             'CNN': CNN,
#             'MLP': MLP,
#             'BiLSTM': BiLSTM,
#             'CLA': CLA
#
#         }
#         model = model_dict[self.args.model].Model(self.args).float()
#
#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model
#
#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader
#
#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim
#
#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion
#
#     def vali(self, vali_data, vali_loader, criterion):
#         total_loss = []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()
#
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.model == 'LSTM' or self.args.model == 'CNN' or self.args.model == 'MLP' or self.args.model == 'BiLSTM' or self.args.model == 'CLA':
#                         outputs = self.model(batch_x)
#                         outputs = outputs[:, -self.args.pred_len:, :]
#                     else:
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
#
#                 if self.args.features == 'MS':
#                     target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
#                     set_target = target_map[self.args.target]
#                     f_dim = set_target
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim].unsqueeze(2).to(self.device)
#                 else:
#                     f_dim = 0
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#
#                 pred = outputs.detach().cpu()
#                 true = batch_y.detach().cpu()
#
#                 loss = criterion(pred, true)
#
#                 total_loss.append(loss)
#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss
#
#     def tunning_train(self, setting):
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')
#         path = os.path.join(self.args.checkpoints, setting)
#         print('loading model')
#         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
#
#         tunning_steps = len(vali_loader)
#         print('tunning_steps:{} '.format(tunning_steps))
#
#         model_optim = self._select_optimizer()  # 优化器
#         criterion = self._select_criterion()  # 损失函数
#
#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             tunning_loss = []
#             self.model.train()
#
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()
#
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#
#                 if self.args.model == 'LSTM' or self.args.model == 'CNN' or self.args.model == 'MLP' or self.args.model == 'BiLSTM' or self.args.model == 'CLA':
#                     outputs = self.model(batch_x)
#                     outputs = outputs[:, -self.args.pred_len:, :]
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
#
#                 if self.args.features == 'MS':
#                     target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
#                     set_target = target_map[self.args.target]
#                     f_dim = set_target
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim].unsqueeze(2).to(self.device)
#                 else:
#                     f_dim = 0
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 loss = criterion(outputs, batch_y)
#                 tunning_loss.append(loss.item())
#                 loss.backward()
#                 model_optim.step()
#
#             tunning_loss = np.average(tunning_loss)
#             test_loss = self.vali(test_data, test_loader, criterion)
#
#             print("Epoch: {0}, Steps: {1} | Tunning Loss: {2:.7f} Test Loss: {3:.7f}".format(
#                 epoch + 1, tunning_steps, tunning_loss, test_loss))
#
#         torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
#
#         return self.model
#
#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')
#         # 水质三维时空数据调整为二维数据
#
#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         time_now = time.time()
#
#         train_steps = len(train_loader)
#         print('train_steps:{} '.format(train_steps))
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
#
#         model_optim = self._select_optimizer()  # 优化器
#         criterion = self._select_criterion()  # 损失函数
#
#         if self.args.use_amp:  # 一般为False
#             scaler = torch.cuda.amp.GradScaler()
#
#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []
#
#             self.model.train()
#             epoch_time = time.time()
#
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#
#                 # print(i, batch_x.shape)
#                 # print(i, batch_y.shape)
#                 # print(i, batch_x_mark.shape)
#                 # print(i, batch_y_mark.shape)
#
#                 # batch_x.shape: 32*96*7 batch_y.shape: 32*144*7 batch_x_mark.shape: 32*96*4 batch_y_mark.shape: 32*144*4
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)
#
#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 # print(dec_inp.shape)
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#                         f_dim = -1 if self.args.features == 'MS' else 0
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                         loss = criterion(outputs, batch_y)
#                         train_loss.append(loss.item())
#                 else:
#                     if self.args.model == 'LSTM' or self.args.model == 'CNN' or self.args.model == 'MLP' or self.args.model == 'BiLSTM' or self.args.model == 'CLA':
#                         outputs = self.model(batch_x)
#                         outputs = outputs[:, -self.args.pred_len:, :]
#                     else:
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
#
#                     if self.args.features == 'MS':
#                         target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
#                         set_target = target_map[self.args.target]
#                         f_dim = set_target
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim].unsqueeze(2).to(self.device)
#                     else:
#                         f_dim = 0
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                     loss = criterion(outputs, batch_y)
#                     train_loss.append(loss.item())
#
#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()
#
#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     model_optim.step()
#
#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss = self.vali(vali_data, vali_loader, criterion)
#             # vali_loss = self.vali_train(vali_data, vali_loader, criterion, model_optim)
#             test_loss = self.vali(test_data, test_loader, criterion)
#
#             print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#                 epoch + 1, train_steps, train_loss, vali_loss, test_loss))
#             # print("Epoch: {0}, Steps: {1} |  Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
#             #     epoch + 1, train_steps, vali_loss, test_loss))
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#             adjust_learning_rate(model_optim, epoch + 1, self.args)
#
#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))
#
#         return self.model
#
#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
#
#         preds = []
#         trues = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#
#         self.model.eval()
#         iter_count = 0
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)
#
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.model == 'LSTM' or self.args.model == 'CNN' or self.args.model == 'MLP' or self.args.model == 'BiLSTM' or self.args.model == 'CLA':
#                         outputs = self.model(batch_x)
#                         outputs = outputs[:, -self.args.pred_len:, :]
#                     else:
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
#
#                 if self.args.features == 'MS':
#                     target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
#                     set_target = target_map[self.args.target]
#                     f_dim = set_target
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim].unsqueeze(2).to(self.device)
#                 else:
#                     f_dim = 0
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()
#
#                 pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
#                 true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
#
#                 preds.append(pred)
#                 trues.append(true)
#                 if i % 20 == 0:
#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
#
#         preds = np.array(preds)
#         trues = np.array(trues)
#         print('test shape:', preds.shape, trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         print('test shape:', preds.shape, trues.shape)
#
#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#
#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         print('mse:{}, mae:{}'.format(mse, mae))
#         f = open("result.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}'.format(mse, mae))
#         f.write('\n')
#         f.write('\n')
#         f.close()
#
#         np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         np.save(folder_path + 'pred.npy', preds)
#         np.save(folder_path + 'true.npy', trues)
#
#         return
#
#     def predict(self, setting, load=False):
#         pred_data, pred_loader = self._get_data(flag='pred')
#
#         if load:
#             path = os.path.join(self.args.checkpoints, setting)
#             best_model_path = path + '/' + 'checkpoint.pth'
#             self.model.load_state_dict(torch.load(best_model_path))
#
#         preds = []
#
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 pred = outputs.detach().cpu().numpy()  # .squeeze()
#                 preds.append(pred)
#
#         preds = np.array(preds)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#
#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#
#         np.save(folder_path + 'real_prediction.npy', preds)
#
#         return


# recurrent fine-tunning transfer learning

class Exp_Main_Transfer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_Transfer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'LSTM': LSTM,
            'CNN': CNN,
            'MLP': MLP,
            'BiLSTM': BiLSTM,
            'CLA': CLA

        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'LSTM' or self.args.model == 'CNN' or self.args.model == 'MLP' or self.args.model == 'BiLSTM' or self.args.model == 'CLA':
                        outputs = self.model(batch_x)
                        outputs = outputs[:, -self.args.pred_len:, :]
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                if self.args.features == 'MS':
                    target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
                    set_target = target_map[self.args.target]
                    f_dim = set_target
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim].unsqueeze(2).to(self.device)
                else:
                    f_dim = 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def vali_train(self, vali_data, vali_loader, criterion, model_optim):
        total_loss = []
        self.model.train()
        # self.model.eval()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder

            if self.args.model == 'LSTM' or self.args.model == 'CNN' or self.args.model == 'MLP' or self.args.model == 'BiLSTM' or self.args.model == 'CLA':
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

            if self.args.features == 'MS':
                target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
                set_target = target_map[self.args.target]
                f_dim = set_target
                batch_y = batch_y[:, -self.args.pred_len:, f_dim].unsqueeze(2).to(self.device)
            else:
                f_dim = 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        total_loss = np.average(total_loss)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # 水质三维时空数据调整为二维数据


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print('train_steps:{} '.format(train_steps))
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()  # 优化器
        criterion = self._select_criterion()  # 损失函数

        if self.args.use_amp:  # 一般为False
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # ''' 无迁移时

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                # print(i, batch_x.shape)
                # print(i, batch_y.shape)
                # print(i, batch_x_mark.shape)
                # print(i, batch_y_mark.shape)

                # batch_x.shape: 32*96*7 batch_y.shape: 32*144*7 batch_x_mark.shape: 32*96*4 batch_y_mark.shape: 32*144*4
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # print(dec_inp.shape)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model == 'LSTM' or self.args.model == 'CNN' or self.args.model == 'MLP' or self.args.model == 'BiLSTM' or self.args.model == 'CLA':
                        outputs = self.model(batch_x)
                        outputs = outputs[:, -self.args.pred_len:, :]
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    if self.args.features == 'MS':
                        target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
                        set_target = target_map[self.args.target]
                        f_dim = set_target
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim].unsqueeze(2).to(self.device)
                    else:
                        f_dim = 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()



            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # '''
            vali_loss = self.vali_train(vali_data, vali_loader, criterion, model_optim)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                 epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # print("Epoch: {0}, Steps: {1} |  Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
            #     epoch + 1, train_steps, vali_loss, test_loss))  #无迁移时

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        iter_count = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'LSTM' or self.args.model == 'CNN' or self.args.model == 'MLP' or self.args.model == 'BiLSTM' or self.args.model == 'CLA':
                        outputs = self.model(batch_x)
                        outputs = outputs[:, -self.args.pred_len:, :]
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                if self.args.features == 'MS':
                    target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
                    set_target = target_map[self.args.target]
                    f_dim = set_target
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim].unsqueeze(2).to(self.device)
                else:
                    f_dim = 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                '''
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                '''
                # if i % 20 == 0:
                #     visual(true.reshape(-1, true.shape[-1])[:,0], pred.reshape(-1, pred.shape[-1])[:,0], os.path.join(folder_path, str(i) + '.pdf'))
                #     np.save(os.path.join(folder_path, str(i) + '.pdf'), true.reshape(-1, true.shape[-1]))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        visual(trues.reshape(-1, trues.shape[-1])[0:49, 0], preds.reshape(-1, preds.shape[-1])[0:49, 0],
               os.path.join(folder_path, '0' + '.pdf'))
        np.save(os.path.join(folder_path, 'true' + '.npy'), trues.reshape(-1, trues.shape[-1]))
        np.save(os.path.join(folder_path, 'pred' + '.npy'), preds.reshape(-1, preds.shape[-1]))

        # np.save('F:\\water_data\\water.npy', water)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        ph_mae, ph_mse, ph_rmse,ph_mape, ph_mspe = metric(preds[:,:,0], trues[:,:,0])
        print('pH  mse:{}, mae:{}'.format(ph_mse, ph_mae))
        DO_mae, DO_mse, DO_rmse, DO_mape, DO_mspe = metric(preds[:, :,1], trues[:, :,1])
        print('DO  mse:{}, mae:{}'.format(DO_mse, DO_mae))
        COD_mae, COD_mse, COD_rmse, COD_mape, COD_mspe = metric(preds[:, :,2], trues[:, :,2])
        print('COD  mse:{}, mae:{}'.format(COD_mse, COD_mae))
        NH_mae, NH_mse, NH_rmse, NH_mape, NH_mspe = metric(preds[:, :,3], trues[:, :,3])
        print('NH  mse:{}, mae:{}'.format(NH_mse, NH_mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
