import argparse
from data.MyDataset import Dataset_NGSIM
from models.Net import MyNet
import torch
import torch.nn as nn
import time
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='LOCAL_X FORECASTING')

# 命令行参数
parser.add_argument('--seq_len', type=int, default=30, help='滑动窗口大小')
parser.add_argument('--hidden_size', type=int, default=256, help='LSTM隐藏层特征维度')
parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
parser.add_argument('--dropout', type=float, default=0.05, help='droput')
parser.add_argument('--is_bidirectional', type=bool, default=False, help='是否使用双向LSTM')
parser.add_argument('--lr', type=float, default=0.0001, help='学习率大小')
parser.add_argument('--epoches', type=int, default=1, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=16, help='批量大小')

args = parser.parse_args()

# 选取设备CPU/GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 选取特征
feature_x = ["local_x(m)", "local_y(m)", "v_Vel(m)", "v_Acc(m)"]
feature_y = ["local_x(m)"]
feature = list(set(feature_x + feature_y))

# 建立模型，选择优化器，损失函数
model = MyNet(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)
criterion = nn.MSELoss()

# 加载数据集
train_dataset = Dataset_NGSIM('train', args.seq_len, feature_x, feature_y)
test_dataset = Dataset_NGSIM('test', args.seq_len, feature_x, feature_y)
train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=True, drop_last=True)

# 训练
print("-----start training-----")
totol_loss = []
for epoch in range(args.epoches):
    train_loss = []
    model.train()
    start = time.time()
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - start))
    train_loss = np.average(train_loss)
    totol_loss.append(train_loss)
    print("Epoch: {0} | Train Loss: {1:.7f}".format(epoch + 1, train_loss))

# 保存训练误差
folder_path = "./result/" + feature_y[0][:-3]
totol_loss = np.array(totol_loss)
np.save(folder_path + "/train_loss.npy", totol_loss)
print("-----end training-----")

# 测试
print("-----start testing-----")
model.eval()
preds = []
trues = []
std = test_dataset.scaler.std[feature_y].values
mean = test_dataset.scaler.mean[feature_y].values
for x, y in test_dataloader:
    x = x.to(device)
    y = y.to(device)
    pred = model(x)
    preds.append(pred.detach().cpu().numpy().squeeze())
    trues.append(y.detach().cpu().numpy().squeeze())


# 预测结果反归一化
preds = np.array(preds)
trues = np.array(trues)
print('test shape:', preds.shape, trues.shape)
preds = preds.reshape(-1)
trues = trues.reshape(-1)
loss = np.mean((preds - trues) ** 2)
print("test_loss: {}".format(loss))
preds = (preds * std) + mean
trues = (trues * std) + mean
print('test shape:', preds.shape, trues.shape)

# 保存预测结果
np.save(folder_path + "/pred.npy", preds)
np.save(folder_path + "/true.npy", trues)
print("-----end testing-----")
