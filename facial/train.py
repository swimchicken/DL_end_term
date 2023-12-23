# build by swimchicken
# 23/12/23 data

# TODO 引入資料處理模組 and 深度學習模組

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, add
from tensorflow.keras.utils import plot_model

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import accuracy_score

# TODO 引入資料並預處理

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names = ['emotion', 'pixels', 'usage']

df = pd.read_csv('../facial/dataset/fer2013.csv', names=names, na_filter=False)
df = df.iloc[1:]


# print(df.head())


def get_train_data(data):
    train_data = data['pixels'].apply(lambda pixels: np.fromstring(pixels, sep=' ', dtype=np.float32))
    train_data = np.vstack(train_data).reshape(len(df), 48, 48, 1)
    return train_data


x = get_train_data(df)
y = df['emotion'].to_numpy().astype('int')

# print("第一章圖像資料: ", x[0])
# print("第一章類別資料: ", y[0])
# plt.imshow(x[0].reshape(48, 48))
# plt.show()

# 將訓練類別拆成訓練集跟測試集

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# 轉成one-hot編碼

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# print(y_train.shape)
# print(y_train[0])


# 引入時間模組
class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_time_start = None
        self.times = None

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.times.append(time.time() - self.epoch_time_start)


time_callback_vgg = TimeHistory()
time_callback_incep = TimeHistory()
time_callback_resid = TimeHistory()


# TODO 創建各模型


# vgg網路

def vgg_block(layer_in, n_filters, n_conv):
    # add convolutional layers
    for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(layer_in)
    # add max pooling layer
    layer_in = MaxPooling2D((2, 2), strides=(2, 2))(layer_in)
    return layer_in


# define model input
visible = Input(shape=(48, 48, 1))
# add vgg module
layer = vgg_block(visible, 64, 2)
# add vgg module
layer = vgg_block(layer, 128, 2)
# add vgg module
layer = vgg_block(layer, 256, 4)

layer = Flatten()(layer)
layer = Dense(7, activation='softmax')(layer)

model_vgg = Model(inputs=visible, outputs=layer)
model_vgg.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history_vgg = model_vgg.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),
                            callbacks=[time_callback_vgg])

# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, num_conv):
#         super(VGGBlock, self).__init__()
#         layers = []
#         for _ in range(num_conv):
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(nn.ReLU(inplace=True))
#             in_channels = out_channels
#         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         self.vgg_block = nn.Sequential(*layers)
#
#     def forward(self, layer):
#         return self.vgg_block(layer)
#
#
# class VGGNet(nn.Module):
#     def __init__(self):
#         super(VGGNet, self).__init__()
#         self.block1 = VGGBlock(48, 64, 2)
#         self.block2 = VGGBlock(64, 128, 2)
#         self.block3 = VGGBlock(128, 256, 4)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(256 * 6 * 6, 7)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, layer):
#         layer = self.block1(layer)
#         layer = self.block2(layer)
#         layer = self.block3(layer)
#         layer = self.flatten(layer)
#         layer = self.fc(layer)
#         layer = self.softmax(layer)
#         return layer
#
#
# # ===============================================================================
#
# class ResidualModule(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualModule, self).__init__()
#
#         self.merge_input = nn.Identity()
#
#         # Check if the number of filters needs to be increased
#         if in_channels != out_channels:
#             self.merge_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.activation = nn.ReLU()
#
#     def forward(self, x):
#         merge_input = self.merge_input(x)
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(conv1)
#         layer_out = conv2 + merge_input
#         layer_out = self.activation(layer_out)
#         return layer_out
#
#
# class ResidualNet(nn.Module):
#     def __init__(self):
#         super(ResidualNet, self).__init__()
#         self.residual_module = ResidualModule(1, 64)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(48 * 48 * 64, 7)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.residual_module(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         x = self.softmax(x)
#         return x
#
#
# model_vgg = VGGNet()
# model_res = ResidualNet()
#
# print(model_vgg)
# print(model_res)
#
# # TODO 訓練並預測模型
#
# # VGG訓練
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_vgg = VGGNet().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model_vgg.parameters(), lr=0.001)
#
# print("x_train-shape: ", x_train.shape)
# print("y_train-shape: ", y_train.shape)
#
# X_train_tensor = torch.Tensor(x_train).to(device)
# y_train_tensor = torch.LongTensor(y_train).to(device)
# X_test_tensor = torch.Tensor(x_test).to(device)
# y_test_tensor = torch.LongTensor(y_test).to(device)
#
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
#
# epochs = 10
#
# for epoch in range(epochs):
#     model_vgg.train()
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model_vgg(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#     model_vgg.eval()
#     with torch.no_grad():
#         all_predictions = []
#         all_labels = []
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model_vgg(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             all_predictions.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#         accuracy = accuracy_score(all_labels, all_predictions)
#         print(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {accuracy:.4f}')
