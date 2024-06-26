#Fatemeh's PIONEER
import numpy as np
import random
from time import time
from copy import deepcopy
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


def binarize(arr):
    return np.where(arr < 0, -1, 1)

def encoding_rp(X_data, base_matrix, binary=False):
    enc_hvs = np.matmul(base_matrix, X_data.T)
    if binary:
        enc_hvs = binarize(enc_hvs)
    return enc_hvs.T

def max_match(enc_hv, class_hvs):
    predicts = np.matmul(enc_hv, class_hvs.T)
    return predicts.argmax(axis=enc_hv.ndim - 1)

#https://github.com/1adrianb/binary-networks-pytorch/blob/51bdeee64d3da6306aebe4f2464eebd778bf7a38/bnn/ops.py
class SignActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input

class RPHDC(nn.Module):
    def __init__(self, d, D, n_class):
        super(RPHDC, self).__init__()
        self.PROJ = torch.nn.Parameter(torch.randn(d, D))
        self.CLASS = torch.nn.Parameter(torch.randn(D, n_class))
        self.PROJ.requires_grad = True
        self.CLASS.requires_grad = True
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = torch.matmul(x, self.PROJ)
        if benchmark.replace('28x28', '') in ['mnist', 'fmnist', 'pamap2', 'cifar10_hog']: #pamap2?
            x = SignActivation.apply(x)
        x = self.dropout(x)
        x = torch.matmul(x, self.CLASS)
        return x

        
def train(model, epoch, optimizer):
    model.train()
    correct = 0.
    total = 0.
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
    train_acc = correct / total
    test_acc = -1
    test_loss = -1
    if test_loader != None:
        model.eval()
        test_loss = 0.
        correct = 0.
        total = 0.
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            test_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
        test_acc = correct / total
    print('Epoch: {0}| train accuracy: {1:.4f}| test loss: {2:.1f}| test accuracy: {3:.4f}'.format(epoch, train_acc, test_loss, test_acc))
    return train_acc.item(), test_acc.item()


def train_hd(enc_train, enc_test, EPOCH=100, shuffle=True, log=False):
    D = len(enc_train[0])
    class_hvs = np.zeros((max(y_train)+1, D))
    n = 0
    for i in range(EPOCH):
        pickList = np.arange(0, len(enc_train))
        if shuffle: np.random.shuffle(pickList)
        correct = 0
        for j in pickList:
            predict = max_match(enc_train[j], class_hvs)
            if predict != y_train[j]:
                class_hvs[predict] -= enc_train[j]
                class_hvs[y_train[j]] += enc_train[j]
            else:
                correct += 1
        acc_train = correct/len(enc_train)
        if log: print(i+1, 'acc_train %.4f' %acc_train)
        if i == 0:
            predict = max_match(enc_test, class_hvs)
            acc_test1 = sum(predict == y_test)/len(y_test)
        if acc_train == 1 or i == EPOCH - 1:
            predict = max_match(enc_test, class_hvs)
            acc_test = sum(predict == y_test)/len(y_test)
            break
    return acc_train, acc_test1, acc_test, class_hvs

#loading the dataset, setting parameters
parser = argparse.ArgumentParser()
parser.add_argument('-D', required=True)
parser.add_argument('-dataset', required=True)
args = parser.parse_args()
benchmark = args.dataset
D = int(args.D)

def get_dataset(benchmark, normalize=True):
    if 'mnist' in benchmark:
        benchmark += '28x28'
    path = '/home/eagle/research/datasets/{}.pickle'.format(benchmark)
    with open(path, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')
    X_train, y_train, X_test, y_test = dataset
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    if benchmark in ['mnist28x28', 'fmnist28x28', 'cifar10'] and normalize:
        X_train = X_train/255.
        X_test = X_test/255.
    if benchmark in ['cifar10']:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test  = X_test.reshape(X_test.shape[0], -1)
    if benchmark in ['pamap2', 'cardio2', 'cardio3'] and normalize:
        percentile_5 = np.percentile(X_train, 5)
        percentile_95 = np.percentile(X_train, 95)
        X_train = np.clip(X_train, percentile_5,  percentile_95)
        X_test = np.clip(X_test, percentile_5,  percentile_95)
        max_abs = np.max(np.abs(X_train))
        X_train = X_train / max_abs
        X_test = X_test / max_abs
    del dataset
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = get_dataset(benchmark)
d = len(X_train[0])
n_class = np.unique(y_train).size

X_train_T, y_train_T = torch.Tensor(X_train), torch.Tensor(y_train)
train_dataset = TensorDataset(X_train_T, y_train_T)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)

X_test_T,  y_test_T  = torch.Tensor(X_test),  torch.Tensor(y_test)
test_dataset  = TensorDataset(X_test_T,  y_test_T)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

#here the model is created 
criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = RPHDC(d, D, n_class).to(device)

lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
for epoch in range(1, 101):
    if epoch%25 == 0:
        lr = lr * 0.2
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_acc, test_acc = train(model, epoch, optimizer)


#HD with learned float parameters
PROJ = model.state_dict()['PROJ'].T.cpu().numpy()
B = deepcopy(PROJ) #B is the projection matrix you can use instead of initializing it with random numbers
#the following is for training that you don't need
# enc_train = encoding_rp(X_train, B, binary=benchmark.replace('28x28', '') in ['mnist', 'pamap2', 'cifar10_hog']) #pamap2?
# enc_test = encoding_rp(X_test, B, binary=benchmark.replace('28x28', '') in ['mnist', 'pamap2', 'cifar10_hog'])
# acc_train_fp, acc_test1_fp, acc_test_fp, class_hvs_fp = train_hd(enc_train, enc_test, EPOCH=50, log=False)

#HD with learned binary parameters
B = deepcopy(PROJ)  #B is the projection matrix you can use instead of initializing it with random numbers
B = np.where(B >= 0, 1, -1)

#the following is for training that you don't need

# enc_train2 = encoding_rp(X_train, B, binary=benchmark.replace('28x28', '') in ['mnist', 'pamap2', 'cifar10_hog']) #pamap2?
# enc_test2 = encoding_rp(X_test, B, binary=benchmark.replace('28x28', '') in ['mnist', 'pamap2', 'cifar10_hog'])
# acc_train_b, acc_test1_b, acc_test_b, class_hvs_b = train_hd(enc_train, enc_test, EPOCH=50, log=False)

# print(train_acc, '\t', test_acc, '\t', acc_train_fp, '\t', acc_test1_fp, '\t', acc_test_fp, '\t\t\t', acc_train_b, '\t', acc_test1_b, '\t', acc_test_b)