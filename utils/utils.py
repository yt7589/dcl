import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class LossRecord(object):
    def __init__(self, batch_size):
        self.rec_loss = 0
        self.count = 0
        self.batch_size = batch_size

    def update(self, loss):
        if isinstance(loss, list):
            avg_loss = sum(loss)
            avg_loss /= (len(loss)*self.batch_size)
            self.rec_loss += avg_loss
            self.count += 1
        if isinstance(loss, float):
            self.rec_loss += loss/self.batch_size
            self.count += 1

    def get_val(self, init=False):
        pop_loss = self.rec_loss / self.count
        if init:
            self.rec_loss = 0
            self.count = 0
        return pop_loss


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def Linear(in_features, out_features, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

def draw_accs_curve():
    '''
    绘制在训练数据集、测试数据集精度曲线
    '''
    print('记录训练过程中精度变化曲线')
    with open('./logs/step.txt', 'r') as step_fd:
        steps = step_fd.read()
        steps = steps[:-1]
        steps = np.array([float(x) for x in steps.split(',')], np.float32)
    print(steps.dtype)
    with open('./logs/train_acc.txt', 'r') as train_fd:
        train_accs = train_fd.read()
        train_accs = train_accs[:-1]
        train_accs = np.array([float(x) for x in train_accs.split(',')], dtype=np.float32)
    with open('./logs/test_acc.txt', 'r') as test_fd:
        test_accs = test_fd.read()
        test_accs = test_accs[:-1]
        test_accs = np.array([float(x) for x in test_accs.split(',')], dtype=np.float32)
    print('steps:{0};'.format(steps))
    print('train_accs: {0};'.format(train_accs))
    print('test_accs: {0};'.format(test_accs))
    fig, ax = plt.subplots()
    ax.plot(steps, train_accs*100.0, label='train')
    ax.plot(steps, test_accs*100.0, label='test')
    ax.set_xlabel('steps')
    ax.set_ylabel('accuracy')
    ax.set_title("accuracy curve")
    ax.legend()
    plt.show()



class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)

        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)
