import torch
import logging as log
import json

# 函数的作用是将一个批次的数据（包括输入数据和标签）移动到指定的计算设备（通常是 GPU 或 CPU）
def batch_data_to_device(data, device):
    batch_x, y = data
    y = y.to(device)

    seq_num, x = batch_x
    seq_num = seq_num.to(device)
    x_len = len(x[0])
    for i in range(0, len(x)):
        for j in range(0, x_len):
            x[i][j] = x[i][j].to(device)

    return [[seq_num, x], y]

def varible(tensor, gpu):
    if gpu >= 0:
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def save_checkpoint(state, track_list, filename):
    with open(filename + '.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename + '.model')


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr