import numpy as np
import torch
import random
from collections import OrderedDict
import os
from Utils.print_utils import print_info_message
import time
from pprint import pprint


__all__ = ['set_seed', 'AverageMeter', 'convert_state_dict', 'delete_state_module',
           'get_time', 'display_lr', 'get_lr', 'multistep_lr', 'show_kv']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_weights(model,weights, device):
    ckpt=torch.load(weights, map_location=device)['state_dict']
    try:
        model_dict = model.state_dict()
        # head_tail = ['head.weight', 'head.bias', 'tail.0.weight', 'tail.0.bias', 'tail.2.weight', 'tail.2.bias'] and k not in head_tail
        overlap_={k: v for k, v in ckpt.items() if k in model_dict}
    except:
        model_dict = model.module.state_dict()
        overlap_={k: v for k, v in ckpt.items() if k in model_dict}

    model_dict.update(overlap_)
    print(f'{(len(overlap_) * 1.0/len(model_dict) * 100):.4f}% is loaded!')


def convert_state_dict(state_dict):
    "https://github.com/XU-GITHUB-curry/FBSNet/blob/main/utils/convert_state.py"
    """
    Converts a state dict saved from a dataParallel module to normal module state_dict inplace
    Args:
        state_dict is the loaded DataParallel model_state
    """
    state_dict_new = OrderedDict()
    # print(type(state_dict))
    for k, v in state_dict.items():
        # print(k)
        name = k[7:]  # remove the prefix module.
        # My heart is borken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
    return state_dict_new


def delete_state_module(weights):
    """
    From BasicSR
    """
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    # model.load_state_dict(weights_dict)
    return weights


def show_kv(weight):
    ckpt = torch.load(weight, map_location='cpu')['state_dict']
    print([k for k, v in ckpt.items()])


def weight_convert(old_weight, new_txt, rename=False):
    old = torch.load(old_weight, map_location='cpu')

    old_list = ['project_in_', 'dwconv_', 'project_out_', 'attn', 'channel_select', 'spatial_select',
           'dwconv5x5', 'ddwconv_spatial_1', 'ddwconv_spatial_2', 'conv1', 'act', 'proj1', 'proj2']

    new_list = ['linear_in', 'SAL', 'linear_out', 'DFFM', 'channel_expand', 'spatial_expand',
           'region', 'spatial_1', 'spatial_2', 'fusion', 'gate', 'proj_value', 'proj_query']

    # with open(new_txt) as f:
    #     new_list = [line.strip() for line in f.readlines()]
    # old_list = [k for k, v in old.items()]
    # pprint(old_list)

    print(f'old_len:{len(old_list)} | new_len:{len(new_list)}')
    assert len(old_list) == len(new_list)

    for i, olist in enumerate(old_list):
        for k in old['state_dict'].keys():
            k.replace(old_list[i], new_list[i])
            print(f'{old_list[i]} -> {new_list[i]}')

    torch.save(old, r'D:\Software\Professional\AWork\Paper\MyWorks\Codes\EIMN\Model\x2.pth')

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def display_lr(epoch, lr, best_pred):
    print_info_message('==> {} Epoches {}, learning rate = {:.6f}, previous best = {:.4f}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, lr, best_pred))


def get_lr(optimizer):
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    lr = optimizer.param_groups[0]['lr']
    '''
    print(optim.param_groups)
    A.
    optim=torch.optim.SGD(
        [{'params': model.head.parameters(), 'lr': 0.0001},
         {'params': model.body.parameters(), 'lr': 0.0001 * 10},
         {'params': model.tail.parameters(), 'lr': 0.0001 / 10}],
        lr=0.0001
    )
    
    optimizer.param_groups[0] | optimizer.param_groups[1] | optimizer.param_groups[2]:
    [
        {'params': [Parameter containing: tensor([], requires_grad=True), Parameter containing: tensor([], requires_grad=True)], 
        'lr': 0.0001, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None}
        {'params': [Parameter containing: tensor([], requires_grad=True), Parameter containing: tensor([], requires_grad=True)], 
        'lr': 0.001, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None}
        {'params': [Parameter containing: tensor([], requires_grad=True), Parameter containing: tensor([], requires_grad=True)], 
        'lr': 0.00001, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None}
    ]
    
    B.
    optim=torch.optim.SGD(
        model.parameters(),
        lr=0.0001
    )
    
    [{'params': [Parameter containing: tensor([], requires_grad=True), Parameter containing: tensor([], requires_grad=True), 
                Parameter containing: tensor([], requires_grad=True), Parameter containing: tensor([], requires_grad=True),
                Parameter containing: tensor([], requires_grad=True), Parameter containing: tensor([], requires_grad=True)], 
    'lr': 0.0001, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None}]
    '''
    return lr


def multistep_lr(now_epoch, optimizer, lr, reduce_rate=0.1):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * (reduce_rate ** (now_epoch // 10))




'''
get lr
1.self.lr_rate=self.optim_net.state_dict()['param_groups'][0]['lr']
2.self.lr_rate=self.optim_net.param_groups[0]['lr']
3.self.lr_rate=self.multisteplr.get_lr()
# if (now_epoch+1) % 10 == 0:
    for param_group in self.optim_net.param_groups:
        param_group["lr"] = opt['psnr_lr'] * (0.1 ** (now_epoch // 10))
    self.lr_rate=self.optim_net.state_dict()['param_groups'][0]['lr']
    self.lr_rate=self.optim_net.param_groups[0]['lr']
'''

if __name__ == '__main__':
    weight_convert(
        old_weight=r'D:\Software\Professional\AWork\Paper\MyWorks\Codes\EIMN\Model\x2_38.2433.pth',
        new_txt=r'C:\Users\18378\Desktop\EIMN_original.txt',
        rename=False
    )