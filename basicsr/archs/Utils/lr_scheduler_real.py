##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
LR_Scheduler: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/utils/lr_scheduler.py
              https://github.com/Jiaming-Liu/pytorch-lr-scheduler/blob/master/lr_scheduler.py
Synchronized-BatchNorm-PyTorch: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
'''


import math
import time
import torch
import wandb
from bisect import bisect_right
from collections import Counter
from Utils.print_utils import print_info_message


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, quiet=False, step_gamma=0.1, milestones=[100,200], eta_min = 0):
        self.mode = mode
        self.quiet = quiet
        if not quiet:
            print_info_message('Using {} LR scheduler with warm-up epochs of {}!'.format(self.mode, warmup_epochs))
        if mode == 'step':
            assert lr_step
        self.step_gamma = step_gamma
        self.milestones = Counter(milestones)
        self.base_lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = (num_epochs - warmup_epochs) * iters_per_epoch
        self.eta_min = eta_min  # The minimum lr. Default: 0.

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.base_lr * 1.0 * T / self.warmup_iters
        elif self.mode == 'cos':
            T = T - self.warmup_iters
            # lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * T / self.total_iters * math.pi))
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + math.cos(1.0 * T / self.total_iters * math.pi))
        elif self.mode == 'poly':
            T = T - self.warmup_iters
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), 0.9)
        elif self.mode == 'step':
            lr = self.base_lr * (self.step_gamma ** (epoch // self.lr_step))
        elif self.mode == 'linear':
            T = T - self.warmup_iters
            lr = self.base_lr - (self.base_lr * (1.0 * T / self.total_iters))
        elif self.mode == 'mutlistep':
            milestones = list(sorted(self.milestones.elements()))
            lr = self.base_lr * (self.step_gamma ** (bisect_right(milestones, epoch)))
        else:
            raise NotImplemented
        if epoch > self.epoch and (epoch == 0 or best_pred > 0.0):
            if not self.quiet:
                print_info_message('==> {} Epoches {}, learning rate = {:.6f}, previous best = {:.4f}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),epoch,lr,best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate_single(optimizer, lr)

    """Incease the additional head LR to be 10 times"""
    def _adjust_learning_rate_mutli(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

    def _adjust_learning_rate_single(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

    def get_lr(self, optimizer):
        return [group['lr'] for group in optimizer.param_groups]

#torch.optim.lr_scheduler.ReduceLROnPlateau()

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# # 初始化
# ema = EMA(model, 0.999)
# ema.register()
#
# # 训练过程中，更新完参数后，同步update shadow weights
# def train():
#     optimizer.step()
#     ema.update()
#
# # eval前，apply shadow weights；eval之后，恢复原来模型的参数
# def evaluate():
#     ema.apply_shadow()
#     # evaluate
#     # model save
#     ema.restore()

if __name__ == '__main__':
    milestones=[20,30,40,50,60,10]
    milestones = Counter(milestones)
    print(milestones)   # Counter({20: 1, 30: 1, 40: 1, 50: 1, 60: 1, 70: 1})   # Counter({20: 2, 40: 1, 50: 1, 60: 1, 70: 1})
    milestones = list(sorted(milestones.elements()))
    print(milestones)   # [20, 30, 40, 50, 60, 70]                              # [20, 20, 40, 50, 60, 70]
    out=bisect_right(milestones, 30)
    print(out)


    scheduler=LR_Scheduler('multistep',0.5,30,10,milestones=[5,10,15,20,25,30])
    lr = scheduler(out,)
