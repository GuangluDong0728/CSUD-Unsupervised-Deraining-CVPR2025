import os
import torch
import glob
import time
import shutil
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from torchvision.utils import save_image,make_grid
from Utils.Gloab import mean_std


__all__ = ['MySaver']

class MySaver():
    def __init__(self,directory, exp_name):#path_weights
        self.mean, self.std = mean_std(dataset='imagenet')
        timee = time.strftime("%Y%m%d-%H%M%S")

        path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        self.path_image = os.path.join(path, rf'experiments{os.sep}{directory}{os.sep}{exp_name}{os.sep}{timee}{os.sep}images')
        self.path_record = os.path.join(path, rf'experiments{os.sep}{directory}{os.sep}{exp_name}{os.sep}{timee}')
        self.path_weights = os.path.join(path, rf'experiments{os.sep}{directory}{os.sep}{exp_name}{os.sep}{timee}{os.sep}weights')

        if not os.path.exists(self.path_image):
            os.makedirs(self.path_image)
        if not os.path.exists(self.path_weights):
            os.makedirs(self.path_weights)

    def save_record_train(self,epoch, train_loss):
        with open(os.path.join(self.path_record, 'train.txt'), 'a+') as f:
            f.write('{} epoch:{} train_loss:{} \n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), epoch,train_loss))

    def save_record_val(self,epoch, lr, test_loss, avg_psnr, avg_ssim,niqe):
        with open(os.path.join(self.path_record, 'val.txt'), 'a+') as f:
            f.write('{} epoch:{} lr:{} val_loss:{:.4f} psnr:{} ssim:{} niqe:{} \n'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), epoch, lr, test_loss, avg_psnr, avg_ssim, niqe))

    def save_edge_modulate(self,fea, epoch, iter, save_iter, flag='edge'):
        if iter % save_iter == 0:
            fea_ = fea[0, :, :, :].unsqueeze(0).permute(1, 0, 2, 3)
            # fea,_ = torch.max(fea,dim=1,keepdim=True)#.permute(1,0,2,3)  #b,1,h,w -> 1,b,h,w
            save_image(fea_, os.path.join(self.path_image, f'{epoch}_{iter}_{flag}.png'))

    def save_features(self,features, epoch, iter, save_iter, flag='cp', save_path=None):
        '''
        :param features: Tensor [0.,1.] BCHW
        :return:
        '''
        if iter % save_iter == 0:
            fea = features[0, ...].data.cpu().unsqueeze(0).permute(1, 0, 2, 3)  # [C,1,H,W]
            fea_vis = make_grid(fea, padding=0)
            fea_vis = single2uint(fea_vis.numpy().transpose(1, 2, 0))
            fea_vis = cv2.applyColorMap(fea_vis, colormap=cv2.COLORMAP_JET)

            cv2.imwrite(os.path.join(self.path_image, f'{epoch}_{iter}_fea_{flag}.png'), fea_vis)

    def save_checkpoint_interval(self,state,epoch,new_pred,best_pred,interval=10):
        filename = 'checkpoint_{}_{:.3f}.pth'.format(epoch, state['pred'])
        filepath = os.path.join(self.path_weights, f'{interval}_weights')

        if epoch % interval == 0:
            state['best_pred'] = state['pred'] if state['pred'] > best_pred else best_pred
            torch.save(state, os.path.join(filepath, filename))
        if state['pred'] > best_pred:
            state['best_pred']=state['pred']
            torch.save(state, os.path.join(self.path_weights, 'best.pth'))

    def save_checkpoint_override(self,state,epoch,new_pred,best_pred,args,print_=True):
        filename = f'checkpoint_{args.model}_{args.patch_size[0]}.pth'
        filepath = self.path_weights

        state['best_pred'] = state['pred'] if state['pred'] > best_pred else best_pred
        torch.save(state, os.path.join(filepath, filename))
        if state['pred'] > best_pred:
            # state['best_pred']=state['pred']
            torch.save(state, os.path.join(self.path_weights, 'best.pth'))
        if print_:
            print(f'checkpoint save at {self.path_weights}')

    def save_configs(self,args, para, flops, h, w):
        yy = str(args)[10:-1]
        xx = yy.replace(',', '\n')
        with open(os.path.join(self.path_record, 'configs.txt'), 'a+') as f:
            f.write('{}\n {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), xx))
            f.write(f'Model Parameters:{para:,}, Flops:{flops:,} of input size:{h}x{w}')

    def save_sr(self,iter, save_iter, sr, epoch, denormal=False, flag='hr'):
        if iter % save_iter == 0:
            image_np = sr[0, :, :, :].data.cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
            if denormal:
                image_np = denormalize(image_np,mean=self.mean,std=self.std)
            image_np = np.uint8((image_np.clip(0, 1) * 255).round())
            out = Image.fromarray(image_np)
            out.save(os.path.join(self.path_image, f'{epoch}_{iter}_{flag}.png'))


def uint2single(img):
    # uint8 [0,255] -> float32 [0.,1.]
    return np.float32(img / 255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())


def denormalize(img,mean=(0,0,0),std=(1,1,1)):
    if type(img)==np.ndarray: #(h,w,c) numpy (0,1)
        img*=std
        img+=mean
    return img


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))