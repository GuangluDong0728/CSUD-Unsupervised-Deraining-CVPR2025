import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import cv2
# import kornia.utils as KU
import torch.nn as nn
from PIL import Image
import math
import numpy as np
from torchvision.transforms import transforms
import os

# img_path = '/root/autodl-tmp/unsupervised deraining/datasets/rebutall/test/input'
# targeet_path = '/root/autodl-tmp/unsupervised deraining/datasets/rebutall/test/gt'

# img_list = sorted(os.listdir(img_path))
# num_img = len(img_list)

# def psnr(pred, gt):

#     pred = pred.clamp(0, 1).cpu().numpy()
#     gt = gt.clamp(0, 1).cpu().numpy()
#     imdff = pred - gt
#     rmse = math.sqrt(np.mean(imdff ** 2))
#     if rmse == 0:
#         return 100
#     return 20 * math.log10(1.0 / rmse)

# def imsave(img, filename):
#     img = img.squeeze().cpu()
#     img = KU.tensor_to_image(img) * 255.
#     cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@MODEL_REGISTRY.register()
class UnsupervisedModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(UnsupervisedModel, self).__init__(opt)

        # define network
        self.net_t = build_network(opt['network_t'])
        self.net_t = self.model_to_device(self.net_t)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # self.print_network(self.net_d)

        # load pretrained models
        load_path1 = self.opt['path'].get('pretrain_network_t', None)
        if load_path1 is not None:
            param_key = self.opt['path'].get('param_key_t', 'params')
            self.load_network(self.net_t, load_path1, self.opt['path'].get('strict_load_t', True), param_key)
            # self.load_network(self.net_ir_s, load_path1, self.opt['path'].get('strict_load_ir', True), param_key)
            # self.net_ir.load_state_dict(torch.load(load_path1)["state_dict"])
            # self.net_ir_s.load_state_dict(torch.load(load_path1)["state_dict"])

        load_path2 = self.opt['path'].get('pretrain_network_g', None)
        if load_path2 is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path2, self.opt['path'].get('strict_load_g', True), param_key)

        load_path3 = self.opt['path'].get('pretrain_network_d', None)
        if load_path3 is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path3, self.opt['path'].get('strict_load_d', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_t.train()
        self.net_g.train()
        self.net_d.train()
   
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_t_ema = build_network(self.opt['network_t']).to(self.device)
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path1 = self.opt['path'].get('pretrain_network_t', None)
            load_path2 = self.opt['path'].get('pretrain_network_g', None)
            if load_path1 is not None:
                self.load_network(self.net_t_ema, load_path1, self.opt['path'].get('strict_load_t', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight

            if load_path2 is not None:
                self.load_network(self.net_g_ema, load_path2, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight

            self.net_g_ema.eval()
            self.net_t_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('bgm_opt'):
            self.cri_bgm = build_loss(train_opt['bgm_opt']).to(self.device)
        else:
            self.cri_bgm = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None


        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_t'].pop('type')
        self.optimizer_t = self.get_optimizer(optim_type, self.net_t.parameters(), **train_opt['optim_t'])
        self.optimizers.append(self.optimizer_t)

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data1, data2):
        self.lq = data1['lq'].to(self.device)
        if 'gt' in data2:
            self.gt = data2['gt'].to(self.device)

    def feed_val_data(self, data):
        self.lq_val = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt_val = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.optimizer_t.zero_grad()

        real_lr_A = self.lq
        real_hr_B = self.gt

        #baseline
        fake_lr_B = self.net_t(real_hr_B, real_lr_A)
        fake_hr_B = self.net_g(fake_lr_B.detach())
        fake_hr_A = self.net_g(real_lr_A.detach())
        fake_lr_A = self.net_t(fake_hr_A, real_lr_A)
        fake_lr_A2 = self.net_t(fake_hr_A, fake_lr_B)
        fake_lr_B2 = self.net_t(fake_hr_B, fake_lr_B)
        #self-reconstruction
        fake_hr_Bn1 = self.net_t(real_hr_B, real_hr_B)#约束生成器
        fake_hr_Bn2 = self.net_t(real_hr_B, fake_hr_A)#约束生成器和复原器
        fake_hr_Bn3 = self.net_t(real_hr_B, fake_hr_B)#约束复原器


        l_g_total = 0
        l_t_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(fake_hr_B, real_hr_B)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if self.cri_ssim:
                l_g_ssim= self.cri_ssim(fake_hr_B, real_hr_B)
                l_g_total += l_g_ssim
                loss_dict['l_g_ssim'] = l_g_ssim

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(fake_hr_B, real_hr_B)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            #self-reconstruction loss
            l_g_self = self.cri_pix(fake_hr_Bn3, real_hr_B) + self.cri_pix(fake_hr_Bn2, real_hr_B)
            l_g_total += l_g_self*0.5
            loss_dict['l_g_self'] = l_g_self*0.5
            l_t_self = self.cri_pix(fake_hr_Bn1, real_hr_B) + self.cri_pix(fake_hr_Bn2, real_hr_B)
            l_t_total += l_t_self*5
            loss_dict['l_t_self'] = l_t_self*5

            r1, g1, b1 = torch.split(real_hr_B, [1, 1, 1], dim=1)
            r2, g2, b2 = torch.split(fake_lr_B, [1, 1, 1], dim=1)
            # #RGB LOSS
            l_t_rgb = self.cri_pix(r1-r2, g1-g2) + self.cri_pix(g1-g2, b1-b2) + self.cri_pix(b1-b2, r1-r2)
            l_t_total += l_t_rgb*10
            loss_dict['l_t_rgb'] = l_t_rgb*10
            # gan loss
            adversarial_loss1 = self.cri_gan(self.net_d(fake_lr_B), True)
            adversarial_loss2 = self.cri_gan(self.net_d(fake_lr_A2), True)
            adversarial_loss3 = self.cri_gan(self.net_d(fake_lr_A), True)
            adversarial_loss4 = self.cri_gan(self.net_d(fake_lr_B2), True)
            l_t_gan = adversarial_loss1 + adversarial_loss2 + adversarial_loss3 + adversarial_loss4

            l_t_total += l_t_gan
            loss_dict['l_t_gan'] = l_t_gan

            l_t_total.backward(retain_graph=True)
            l_g_total.backward(retain_graph=True)
            self.optimizer_g.step()
            self.optimizer_t.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(real_lr_A)
        l_d_real = self.cri_gan(real_d_pred, True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(fake_lr_B.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq_val)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq_val)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_val_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt_val

            # tentative for out of GPU memory
            del self.lq_val
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq_val.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt_val.detach().cpu()
        return out_dict

    # def ceshi(self):
    #     print("laileao")
    #     device_ids = [i for i in range(torch.cuda.device_count())]
    #     net_g = nn.DataParallel(self.net_g, device_ids=device_ids)
    #     net_g.cuda()
    #     net_g.eval()
    #     transform = transforms.ToTensor()
    #     PSNR = 0
    #     for img in img_list:
    #         # print(img)
    #         image = Image.open(img_path + '/' + img).convert('RGB')
    #         target = Image.open(targeet_path + '/' + img).convert('RGB')
    #         image = transform(image)
    #         target = transform(target)
    #         image = image.cuda()
    #         target = target.cuda()
    #         [A, B, C] = image.shape
    #         image = image.reshape([1, A, B, C])
    #         [A, B, C] = target.shape
    #         target = target.reshape([1, A, B, C])
    #         with torch.set_grad_enabled(False):
    #             B,C,H,W = image.size()
    #             _, _, h_old, w_old = image.size()
    #             h_pad = (h_old // 16 + 1) * 16 - h_old
    #             w_pad = (w_old // 16+ 1) * 16 - w_old
    #             img_lq = torch.cat([image, torch.flip(image, [2])], 2)[:, :, :h_old + h_pad, :]
    #             img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
    #             pre = net_g(img_lq)
    #             pre= pre[:,:,:H,:W]
    #         psnr_out = psnr(pre, target)
    #         PSNR += psnr_out
    #     print("PSNR =", PSNR / num_img)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_t_ema'):
            self.save_network([self.net_t, self.net_t_ema], 'net_t', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_t, 'net_t', current_iter)
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        # self.save_training_state(epoch, current_iter)
