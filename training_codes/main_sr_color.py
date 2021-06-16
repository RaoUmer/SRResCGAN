################################
## Imports
################################
import os
import math
import torch
import numpy as np
import logging
from collections import OrderedDict
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn import init
import functools
from utils import util
from utils import utils_common
from utils.utils_common import psnr
from utils import utils_logger
from utils.utils_logger import timer
import argparse, random
from torch.utils.data import DataLoader
from data_loader.LRHR_dataset import LRHRDataset
from imageio import imwrite as save
from models.ResDNet import ResDNet
from models.SRResDNet import SRResDNet
import models.discriminator_vgg_arch as SRGAN_arch
from modules.loss import GANLoss, TV_L2Loss, TV_L1LOSS
import modules.lr_scheduler as lr_scheduler
from modules import filters
import modules
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
torch.backends.cudnn.benchmark = True

def main():
    ################################
    ##  Network parameters
    ################################
        
    parser = argparse.ArgumentParser(description='Image Super-resolution with SRResCGAN')
    # SRResdnet Parameters
    parser.add_argument("--in_nc", type=int, default=3, help='no. of in_chs for D')
    parser.add_argument("--nf", type=int, default=64, help='no. of feat. maps for D')
    parser.add_argument("--resdnet_depth", type=int, default=5, help='no. of resblocks for resdnet')
    # Training Parameters
    parser.add_argument('--train_stdn', type = list, default=[0.0], help=" Number of noise levels (standard deviation) for which the network will be trained.") # [1.0,2.0,2.55,3.0,4.,5.0,5.10,6.0,7.,7.65,8.0,9.0,10.,11.0,12.0,12.75,13.0,14.0,15.0]
    parser.add_argument('--test_stdn', type = list, default=[0.0], help=" Number of noise levels (standard deviation) for testing.") # [2.55, 5.10, 7.65, 12.75] 
    parser.add_argument('--upscale_factor', type = int, default = 4, help='scaling factor.') # [2, 3, 4]
    parser.add_argument('--trainBatchSize', type = int, default = 16, help='training batch size.')
    parser.add_argument('--testBatchSize', type = int, default = 1, help='testing batch size.')
    parser.add_argument('--niter', type = int, default = 51000, help='number of iters to train for.')
    parser.add_argument('--use_bn', type=bool, default = False, help='use Batch-Norm?')
    parser.add_argument('--cuda', type=bool, default = True, help='use cuda?')
    parser.add_argument('--seed', type = int, default = 123, help='random seed to use. Default=123.')
    parser.add_argument('--use_filters', type=bool, default = True, help='use Filters: LP, HP?')
    parser.add_argument('--resume', type=bool, default = True, help='resume training?')
    parser.add_argument('--resume_start_epoch', type = int, default = 0, help='Where to resume training?.')
    parser.add_argument('--pretrainedModelPath', type = str, default = 'pretrained_nets/SRResDNet/G_perceptual.pth', help='location of pretrained model.')
    parser.add_argument('--pretrain', type = bool, default = True, help='Initialize the model paramaters from a pretrained net.')
    # DataSet Parameters
    parser.add_argument('--imdbTrainPath', type = str, default = 'datasets/', help='location of the training dataset.') 
    parser.add_argument('--imdbTestPath', type = str, default = 'datasets/', help='location of the testing dataset.')
    parser.add_argument('--patch_size', type = int, default = 32, help='patch size for training. [x2-->64,x3-->42,x4-->32]')
    parser.add_argument('--rgb_range', type = int, default = 255, help='data range of the training images.') 
    parser.add_argument('--is_train', type = bool, default = True, help=' True for training phase')
    parser.add_argument('--is_mixup', type = bool, default = True, help=' mixup_data augmentation for training data')
    parser.add_argument('--use_chop', type = bool, default = False, help=' chop for less memory consumption during test.')
    parser.add_argument('--alpha', type=float, default=1.2, help='alpha for data mixup (uniform=1., ERM=0.)')
    parser.add_argument('--numWorkers', type = int, default = 4, help='number of threads for data loader to use.')
    # Optimizer Parameters
    parser.add_argument('--lr_G', type = float, default = 1e-4, help='learning rate for G.')
    parser.add_argument('--beta1_G', type = float, default = 0.9, help='learning rate. Default=0.9.')
    parser.add_argument('--beta2_G', type = float, default = 0.999, help='learning rate. Default=0.999.')
    parser.add_argument('--eps_G', type = float, default = 1e-8, help='learning rate. Default=1e-8.')
    parser.add_argument('--weightdecay_G', type = float, default = 0, help='learning rate. Default=0.')
    parser.add_argument('--lr_D', type = float, default = 1e-4, help='learning rate for D.')
    parser.add_argument('--beta1_D', type = float, default = 0.9, help='learning rate. Default=0.9.')
    parser.add_argument('--beta2_D', type = float, default = 0.999, help='learning rate. Default=0.999.')
    parser.add_argument('--eps_D', type = float, default = 1e-8, help='learning rate. Default=1e-8.')
    parser.add_argument('--weightdecay_D', type = float, default = 0, help='learning rate. Default=0.')
    parser.add_argument('--amsgrad', type=bool, default = False, help='Use the fix for Adam?')
    parser.add_argument('--lr_milestones', type = list, default = [5000, 10000, 20000, 30000], help="Scheduler's learning rate milestones.")
    parser.add_argument('--lr_gamma', type = float, default = 0.5, help="multiplicative factor of learning rate decay.")
    parser.add_argument('--lr_restart', default = None, help='lr restart.')
    parser.add_argument('--lr_restart_weights', default = None, help='lr restart weights.')
    parser.add_argument('--warmup_iter', type = int, default = -1, help='warmup iter.')
    parser.add_argument('--D_update_ratio', type = int, default = 1, help='D_update_ratio.')
    parser.add_argument('--D_init_iters', type = int, default = 0, help='D_init_iters.')
    # losses Parameters
    parser.add_argument('--pixel_criterion', type = str, default = 'l1', help='pixel-wise criteria.')
    parser.add_argument('--feature_criterion', type = str, default = 'l1', help='feature criteria.')
    parser.add_argument('--tv_criterion', type = str, default = 'l1', help='TV criteria.')
    parser.add_argument('--gan_type', type = str, default = 'ragan', help='gan type. default: gan | ragan')
    parser.add_argument('--pixel_weight', type = float, default = 10., help='weight for pixel-wise criteria. default: 1e-2')
    parser.add_argument('--feature_weight', type = float, default = 1., help='weight for feature criteria.')
    parser.add_argument('--tv_weight', type = float, default = 1., help='weight for TV criteria.')
    parser.add_argument('--gan_weight', type = float, default = 1., help='weight for gan | ragan criteria. default: 5e-3')
    # Results Output Parameters
    parser.add_argument('--saveTrainedModelsPath', type = str, default = 'trained_nets', help='location of trained models.')
    parser.add_argument('--save_path_training_states', type = str, default = '/training_states/', help='location of training states.')
    parser.add_argument('--save_path_netG', type = str, default = '/netG/', help='location of trained netG.')
    parser.add_argument('--save_path_netD', type = str, default = '/netD/', help='location of trained netD.')
    parser.add_argument('--save_path_best_psnr', type = str, default = '/best_psnr/', help='location of trained model best PSNR.')
    parser.add_argument('--save_path_best_lpips', type = str, default = '/best_lpips/', help='location of trained model best LPIPS.')
    parser.add_argument('--saveImgsPath', type = str, default = 'results', help='location of saved images on training\validation.')
    parser.add_argument('--saveLogsPath', type = str, default = 'logs', help='location of training logs.')
    parser.add_argument('--save_checkpoint_freq', type = float, default = 20, help='Every how many iters we save the model parameters, default:5e3.')
    parser.add_argument('--saveBest', type=bool, default=True, help='save the best model parameters?')
    
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # shave boader to calculate PSNR and SSIM
    #border = opt.upscale_factor
    border = 0    
    
    # store trained models path
    models_save_path = opt.saveTrainedModelsPath+'_x'+str(opt.upscale_factor)+'/'
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)
    
    # store training states path
    training_states_save_path = models_save_path+opt.save_path_training_states
    if not os.path.exists(training_states_save_path):
        os.makedirs(training_states_save_path)
    
    # store trained netG path
    netG_save_path = models_save_path+opt.save_path_netG
    if not os.path.exists(netG_save_path):
        os.makedirs(netG_save_path)
    
    # store trained netD path
    netD_save_path = models_save_path+opt.save_path_netD
    if not os.path.exists(netD_save_path):
        os.makedirs(netD_save_path)
    
    # store trained model best PSNR path
    best_psnr_save_path = models_save_path+opt.save_path_best_psnr
    if not os.path.exists(best_psnr_save_path):
        os.makedirs(best_psnr_save_path)
    
    # store trained model best LPIPS path
    best_lpips_save_path = models_save_path+opt.save_path_best_lpips
    if not os.path.exists(best_lpips_save_path):
        os.makedirs(best_lpips_save_path)

    # save train images path
    save_train_imgs_path = opt.saveImgsPath+'_x'+str(opt.upscale_factor)+'/train_imgs/'
    if not os.path.exists(save_train_imgs_path):
        os.makedirs(save_train_imgs_path)
    
    # save test images path
    save_test_imgs_path = opt.saveImgsPath+'_x'+str(opt.upscale_factor)+'/test_imgs/' 
    if not os.path.exists(save_test_imgs_path):
        os.makedirs(save_test_imgs_path)
    
    # logs path
    logs_save_path =  opt.saveLogsPath+'_x'+str(opt.upscale_factor)+'/'
    if not os.path.exists(logs_save_path):
        os.makedirs(logs_save_path)
    
    # setup logging 
    utils_logger.logger_info('train_SRResCGAN', log_path=os.path.join(logs_save_path,'train_SRResCGAN.log'))
    logger = logging.getLogger('train_SRResCGAN')
    
    # save the training arguments
    torch.save(opt,os.path.join(logs_save_path,"args.pth"))
    
    logger.info('===================== Selected training parameters =====================')
    logger.info('{:s}.'.format(str(opt)))
    
    ################################
    ## datasets preparation 
    ################################
    #print('===> Loading dataset')
    logger.info('===================== Loading dataset =====================')        
    # training dataset
    train_dataset = LRHRDataset(dataroot=opt.imdbTrainPath, 
                                is_train=opt.is_train, 
                                scale=opt.upscale_factor, 
                                patch_size=opt.patch_size, 
                                rgb_range=opt.rgb_range, 
                                noise_std=opt.train_stdn)
        
    trainset_loader = DataLoader(train_dataset,
                                 shuffle=True,
                                 batch_size=opt.trainBatchSize,
                                 pin_memory=True,
                                 num_workers=opt.numWorkers,
                                 drop_last=True
                                 )
    train_size = int(math.ceil(len(train_dataset) / opt.trainBatchSize))    
    logger.info('training dataset:{:6d}'.format(len(train_dataset)))
    logger.info('training loaders:{:6d}'.format(len(trainset_loader)))
    
    # testing dataset
    test_dataset = LRHRDataset(dataroot=opt.imdbTestPath, 
                                is_train=False, 
                                scale=opt.upscale_factor, 
                                patch_size=opt.patch_size, 
                                rgb_range=opt.rgb_range, 
                                noise_std=opt.test_stdn)
        
    testset_loader = DataLoader(test_dataset,
                                shuffle=False,
                                batch_size=opt.testBatchSize,
                                pin_memory=True,
                                num_workers=1
                                )
    
    logger.info('testing dataset:{:6d}'.format(len(test_dataset)))
    logger.info('testing loaders:{:6d}'.format(len(testset_loader)))
    
    ################################
    ## Functions
    ################################
    def weights_init_normal(m, std=0.02):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            print('initializing [%s] ...' % classname)
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.normal_(m.weight.data, 1.0, std)
            init.constant_(m.bias.data, 0.0)

    def weights_init_kaiming(m, scale=1):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            print('initializing [%s] ...' % classname)
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.constant_(m.weight.data, 1.0)
            m.weight.data *= scale
            init.constant_(m.bias.data, 0.0)
    
    def weights_init_orthogonal(m):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            print('initializing [%s] ...' % classname)
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    
    def zeromean(m):
        # Function zeromean subtracts the mean E(f) from filters f
        # in order to create zero mean filters
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            print('zeromeans initializing [%s] ...' % classname)
            m.weight.data = m.weight.data - torch.mean(m.weight.data)
            #print('init zeromean weights:', m.weight.data.shape,  m.weight.data.min(),  m.weight.data.max())
    
    def init_weights(net, init_type='kaiming', zeromeans=False, scale=1, std=0.02):
        # scale for 'kaiming', std for 'normal'.
        print('initialization method [%s]' % init_type)
        if init_type == 'normal':
            weights_init_normal_ = functools.partial(weights_init_normal, std=std)
            net.apply(weights_init_normal_)
        elif init_type == 'kaiming':
            weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
            net.apply(weights_init_kaiming_)
        elif init_type == 'orthogonal':
            net.apply(weights_init_orthogonal)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        
        if zeromeans:
            weights_zeromean = functools.partial(zeromean)
            net.apply(weights_zeromean)
    
    def crop_forward(model, x, stdn, sf, shave=10, min_size=100000, bic=None):
        """
        chop for less memory consumption during test
        """
        n_GPUs = 1
        scale = sf
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
        
        if bic is not None:
            bic_h_size = h_size*scale
            bic_w_size = w_size*scale
            bic_h = h*scale
            bic_w = w*scale
            
            bic_list = [
                bic[:, :, 0:bic_h_size, 0:bic_w_size],
                bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
                bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
                bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]
            
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if bic is not None:
                    bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)
                
                sr_batch_temp = model(lr_batch, stdn)
                
                if isinstance(sr_batch_temp, list):
                    sr_batch = sr_batch_temp[-1]
                else:
                    sr_batch = sr_batch_temp
                    
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                crop_forward(model, x=patch, stdn=stdn, shave=shave, min_size=min_size) \
                for patch in lr_list
                ]
            
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        
        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
            
        return output
    
    def _net_init(model, init_type):
        print('==> Initializing the network using [%s]'%init_type)
        init_weights(model, init_type)
    
    def train(x, y, y_ref, sigma, 
              netG, netD, netF,
              optimizers,
              cri_pix, l_pix_w, 
              cri_fea, l_fea_w,
              cri_tv, l_tv_w,
              cri_gan, l_gan_w,
              filter_low, filter_high,
              step):
        
        netG.train()
        netD.train()
        
        # feed data 
        var_L = y  # LR
        var_H = x  # GT
        input_ref = y_ref if y_ref else x
        var_ref = input_ref
        #sigma_noise = sigma
        optimizer_G, optimizer_D = optimizers[0], optimizers[1]
        
        # G
        for p in netD.parameters():
            p.requires_grad = False

        optimizer_G.zero_grad()
        fake_H = netG(var_L).clamp(0.,255.)
        #print('train:: output fake_H:', fake_H.shape, fake_H.min(), fake_H.max())  

        l_g_total = 0
        if step % opt.D_update_ratio == 0 and step > opt.D_init_iters:
            if cri_pix:  # pixel loss
                #l_g_pix_f = l_pix_w * cri_pix(filter_low(fake_H), filter_low(var_H))
                l_g_pix_f = l_pix_w * cri_pix(fake_H, var_H)
                l_g_pix_nf = l_pix_w * cri_pix(fake_H, var_H)
                
                if opt.use_filters:
                    l_g_pix = l_g_pix_f
                else:
                    l_g_pix = l_g_pix_nf
                    
                l_g_total += l_g_pix
            
            if cri_fea:  # feature loss
                real_fea = netF(var_H).detach()
                fake_fea = netF(fake_H)
                l_g_fea = l_fea_w * cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            
            if cri_tv:  # TV loss
                l_g_tv = l_tv_w * cri_tv(fake_H, var_H)
                l_g_total += l_g_tv

            if opt.use_filters:
                pred_g_fake = netD(filter_high(fake_H))
            else:
                pred_g_fake = netD(fake_H)
                
            if opt.gan_type == 'gan':
                l_g_gan = l_gan_w * cri_gan(pred_g_fake, True)
            elif opt.gan_type == 'ragan':
                if opt.use_filters:
                    pred_d_real = netD(filter_high(var_ref)).detach()
                else:
                    pred_d_real = netD(var_ref).detach()
                l_g_gan = l_gan_w * (
                    cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan

            l_g_total.backward()
            optimizer_G.step()

        # D
        for p in netD.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()
        l_d_total = 0
        
        if opt.use_filters:
            pred_d_real = netD(filter_high(var_ref))
            pred_d_fake = netD(filter_high(fake_H.detach()))  # detach to avoid BP to G
        else:
            pred_d_real = netD(var_ref)
            pred_d_fake = netD(fake_H.detach())  # detach to avoid BP to G
        
        if opt.gan_type == 'gan':
            l_d_real = cri_gan(pred_d_real, True)
            l_d_fake = cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif opt.gan_type == 'ragan':
            l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2
        
        l_d_total.backward()
        optimizer_D.step()

        # set log
        if step % opt.D_update_ratio == 0 and step > opt.D_init_iters:
            if cri_pix:
                l_g_pix_batch = l_g_pix.item()
            if cri_fea:
                l_g_fea_batch = l_g_fea.item()
            if cri_tv:
                l_g_tv_batch = l_g_tv.item()
            l_g_gan_batch = l_g_gan.item()

        l_d_real_batch = l_d_real.item()
        l_d_fake_batch = l_d_fake.item()
        D_real_batch = torch.mean(pred_d_real.detach())
        D_fake_batch = torch.mean(pred_d_fake.detach())
        
        logger.info("===> train:: l_g_pix Loss:{:.6f}".format(l_g_pix_batch))
        logger.info("===> train:: l_g_fea Loss:{:.6f}".format(l_g_fea_batch))
        logger.info("===> train:: l_g_tv Loss:{:.6f}".format(l_g_tv_batch))
        logger.info("===> train:: l_g_gan Loss:{:.6f}".format(l_g_gan_batch))
        
        logger.info("===> train:: l_d_real Loss:{:.6f}".format(l_d_real_batch))
        logger.info("===> train:: l_d_fake Loss:{:.6f}".format(l_d_fake_batch))
        logger.info("===> train:: l_d_total Loss:{:.6f}".format(l_d_total.item()))
        
        logger.info("===> train:: D_real output:{:.6f}".format(D_real_batch))
        logger.info("===> train:: D_fake output:{:.6f}".format(D_fake_batch))
        
        output = fake_H
        total_loss_g = l_g_total.item()
        psnr_batch = psnr(fake_H, var_H, border=border).item()
        logger.info("===> train:: batch: Total Gen. Loss:{:.6f}".format(total_loss_g))
        logger.info("===> train:: batch: Gen. output PSNR:{:.4f}".format(psnr_batch))
        
        return output, total_loss_g, psnr_batch, netG, netD
    
    def test(model, lpips, epoch):
        model.eval()
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['lpips_dist'] = []
        img_idx = 1
        with torch.no_grad():
            for i, data in enumerate(testset_loader):
                y, x, sigma  = data['LR'], data['HR'], data['sigma']
                
                if opt.cuda:
                    y = y.cuda()
                    x = x.cuda()
                    sigma = sigma.cuda()
                
                y = y.float()
                x = x.float()
                sigma = sigma.float()
                
                #print("test x:", x.shape, x.min(), x.max())
                #print("test y:", y.shape, y.min(), y.max())
                #print("test sigma:", sigma.shape, sigma.min(), sigma.max())
                
                if opt.use_chop:
                    xhat = crop_forward(model, y, opt.upscale_factor)
                else:
                    outputs = model(y)
                    
                xhat = outputs.clamp(0.,255.)
                #print("test xhat:", xhat.shape, xhat.min(), xhat.max())   
                
                # save test images
                gt = x.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
                LR = y.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
                output = xhat.permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)
                
                gt_img_path = save_test_imgs_path+'img'+repr(img_idx)+'_GT'+'.png'
                LR_img_path = save_test_imgs_path+'img'+repr(img_idx)+'_LR'+'.png'
                output_img_path = save_test_imgs_path+'img'+repr(img_idx)+'_SR'+'.png'
                    
                # psnr, ssim, and lpips
                psnr = utils_common.calculate_psnr(output[0], gt[0], border=border)
                ssim = utils_common.calculate_ssim(output[0], gt[0], border=border)
                
                # normalized tensors
                img_x = util.normalized_tensor(x)
                pred_xhat = util.normalized_tensor(xhat)
                lpips_dist = lpips.forward(img_x,pred_xhat).item()
                
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                test_results['lpips_dist'].append(lpips_dist)
                        
                logger.info('{:->4d}--> {:>10s}, psnr:{:.2f}dB'.format(img_idx, output_img_path, psnr))
                logger.info('{:->4d}--> {:>10s}, ssim:{:.4f}'.format(img_idx, output_img_path, ssim))
                logger.info('{:->4d}--> {:>10s}, lpips dist:{:.4f}'.format(img_idx, output_img_path, lpips_dist))
                img_idx +=1
                    
                if epoch%1==0:    
                    save(gt_img_path, gt[0])
                    save(LR_img_path, LR[0])
                    save(output_img_path, output[0])
                     
                del x
                del y
                del sigma
                del xhat
                del gt
                del LR
                del output
                torch.cuda.empty_cache()
    
        avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        avg_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        avg_lpips = sum(test_results['lpips_dist']) / len(test_results['lpips_dist'])
        
        #print("===>test:: Avg. PSNR:{:.2f}".format(avg_psnr))
        #print("===>test:: Avg. SSIM:{:.4f}".format(avg_ssim))
        logger.info("test:: Epoch[{}]: Avg. PSNR: {:.2f} dB".format(epoch, avg_psnr))
        logger.info("test:: Epoch[{}]: Avg. SSIM: {:.4f}".format(epoch, avg_ssim))
        logger.info("test:: Epoch[{}]: Avg. Lpips: {:.6f}".format(epoch, avg_lpips))
        
        return avg_psnr, avg_lpips
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_current_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr'] 
    
    def _set_lr(optimizers, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(optimizers):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(optimizers, schedulers, cur_iter, warmup_iter=-1):
        for scheduler in schedulers:
            scheduler.step()
        # set up warm-up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = _get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            _set_lr(optimizers, warm_up_lr_l)
    
    def save_network(path_model, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(path_model, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
#        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
#        for k, v in load_net.items():
#            if k.startswith('module.'):
#                load_net_clean[k[7:]] = v
#            else:
#                load_net_clean[k] = v
        network.load_state_dict(load_net, strict=strict)
        return network
    
    def custom_load(netG, netD, load_path_G, load_path_D):
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            netG = load_network(load_path_G, netG)

        if load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            netD = load_network(load_path_D, netD)
        return netG, netD
    
    def save_checkpoint_best(model_path, epoch, iter_step, optimizers, schedulers, metrics, label):   
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        state = {**state, **metrics}
        for s in schedulers:
            state['schedulers'].append(s.state_dict())
        for o in optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}_tr_states_best_'.format(epoch)+label+'.pth'
        save_path = os.path.join(model_path, save_filename)
        torch.save(state, save_path)
        logger.info("===> Checkpoint saved to {:s}".format(save_path))
    
    def save_training_state(epoch, iter_step, optimizers, schedulers, metrics):
        """Save training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        state = {**state, **metrics}
        for s in schedulers:
            state['schedulers'].append(s.state_dict())
        for o in optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}_tr_states.pth'.format(epoch)
        save_path = os.path.join(training_states_save_path, save_filename)
        torch.save(state, save_path)
    
    def resume_training(resume_path, optimizers, schedulers):
        """Resume the optimizers and schedulers for training"""
        if opt.cuda:
            resume_state = torch.load(resume_path,map_location=lambda storage,loc:storage.cuda())
        else:
            resume_state = torch.load(resume_path,map_location=lambda storage,loc:storage)
        
        start_epoch = resume_state['epoch']
        iter_step = resume_state['iter']
        epoch_psnr_old = resume_state['epoch_psnr_old']
        epoch_lpips_old = resume_state['epoch_lpips_old']
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            schedulers[i].load_state_dict(s)
        
        return start_epoch, iter_step, epoch_psnr_old, epoch_lpips_old, optimizers, schedulers
    
    def get_network_description(network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))
    
    def print_network(netG, netD, netF):
        # Generator
        if netG is not None:
            s, n = get_network_description(netG)
            if isinstance(netG, nn.DataParallel) or isinstance(netG, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(netG.__class__.__name__,
                                                 netG.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(netG.__class__.__name__)
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        
        # Discriminator
        if netD is not None:
            s, n = get_network_description(netD)
            if isinstance(netD, nn.DataParallel) or isinstance(netD, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(netD.__class__.__name__,
                                                 netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(netD.__class__.__name__)
            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

        # F, Perceptual Network
        if netF is not None:
            s, n = get_network_description(netF)
            if isinstance(netF, nn.DataParallel) or isinstance(netF, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(netF.__class__.__name__,
                                                 netF.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(netF.__class__.__name__)
            logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
    
    ### Utils for managing network parameters ###
    def get_module_name_dict(root, rootname="/"):
        def _rec(module, d, name):
            for key, child in module.__dict__["_modules"].items():
                d[child] = name + key + "/"
                _rec(child, d, d[child])
    
        d = {root: rootname}
        _rec(root, d, d[root])
        return d
    
    def parameters_by_module(net, name=""):
        modulenames = get_module_name_dict(net, name + "/")
        params = [{"params": p, "name": n, "module": modulenames[m]} for m in net.modules() for n,p in m._parameters.items() if p is not None]
        return params
    
    def parameter_count(net):
        parameters = parameters_by_module(net)
    
        nparams = 0
        for pg in parameters:
            for p in pg["params"]:
                nparams+=p.data.numel()
    
        return nparams
    
    ################################
    ## NN Architecture
    ################################
    logger.info('===================== Building model =====================')
    
    # denoiser net
    resdnet = ResDNet(depth=opt.resdnet_depth)
    # Generator net
    netG = SRResDNet(resdnet, scale=opt.upscale_factor)
    
    # Discriminator net
    netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt.in_nc, nf=opt.nf)
    
    # initialize model with pertrained network
    if opt.pretrain:
        logger.info('Initialized model with pretrained net from {:s}.'.format(opt.pretrainedModelPath))
        netG = load_network(opt.pretrainedModelPath, netG)
    else:
        netG = _net_init(netG, init_type='kaiming')
        netD = _net_init(netD, init_type='kaiming')
    
    # Filters: low-pass: W_L, high-pass: W_H
    filter_low = filters.FilterLow()
    filter_high = filters.FilterHigh()
            
    if opt.cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        filter_low = filter_low.cuda()
        filter_high = filter_high.cuda()
    
    # optimizers
    # Adam    
    optimizer_G = torch.optim.Adam(netG.parameters(), 
                                 lr=opt.lr_G,  
                                 betas=(opt.beta1_G, opt.beta2_G), 
                                 eps=opt.eps_G, 
                                 weight_decay=opt.weightdecay_G,
                                 amsgrad=opt.amsgrad)
    
    optimizer_D = torch.optim.Adam(netD.parameters(), 
                                 lr=opt.lr_D,  
                                 betas=(opt.beta1_D, opt.beta2_D), 
                                 eps=opt.eps_D, 
                                 weight_decay=opt.weightdecay_D,
                                 amsgrad=opt.amsgrad)
    optimizers = [optimizer_G, optimizer_D]
    
    # schedulers
    scheduler_G = lr_scheduler.MultiStepLR_Restart(optimizer_G, 
                                                   milestones=opt.lr_milestones,
                                                   restarts=opt.lr_restart,
                                                   weights=opt.lr_restart_weights,
                                                   gamma=opt.lr_gamma)
    scheduler_D = lr_scheduler.MultiStepLR_Restart(optimizer_D, 
                                                   milestones=opt.lr_milestones,
                                                   restarts=opt.lr_restart,
                                                   weights=opt.lr_restart_weights,
                                                   gamma=opt.lr_gamma)
    schedulers =  [scheduler_G, scheduler_D]
    
    # losses criteria  
    # G pixel loss
    l_pix_type = opt.pixel_criterion
    if l_pix_type == 'l1':
        cri_pix = nn.L1Loss()
    elif l_pix_type == 'l2':
        cri_pix = nn.MSELoss()
    else:
        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
    l_pix_w = opt.pixel_weight
    
    # G TV loss
    l_tv_type = opt.tv_criterion
    if l_tv_type == 'l1':
        cri_tv = TV_L1LOSS()
    elif l_tv_type == 'l2':
        cri_tv = TV_L2Loss()
    else:
        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_tv_type))
    l_tv_w = opt.tv_weight

    # G feature loss
    l_fea_type = opt.feature_criterion
    if l_fea_type == 'l1':
        cri_fea = nn.L1Loss()
    elif l_fea_type == 'l2':
        cri_fea = nn.MSELoss()
    else:
        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
    l_fea_w = opt.feature_weight
    
    if cri_fea:  # load VGG perceptual loss
        # PyTorch pretrained VGG19-54, before ReLU.
        if opt.use_bn:
            feature_layer = 49
        else:
            feature_layer = 34
        netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, 
                                              use_bn=opt.use_bn,
                                              use_input_norm=True, 
                                              device='cuda')
        netF.eval()  # No need to train
        netF = netF.cuda()
        
    # GD gan loss
    cri_gan = GANLoss(opt.gan_type, 1.0, 0.0)
    l_gan_w = opt.gan_weight
    
    if opt.cuda:
        cri_pix = cri_pix.cuda()
        cri_fea = cri_fea.cuda()
        cri_gan = cri_gan.cuda()
    
    # lpips distance metric for test
    lpips = modules.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.cuda) # alex, squeeze, vgg
    
    # print networks
    print_network(netG, netD, netF)
    
    ################################
    ## Main
    ################################
    # start training
    logger.info('===================== start training =====================')
    #resume training
    if opt.resume:
        logger.info('===================== resume training =====================')
        resume_path = training_states_save_path
        if not os.listdir(resume_path):
            logger.info('===> No saved training states to resume.')
            current_step = 0
            start_epoch = 0
            epoch_psnr_old = -float('inf')
            epoch_lpips_old = float('inf')
            logger.info('===> start training from epoch: {}, iter: {}.'.format(start_epoch, current_step))
        else:
            resume_start_epoch = opt.resume_start_epoch
            resume_path = training_states_save_path+str(resume_start_epoch)+'_tr_states.pth'
            start_epoch, current_step, epoch_psnr_old, epoch_lpips_old, optimizers, schedulers = resume_training(resume_path, 
                                                                                                                 optimizers, 
                                                                                                                 schedulers)
            logger.info('===> loading pretrained models: G, D.')
            load_path_G =  netG_save_path + str(start_epoch) + '_G.pth'
            load_path_D = netD_save_path + str(start_epoch) + '_D.pth'
            netG, netD = custom_load(netG, netD, load_path_G, load_path_D)
            logger.info('===> Resuming training from epoch: {}, iter: {}.'.format(start_epoch, current_step))
    
    #training loop
    t = timer()
    t.tic()
    total_iters = int(opt.niter)
    total_epochs = int(math.ceil(total_iters / train_size))
    logger.info('Total # of epochs for training: {}.'.format(total_epochs))
    
    for epoch in range(start_epoch+1, total_epochs+1):
        #print('===> Epoch %d' % epoch)
        logger.info("===> train:: Epoch[{}]".format(epoch))
        epoch_loss = 0
        epoch_psnr = 0
    
        for _, data in enumerate(trainset_loader):
            current_step += 1
            if current_step > total_iters:
                break
            
            #### update learning rate
            update_learning_rate(optimizers, 
                                 schedulers, 
                                 cur_iter=current_step, 
                                 warmup_iter=opt.warmup_iter)
            
            logger.info("===> train:: Epoch[{}] \t Iter-step[{}]".format(epoch, current_step)) 
            y, x, sigma  = data['LR'], data['HR'], data['sigma']
            y_ref = None
    
            if opt.cuda:
                    y = y.cuda()
                    x = x.cuda()
                    #y_ref = y_ref.cuda()
                    sigma = sigma.cuda()
            
            # generate mixed inputs, targets and mixing coefficient
            if (opt.is_mixup and random.random() < 0.5):
                x, y, sigma, lam = utils_common.mixup_data(x, y, sigma, alpha=opt.alpha, use_cuda=opt.cuda)
            
            x = x.float()
            y = y.float()
            sigma = sigma.float()    
            #print('train x:', x.shape, x.min(), x.max())
            #print('train y:', y.shape, y.min(), y.max())
            #print('train sigma:', sigma.shape, sigma.min(), sigma.max())
    
            xhat, loss, psnr_batch, netG, netD = train(x, y, y_ref, sigma, 
                                                  netG, netD, netF,
                                                  optimizers,
                                                  cri_pix, l_pix_w, 
                                                  cri_fea, l_fea_w,
                                                  cri_tv, l_tv_w,
                                                  cri_gan, l_gan_w,
                                                  filter_low, filter_high,
                                                  current_step)
            epoch_loss += loss
            epoch_psnr += psnr_batch
            
            # save train images
            gt = x.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            LR = y.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            output = xhat.permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)
            idx = 1
            for j in range(gt.shape[0]):
                if current_step%100==0:
                    save(save_train_imgs_path+'img'+repr(idx)+'_GT'+'.png', gt[j])
                    save(save_train_imgs_path+'img'+repr(idx)+'_LR'+'.png', LR[j])
                    save(save_train_imgs_path+'img'+repr(idx)+'_SR'+'.png', output[j])
                
                # psnr and ssim
                psnr_SR = utils_common.calculate_psnr(output[j], gt[j], border=border)
                ssim_SR = utils_common.calculate_ssim(output[j], gt[j], border=border)
                
                logger.info('===> train:: {:->4d}--> {:>10s}, SR psnr:{:.2f}dB'.format(idx, save_train_imgs_path+'img'+repr(idx)+'_sr'+'.png', psnr_SR))
                logger.info('===> train:: {:->4d}--> {:>10s}, SR ssim:{:.4f}'.format(idx, save_train_imgs_path+'img'+repr(idx)+'_sr'+'.png', ssim_SR))
                idx += 1
            
            del x
            del y
            del y_ref
            del sigma
            del xhat
            del gt
            del LR
            del output
            torch.cuda.empty_cache()
            #break
        
        logger.info("train:: Epoch[{}] Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss/len(trainset_loader)))
        logger.info("train:: Epoch[{}] Complete: Avg. PSNR: {:.2f} dB".format(epoch, epoch_psnr/len(trainset_loader)))
        
        # testing
        logger.info('===================== start testing =====================')
        epoch_psnr_new, epoch_lpips_new = test(netG, lpips, epoch)
        logger.info('===================== end testing =====================')
        
        # current learning rate (lr)
        logger.info("train:: current lr[{:.8f}]".format(get_current_learning_rate(optimizer_G)))
        
        # saving the best model w.r.t. psnr
        if opt.saveBest:
            if epoch_psnr_new > epoch_psnr_old:
                epoch_psnr_old = epoch_psnr_new    
                save_checkpoint_best(best_psnr_save_path, 
                                     epoch, current_step, 
                                     optimizers, schedulers, 
                                     {'epoch_psnr_old':epoch_psnr_old,
                                      'epoch_lpips_old':epoch_lpips_old}, 
                                      "psnr")
                save_network(best_psnr_save_path, netG, 'G_best_psnr', epoch)
                save_network(best_psnr_save_path, netD, 'D_best_psnr', epoch)
        
        # saving the best model w.r.t. lpips
        if opt.saveBest:
            if epoch_lpips_new < epoch_lpips_old:
                epoch_lpips_old = epoch_lpips_new
                save_checkpoint_best(best_lpips_save_path, 
                                     epoch, current_step, 
                                     optimizers, schedulers, 
                                     {'epoch_psnr_old':epoch_psnr_old,
                                      'epoch_lpips_old':epoch_lpips_old}, 
                                      "lpips")
                save_network(best_lpips_save_path, netG, 'G_best_lpips', epoch)
                save_network(best_lpips_save_path, netD, 'D_best_lpips', epoch)
        
        # save models and training states
        if epoch % opt.save_checkpoint_freq == 0:
            logger.info('Saving models and training states.')
            save_network(netG_save_path, netG, 'G', epoch)
            save_network(netD_save_path, netD, 'D', epoch)
            save_training_state(epoch, current_step, optimizers, schedulers, 
                                {'epoch_psnr_old':epoch_psnr_old,
                                 'epoch_lpips_old':epoch_lpips_old})
        #break
    
    logger.info('===================== Saving the final model =====================')
    # save final network states
    save_network(netG_save_path, netG, 'final_G', epoch)
    save_network(netD_save_path, netD, 'final_D', epoch)
    
    logger.info('===================== Training completed in {:.4f} seconds ====================='.format(t.toc()))
    logger.info('===================== end training =====================')
    ###############################
    # End
    ###############################

if __name__== '__main__':
    main()
