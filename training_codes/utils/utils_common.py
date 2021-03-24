import numpy as np
#from scipy.misc import imresize
import scipy
import scipy.io as spio 
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils_model as utils
from utils import utils_image
import cv2
import math
import random

def generate_noisy(x, nstd, random_seed=20180102):
    """
    This function generates noisy images from clear ones.
    Input:
        x: (N, C, H, W) 
        nstd: noise sigma added to clean image
    """
    shape = x.shape
    dtype = x.dtype
    sigma = nstd
    
    torch.manual_seed(random_seed)
    
    noise = sigma * torch.randn(*shape, dtype=dtype)
    
    x_noise = x + noise

    return x_noise.clamp(0.,1.)

def generate_blur_noisy(x, kernel, nstd, same=True, random_seed=20180102):
    """
    This function generates blurry noisy images from clear ones and corresponding kernels.
    Input:
        x: (N, C, H, W) 
        kernel: (N, 1, Hk, Wk) kernels
        nstd: noise sigma added to blur image
        same: whether convolution keeps result same size as x. If False, result will be with size of (N, C, H-Hk+1, W-Wk+1). 
    """
    sigma = nstd
    dtype = x.dtype
    torch.manual_seed(random_seed)

    if same:
        blur = utils.imfilter2D_SpatialDomain(x, kernel, padType='zero', mode="conv")
        shape = blur.shape
        noise = sigma * torch.randn(*shape, dtype=dtype)
        blur_noisy = blur + noise
    else:
        blur = utils.imfilter2D_SpatialDomain(x, kernel, padType='valid', mode="conv")
        shape = blur.shape
        noise = sigma * torch.randn(*shape, dtype=dtype)
        blur_noisy = blur + noise

    return blur_noisy.clamp(0.,1.)

def generate_LR_noisy(x, nstd, sf, random_seed=20180102):
    """
    This function generates LR noisy images from clear ones and corresponding kernels using the following forward model: y = x_ds + n.
    Input:
        x: (N, C, H, W) 
        sf: scale factor (2,3,4) 
        nstd: noise sigma added to LR image
    """
    sigma = nstd
    dtype = x.dtype
    
    lr_img = utils_image.imresize(x, scale=1/sf).type(dtype)
    #print('ds_func:', lr_img.shape, lr_img.min(), lr_img.max())
    shape = lr_img.shape
    #noise = sigma * torch.randn(*shape, dtype=dtype)
    rng = np.random.RandomState(random_seed)
    random_noise = rng.randn(*shape)
    random_noise = torch.from_numpy(random_noise).type(dtype)
    noise = sigma * random_noise
    #print('noise_func:', noise.shape, noise.min(), noise.max())
    LR_noisy = lr_img + noise

    return LR_noisy.clamp(0.,1.)

def generate_LR_blur_noisy(x, kernel, nstd, sf, same=True, random_seed=20180102):
    """
    This function generates LR blurred noisy images from clear ones and corresponding kernels using the following forward model: y = k * x_ds + n.
    Input:
        x: (N, C, H, W) 
        kernel: (N, 1, Hk, Wk) kernels
        sf: scale factor (2,3,4) 
        nstd: noise sigma added to blur image
        same: whether convolution keeps result same size as x. If False, result will be with size of (N, C, H-Hk+1, W-Wk+1). 
    """
    sigma = nstd
    dtype = x.dtype
    torch.manual_seed(random_seed)

    if same:
        lr_img = utils_image.imresize(x, scale=1/sf).type(dtype)
        #print('ds_func:', lr_img.shape, lr_img.min(), lr_img.max())
        LR_blurred = utils.imfilter2D_SpatialDomain(lr_img, kernel, padType='zero', mode="conv")
        #print('blur_func:', LR_blurred.shape, LR_blurred.min(), LR_blurred.max())
        shape = LR_blurred.shape
        noise = sigma * torch.randn(*shape, dtype=dtype)
        #print('noise_func:', noise.shape, noise.min(), noise.max())
        LR_blurred_noisy = LR_blurred + noise
    else:
        lr_img = utils_image.imresize(x, scale=1/sf).type(dtype)
        #print('ds_func:', lr_img.shape, lr_img.min(), lr_img.max())
        LR_blurred = utils.imfilter2D_SpatialDomain(lr_img, kernel, padType='valid', mode="conv")
        #print('blur_func:', LR_blurred.shape, LR_blurred.min(), LR_blurred.max())
        shape = LR_blurred.shape
        noise = sigma * torch.randn(*shape, dtype=dtype)
        #print('noise_func:', noise.shape, noise.min(), noise.max())
        LR_blurred_noisy = LR_blurred + noise

    return LR_blurred_noisy.clamp(0.,1.)

def mixup_data(x, y, sigma, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, targets, and lambda'''
    
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    mixed_y = lam * y + (1 - lam) * y[index,:]
    mixed_sigma = lam * sigma + (1 - lam) * sigma[index]
    
    return mixed_x, mixed_y, mixed_sigma, lam

def psnr(xhat, x, border=0):
    N, C, H, W = x.size()
    x = x[:, :, border:H-border, border:W-border]
    xhat = xhat[:, :, border:H-border, border:W-border]
    loss = 0
    log10 = 1 / torch.log(torch.FloatTensor([10])).numpy()[0]
    for i in range(N):
        mse = F.mse_loss(xhat[i], x[i])
    
        if mse.data.cpu().numpy() == 0:
            loss += 100
            continue
        loss += 20 * torch.log(255.0 / torch.sqrt(mse)) * log10
    loss /= N
    return loss

#def psnr(xhat, x):
#    avg_psnr = 0
#    for i in range(xhat.size(0)):
#        mse = torch.mean(torch.pow(xhat.data[i] - x.data[i], 2))
#        try:
#            avg_psnr += 10 * np.log10(255**2 / mse)
#        except:
#            continue
#    return avg_psnr / x.size(0)

#def modcrop(img, modulo):
#    if len(img.shape) == 2:
#        sz = img.shape
#        sz = sz - np.mod(sz, modulo)
#        img = img[0:sz[0], 0:sz[1]]
#    else:
#        tmpsz = img.shape
#        sz = tmpsz[0:2]
#        sz = sz - np.mod(sz, modulo);
#        img = img[0:sz[0], 0:sz[1], :]
#    
#    return img

def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw = img_in.shape[:2]
    oh, ow = img_tar.shape[:2]

    ip = patch_size

    if ih == oh:
        tp = ip
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = ix, iy
    else:
        tp = ip * scale
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

def modcrop(img, scale):
    # img: Pytorch, C*H*W 
    C, H, W = img.shape
    H_r, W_r = H % scale, W % scale
    img = img[:, :H - H_r, :W - W_r]
    return img

def modcrop_np(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def mse_grad_loss(input, target, grad=False):
    
    assert(input.shape == target.shape), "The tensor inputs must be of the same size."
        
    assert(input.dim() <= 4), "Tensor must be at maximum of 4 dimensions."
        
    while input.dim() < 4:
        input = input.unsqueeze(0)
        
    while target.dim() < 4:
        target = target.unsqueeze(0)        
    
    err = input-target
    loss = err.norm(p=2).pow(2).div(err.numel())
    if grad:
        loss += utils.imGrad(err,bc='reflexive').norm(p=2).pow(2).div(err.numel())
        
    return loss

class MSEGradLoss(nn.Module):
    def __init__(self,grad=False):
        
        super(MSEGradLoss,self).__init__()
        
        self.grad = grad
    
    def forward(self, input, target):
        err = input - target
        loss = err.norm(p=2).pow(2).div(err.numel())
        if self.grad:
            loss += utils.imGrad(err,bc='reflexive').norm(p=2).pow(2).div(err.numel())
        
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + 'gradMSE = ' + str(self.grad)+ ')'   

class L1GradLoss(nn.Module):
    def __init__(self,grad=False):
        
        super(L1GradLoss,self).__init__()
        
        self.grad = grad
    
    def forward(self, input, target):
        err = input - target
        #loss = err.view(-1,1).abs().sum().div(err.numel())
        loss = err.norm(p=1).div(err.numel())
        if self.grad:
            loss += utils.imGrad(err,bc='reflexive').norm(p=1).div(err.numel())
        
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + 'gradL1 = ' + str(self.grad)+ ')'
        
class PSNRLoss(nn.Module):
    
    def __init__(self, peakval=255.0):
        
        super(PSNRLoss,self).__init__()       
        
        self.peakval = peakval
        self.log10 = 1 / torch.log(torch.FloatTensor([10])).numpy()[0]
        
    def forward(self, input, target):
        N, C, H, W = target.size()
        
        if self.peakVal is None:
            self.peakVal = target.view(N,-1).max(dim=1)[0]        
        
        loss = 0
        for i in range(N):
            mse = F.mse_loss(input[i], target[i])
        
            if mse.data.cpu().numpy() == 0:
                loss += 100
                continue
            loss += 20 * torch.log(self.peakval / torch.sqrt(mse)) * self.log10
        loss /= N
            
        return loss
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'peakVal = ' + str(self.peakval) + ')'   

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                           [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def calc_metrics(img1, img2, crop_border, test_Y):
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2

    psnr = calculate_psnr(im1_in * 255, im2_in * 255, border=crop_border)
    ssim = calculate_ssim(im1_in * 255, im2_in * 255, border=crop_border)
    
    return psnr, ssim
    
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    link: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def load_resdnet_params(model, pretrained_model_path, depth):
    # load the weights
    weights = loadmat(pretrained_model_path)
    state_dict = model.state_dict()
    # load l2proj

    #state_dict['l2proj.alpha'] = torch.FloatTensor([weights['net']['layers'][-4]['weights']])
    # load conv2d and conv2dT
    state_dict['conv1.bias'] = torch.FloatTensor(np.array(weights['net']['layers'][0]['weights'][1]))
    state_dict['conv1.weight_g'] = torch.FloatTensor(np.array(weights['net']['layers'][0]['weights'][2]))[:,None,None,None]
    state_dict['conv1.weight_v'] = torch.FloatTensor(np.array(weights['net']['layers'][0]['weights'][0])).permute(3,2,1,0)

    state_dict['conv_out.bias'] = torch.FloatTensor(np.array(weights['net']['layers'][-5]['weights'][1]))
    state_dict['conv_out.weight_g'] = torch.FloatTensor(np.array(weights['net']['layers'][-5]['weights'][2]))[:,None,None,None]
    state_dict['conv_out.weight_v'] = torch.FloatTensor(np.array(weights['net']['layers'][-5]['weights'][0])).permute(3,2,1,0)
    # fill layers
    for i in range(depth):
        layer = [k for k in state_dict.keys() if 'layer1.'+str(i) in k]
        state_dict[layer[0]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][1]))
        state_dict[layer[1]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][2]))[:,None,None,None]
        state_dict[layer[2]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][0])).permute(3,2,1,0)
        state_dict[layer[3]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][-2]))
        state_dict[layer[4]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][-1]))
        state_dict[layer[5]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][4]))
        state_dict[layer[6]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][5]))[:,None,None,None]
        state_dict[layer[7]] = torch.FloatTensor(np.array(weights['net']['layers'][i+1]['weights'][3])).permute(3,2,1,0)
        # load all weights to model
    model.load_state_dict(state_dict)
    return model

#def load_resdnet_params(pretrain_path, model, depth=5):
#    pretrain_net = torch.load(pretrain_path, map_location = lambda storage, loc:storage)
#    states_pretrain_net = pretrain_net['model_state_dict']
#    #print('states_pretrain_net len:', len(states_pretrain_net))
#    #states_resdnet = [k for k in states_pretrain_net.keys()]
#    #print('states resdnet:', states_resdnet)
#    states_net = model.state_dict()
#    #print('states_net len:', len(states_net))
#    #states_srresdnet = [k for k in states_net.keys()]
#    #print('states srresdnet:', states_srresdnet)
#    
#    
#    states_net['head_conv.bias'] = states_pretrain_net['bias_f']
#    states_net['head_conv.weight_g'] = states_pretrain_net['scale_f'][:,None,None,None]
#    states_net['head_conv.weight_v'] = states_pretrain_net['conv_weights']
#    states_net['tconv.bias'] = states_pretrain_net['bias_t']
#    states_net['tconv.weight_g'] = states_pretrain_net['scale_f'][:,None,None,None]
#    states_net['tconv.weight_v'] = states_pretrain_net['conv_weights']
#    states_net['l2proj.alpha'] = states_pretrain_net['alpha']
#
#    for i in range(depth):
#            states_resdnet = [k for k in states_pretrain_net.keys() if 'resPA.'+str(i) in k]
#            states_srresdnet = [k for k in states_net.keys() if 'resBlocks.'+str(i) in k]
#            #print('states_resdnet:', states_resdnet)
#            #print('states_srresdnet:', states_srresdnet)
#
#            states_net[states_srresdnet[2]] = states_pretrain_net[states_resdnet[0]]
#            states_net[states_srresdnet[6]] = states_pretrain_net[states_resdnet[1]] 
#            states_net[states_srresdnet[1]] = states_pretrain_net[states_resdnet[2]][:,None,None,None] 
#            states_net[states_srresdnet[5]] = states_pretrain_net[states_resdnet[3]][:,None,None,None] 
#            states_net[states_srresdnet[0]] = states_pretrain_net[states_resdnet[4]] 
#            states_net[states_srresdnet[4]] = states_pretrain_net[states_resdnet[5]] 
#            states_net[states_srresdnet[3]] = states_pretrain_net[states_resdnet[6]] 
#            states_net[states_srresdnet[7]] = states_pretrain_net[states_resdnet[7]]
#    
#    model.load_state_dict(states_net)
#    return model

def load_srresdnet_params(pretrain_path, model, res_depth=8, upsamp_depth=1):
    pretrain_net = torch.load(pretrain_path, map_location = lambda storage, loc:storage)
    states_pretrain_net = pretrain_net['model_state_dict']
#    print('states_pretrain_net len:', len(states_pretrain_net))
#    states_srresdnet = [k for k in states_pretrain_net.keys()]
#    print('states srresdnet:', states_srresdnet)
    states_net = model.state_dict()
#    print('states_net len:', len(states_net))
#    states_srwdnet = [k for k in states_net.keys()]
#    print('states srresdnet:', states_srwdnet)
    
    # head_conv
    states_net['head_conv.bias'] = states_pretrain_net['head_conv.bias']
    states_net['head_conv.weight_g'] = states_pretrain_net['head_conv.weight_g']
    states_net['head_conv.weight_v'] = states_pretrain_net['head_conv.weight_v']
    # tconv
    states_net['tconv.bias'] = states_pretrain_net['tconv.bias']
    states_net['tconv.weight_g'] = states_pretrain_net['tconv.weight_g']
    states_net['tconv.weight_v'] = states_pretrain_net['tconv.weight_v']
    # tail_conv
    states_net['tail_conv.bias'] = states_pretrain_net['tail_conv.bias']
    states_net['tail_conv.weight_g'] = states_pretrain_net['tail_conv.weight_g']
    states_net['tail_conv.weight_v'] = states_pretrain_net['tail_conv.weight_v']
    # proj alpha
    states_net['l2proj.alpha'] = states_pretrain_net['l2proj.alpha']

    for i in range(res_depth):
            states_srresdnet = [k for k in states_pretrain_net.keys() if 'resBlocks.'+str(i) in k]
            states_srwdnet = [k for k in states_net.keys() if 'resBlocks.'+str(i) in k]
            #print('states_srresdnet:', states_srresdnet)
            #print('states_srwdnet:', states_srwdnet)

            states_net[states_srwdnet[0]] = states_pretrain_net[states_srresdnet[0]]
            states_net[states_srwdnet[1]] = states_pretrain_net[states_srresdnet[1]] 
            states_net[states_srwdnet[2]] = states_pretrain_net[states_srresdnet[2]]
            states_net[states_srwdnet[3]] = states_pretrain_net[states_srresdnet[3]]
            states_net[states_srwdnet[4]] = states_pretrain_net[states_srresdnet[4]] 
            states_net[states_srwdnet[5]] = states_pretrain_net[states_srresdnet[5]] 
            states_net[states_srwdnet[6]] = states_pretrain_net[states_srresdnet[6]] 
            states_net[states_srwdnet[7]] = states_pretrain_net[states_srresdnet[7]]
    
    for i in range(upsamp_depth):
            states_srresdnet = [k for k in states_pretrain_net.keys() if 'upsampling.upsampling.'+str(i) in k]
            states_srwdnet = [k for k in states_net.keys() if 'upsampling.upsampling.'+str(i) in k]
            #print('states_srresdnet:', states_srresdnet)
            #print('states_srwdnet:', states_srwdnet)

            states_net[states_srwdnet[0]] = states_pretrain_net[states_srresdnet[0]]
            states_net[states_srwdnet[1]] = states_pretrain_net[states_srresdnet[1]] 
            states_net[states_srwdnet[2]] = states_pretrain_net[states_srresdnet[2]]
            
    model.load_state_dict(states_net)
    
    return model

if __name__=='__main__':
    x = torch.randn(5,3,50,50).type(torch.FloatTensor)
    xhat = torch.randn(5,3,50,50).type(torch.FloatTensor)
    
    psnr1 = psnr(xhat, x, border=4)
    print('psnr1:', psnr1.item())
    

