import os.path as osp
import glob
import cv2
import numpy as np
import torch
from models.SRResCGAN import Generator
from collections import OrderedDict
from utils import timer, augment_img_tensor, inv_augment_img_tensor

model_path = 'trained_nets_x4/srrescgan_model.pth'  # trained G_{SR} model of SRResCGAN
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*' # testset LR images path

model = Generator(scale=4) # SRResCGAN generator net
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

test_results = OrderedDict()
test_results['time'] = []
idx = 0
for path_lr in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path_lr))[0]
    print(idx, base)
    
    # read images: LR
    img_lr = cv2.imread(path_lr, cv2.IMREAD_COLOR)
    img_LR = torch.from_numpy(np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img_LR.unsqueeze(0)
    img_LR = img_LR.to(device)
    
    # self-ensemble strategy 
    # augmentations transformation
    img_LR0 = augment_img_tensor(img_LR, mode=0)
    img_LR1 = augment_img_tensor(img_LR, mode=1)
    img_LR2 = augment_img_tensor(img_LR, mode=2)
    img_LR3 = augment_img_tensor(img_LR, mode=3)
    img_LR4 = augment_img_tensor(img_LR, mode=4)
    img_LR5 = augment_img_tensor(img_LR, mode=5)
    img_LR6 = augment_img_tensor(img_LR, mode=6)
    img_LR7 = augment_img_tensor(img_LR, mode=7)
    
    # testing
    t = timer()
    t.tic()
    with torch.no_grad():
        output_SR0 = model(img_LR0)
        output_SR1 = model(img_LR1)
        output_SR2 = model(img_LR2)
        output_SR3 = model(img_LR3)
        output_SR4 = model(img_LR4)
        output_SR5 = model(img_LR5)
        output_SR6 = model(img_LR6)
        output_SR7 = model(img_LR7)
    end_time = t.toc()
    
    # inverse augmentations transformation
    output_SR0_ = inv_augment_img_tensor(output_SR0, mode=0)
    output_SR1_ = inv_augment_img_tensor(output_SR1, mode=1)
    output_SR2_ = inv_augment_img_tensor(output_SR2, mode=2)
    output_SR3_ = inv_augment_img_tensor(output_SR3, mode=3)
    output_SR4_ = inv_augment_img_tensor(output_SR4, mode=4)
    output_SR5_ = inv_augment_img_tensor(output_SR5, mode=5)
    output_SR6_ = inv_augment_img_tensor(output_SR6, mode=6)
    output_SR7_ = inv_augment_img_tensor(output_SR7, mode=7)
            
    output_SR = torch.stack((output_SR0_,output_SR1_,output_SR2_,output_SR3_,
                             output_SR4_,output_SR5_,output_SR6_,output_SR7_))
    output_SR = torch.mean(output_SR, dim=0)
    
    output_sr = output_SR.data.squeeze().float().cpu().clamp_(0, 255).numpy()
    output_sr = np.transpose(output_sr[[2, 1, 0], :, :], (1, 2, 0))
    test_results['time'].append(end_time)
    
    print('{:->4d}--> {:>10s}, time: {:.4f} sec.'.format(idx, base, end_time))
    
    # save images            
    cv2.imwrite('sr_results_x4/{:s}.png'.format(base), output_sr)
    
    del img_lr, img_LR, img_LR0, img_LR1, img_LR2, img_LR3, img_LR4, img_LR5, img_LR6, img_LR7
    del output_sr, output_SR, output_SR0, output_SR1, output_SR2, output_SR3, output_SR4, output_SR5, output_SR6, output_SR7
    del output_SR0_, output_SR1_, output_SR2_, output_SR3_, output_SR4_, output_SR5_, output_SR6_, output_SR7_
    torch.cuda.empty_cache()

avg_time = sum(test_results['time']) / len(test_results['time'])
print('Avg. Time:{:.4f}'.format(avg_time))