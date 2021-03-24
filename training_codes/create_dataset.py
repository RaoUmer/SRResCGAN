import argparse
import os
import torch.utils.data
from models import DSGAN
from utils import utils_dsgan as utils
from PIL import Image
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(description='Apply the trained model to create a dataset')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to use')
parser.add_argument('--artifacts', default='', type=str, help='selecting different artifacts type')
parser.add_argument('--name', default='', type=str, help='additional string added to folder path')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--track', default='train', type=str, help='selecting train or valid track')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

# define input and target directories
input_target_hr_dir = 'datasets/DF2K/HR_target/' # HR_target
input_target_lr_dir = 'datasets/DF2K/LR_target/' # LR_target
target_hr_files = [os.path.join(input_target_hr_dir, x) for x in os.listdir(input_target_hr_dir) if utils.is_image_file(x)]
target_lr_files = [os.path.join(input_target_lr_dir, x) for x in os.listdir(input_target_lr_dir) if utils.is_image_file(x)]

tdsr_hr_dir = 'datasets/DF2K/generated/HR/'
tdsr_lr_dir = 'datasets/DF2K/generated/LR/'

if not os.path.exists(tdsr_hr_dir):
    os.makedirs(tdsr_hr_dir)
if not os.path.exists(tdsr_lr_dir):
    os.makedirs(tdsr_lr_dir)

# prepare neural networks
model_path = 'pretrained_nets/DSGAN/300_G.pth'
model_g = DSGAN.Generator(n_res_blocks=opt.num_res_blocks)
model_g.load_state_dict(torch.load(model_path), strict=True)
model_g.eval()
model_g = model_g.cuda()
print('# generator parameters:', sum(param.numel() for param in model_g.parameters()))

# generate the noisy images
idx = 0
with torch.no_grad():
    for file_hr, file_lr in zip(target_hr_files, target_lr_files):
        idx +=1
        print('Image No.:', idx)
        # load HR image
        input_img_hr = Image.open(file_hr)
        input_img_hr = TF.to_tensor(input_img_hr)

        # Save input_img as HR image for TDSR
        path = os.path.join(tdsr_hr_dir, os.path.basename(file_hr))
        TF.to_pil_image(input_img_hr).save(path, 'PNG')

        # load LR image
        input_img_lr = Image.open(file_lr)
        input_img_lr = TF.to_tensor(input_img_lr)

        # Apply model to generate the noisy resize_img
        if torch.cuda.is_available():
            input_img_lr = input_img_lr.unsqueeze(0).cuda()

        resize_noisy_img = model_g(input_img_lr).squeeze(0).cpu()

        # Save resize_noisy_img as LR image for TDSR
        path = os.path.join(tdsr_lr_dir, os.path.basename(file_lr))
        TF.to_pil_image(resize_noisy_img).save(path, 'PNG')
        #break
