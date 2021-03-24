import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from data_loader import common
import random

class LRHRDataset(Dataset):
    '''
    Dataset for LR and HR images.
    '''

    def __init__(self, dataroot, is_train, scale, patch_size, rgb_range, noise_std):
        super(LRHRDataset, self).__init__()

        self.train = is_train
        self.scale = scale
        self.patch_size = patch_size
        self.rgb_range = rgb_range
        self.noise_std = noise_std
        self.paths_HR = None
        self.paths_LR = None
        dataroot_HR = []
        dataroot_LR = []
        
        if self.train:
            self.dataroot_hr = dataroot+'DF2K/generated/HR/'
            self.dataroot_lr = dataroot+'DF2K/generated/LR/'
            
            # read image list from image/binary files
            self.paths_imgs_hr = common.get_image_paths("img", self.dataroot_hr)
            self.paths_imgs_lr = common.get_image_paths("img", self.dataroot_lr)
            
            HR_list = dataroot_HR +  self.paths_imgs_hr
            LR_list = dataroot_LR + self.paths_imgs_lr 

        else:
            self.dataroot_hr = dataroot+'DF2K/valid/clean/'
            self.dataroot_lr = dataroot+'DF2K/valid/corrupted/'
            
            self.paths_imgs_hr = common.get_image_paths("img", self.dataroot_hr)
            self.paths_imgs_lr = common.get_image_paths("img", self.dataroot_lr)
            
            HR_list = dataroot_HR +  self.paths_imgs_hr
            LR_list = dataroot_LR + self.paths_imgs_lr
        
        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 3
        
        self.paths_HR = HR_list
        self.paths_LR = LR_list

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))
    
    def __getitem__(self, idx):
        lr, hr, lr_path, hr_path = self._load_file(idx)
        
        if self.train:
            lr, hr, sigma = self._get_patch(lr, hr)
        else:
            lr, hr, sigma = self._get_patch(lr, hr)
            
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.rgb_range)
        sigma_tensor = torch.tensor(sigma)
        return {'LR': lr_tensor, 'HR': hr_tensor, 'sigma':sigma_tensor, 'LR_path': lr_path, 'HR_path': hr_path}


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_HR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx
    
    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        lr = common.read_img(lr_path, 'img')
        hr = common.read_img(hr_path, 'img')

        return lr, hr, lr_path, hr_path
    
    def _get_patch(self, lr, hr):
        sigma = random.choice(self.noise_std)
        noise = 'G' + str(sigma)
        if self.train:
            LR_size = self.patch_size
            # random crop and augment
            lr, hr = common.get_patch(
                lr, hr, LR_size, self.scale)
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, noise)
        else:
            lr = common.add_noise(lr, noise)

        return lr, hr, sigma
    
def imshow(img):
    npimg = img.numpy()
    #print('npimg:', npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__ == '__main__':
    
    dataroot = "datasets/"
    
    is_train = True
    batch_size = 4
    scale = 4
    patch_size = 32
    rgb_range = 255
    noise_std = [0.0] # 7.65, 12.75
    
    train_dataset = LRHRDataset(dataroot, is_train, scale, patch_size, rgb_range, noise_std)
    
    print('trainset samples:', len(train_dataset))
    
      
    trainset_loader = DataLoader(train_dataset,
                                 shuffle=True,
                                 batch_size=batch_size,
                                 pin_memory=True,
                                 num_workers=0
                                 )
    print('#train_loader:', len(trainset_loader))
          
    for epoch in range(1):
        print('epoch:', epoch)
        for i, data in enumerate(trainset_loader):
            y, x, sigma, y_path, x_path  = data['LR'], data['HR'], data['sigma'], data['LR_path'], data['HR_path']

            print('GTs:', x.shape, x.min(), x.max())
            print('Input:', y.shape, y.min(), y.max())
            print('Sigma:', sigma)
            #print('lam:', lam)
            print('HR_path:', x_path)
            print('LR_path:', y_path)
        
            # show images
            plt.figure()
            imshow(torchvision.utils.make_grid(x/255))
            plt.figure()
            imshow(torchvision.utils.make_grid(y/255))
            break
