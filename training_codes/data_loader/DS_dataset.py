from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from data_loader import utils

import numpy as np
import torchvision
import yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, noisy_dir, crop_size, upscale_factor=4, cropped=False, flips=False, rotations=False, **kwargs):
        super(TrainDataset, self).__init__()
        # get all directories used for training
        if isinstance(noisy_dir, str):
            noisy_dir = [noisy_dir]
        self.files = []
        for n_dir in noisy_dir:
            self.files += [join(n_dir, x) for x in listdir(n_dir) if utils.is_image_file(x)]
        # intitialize image transformations and variables
        self.input_transform = T.Compose([
            T.RandomVerticalFlip(0.5 if flips else 0.0),
            T.RandomHorizontalFlip(0.5 if flips else 0.0),
            T.RandomCrop(crop_size)
        ])
        self.crop_transform = T.RandomCrop(crop_size // upscale_factor)
        self.upscale_factor = upscale_factor
        self.cropped = cropped
        self.rotations = rotations
        self.repeat = 3

    def __getitem__(self, idx):
        # get downscaled and cropped image (if necessary)
        idx = self._get_index(idx)
        noisy_image = self.input_transform(Image.open(self.files[idx]))
        if self.rotations:
            angle = random.choice([0, 90, 180, 270])
            noisy_image = TF.rotate(noisy_image, angle)
        if self.cropped:
            cropped_image = self.crop_transform(noisy_image)
        noisy_image = TF.to_tensor(noisy_image)
        resized_image = utils.imresize(noisy_image, 1.0 / self.upscale_factor, True)
        if self.cropped:
            return resized_image, TF.to_tensor(cropped_image)
        else:
            return resized_image

    def __len__(self):
        return (len(self.files) * self.repeat) + 250
    
    def _get_index(self, idx):
        return idx % len(self.files)


class DiscDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor=4, flips=False, rotations=False, **kwargs):
        super(DiscDataset, self).__init__()
        self.files = [join(dataset_dir, x) for x in listdir(dataset_dir) if utils.is_image_file(x)]
        self.input_transform = T.Compose([
            T.RandomVerticalFlip(0.5 if flips else 0.0),
            T.RandomHorizontalFlip(0.5 if flips else 0.0),
            T.RandomCrop(crop_size // upscale_factor)
        ])
        self.rotations = rotations

    def __getitem__(self, index):
        # get real image for discriminator (same as cropped in TrainDataset)
        image = self.input_transform(Image.open(self.files[index]))
        if self.rotations:
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
        return TF.to_tensor(image)

    def __len__(self):
        return len(self.files)


class ValDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, upscale_factor, crop_size_val, **kwargs):
        super(ValDataset, self).__init__()
        self.hr_files = [join(hr_dir, x) for x in listdir(hr_dir) if utils.is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.crop_size = crop_size_val
        if lr_dir is None:
            self.lr_files = None
        else:
            self.lr_files = [join(lr_dir, x) for x in listdir(lr_dir) if utils.is_image_file(x)]

    def __getitem__(self, index):
        # get downscaled, cropped and gt (if available) image
        hr_image = Image.open(self.hr_files[index])
        w, h = hr_image.size
        cs = utils.calculate_valid_crop_size(min(w, h), self.upscale_factor)
        if self.crop_size is not None:
            cs = min(cs, self.crop_size)
        cropped_image = TF.to_tensor(T.CenterCrop(cs // self.upscale_factor)(hr_image))
        hr_image = T.CenterCrop(cs)(hr_image)
        hr_image = TF.to_tensor(hr_image)
        resized_image = utils.imresize(hr_image, 1.0 / self.upscale_factor, True)
        if self.lr_files is None:
            return resized_image, cropped_image, resized_image
        else:
            lr_image = Image.open(self.lr_files[index])
            lr_image = TF.to_tensor(T.CenterCrop(cs // self.upscale_factor)(lr_image))
            return resized_image, lr_image

    def __len__(self):
        return len(self.hr_files)

def imshow(img):
    npimg = img.numpy()
    #print('npimg:', npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__=='__main__':
    with open('paths.yml', 'r') as stream:
        PATHS = yaml.load(stream)
    opts_train = dict(crop_size=512, 
                      upscale_factor=4)
    
    opts_disc = dict(crop_size=512, 
                      upscale_factor=4)
    
    opts_val = dict(crop_size_val=256,
                    upscale_factor=4)

    train_set = TrainDataset(PATHS['df2k']['target'], cropped=False, **(opts_train))
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=4, shuffle=True)
    print('train_set samples:', len(train_set))
    print('#train_loader:', len(train_loader))
    
    disc_set = DiscDataset(PATHS['df2k']['source'], **(opts_disc))
    disc_loader = DataLoader(dataset=disc_set, num_workers=0, batch_size=4, shuffle=True)
    print('disc_set samples:', len(disc_set))
    print('#disc_loader:', len(disc_loader))
    
    val_set = ValDataset(PATHS['df2k']['valid']['clean'], PATHS['df2k']['valid']['corrupted'], **(opts_val))
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=4, shuffle=False)
    print('val_set samples:', len(val_set))
    print('#val_loader:', len(val_loader))
    
    counter = 0
#    for epoch in range(1):
#        print('epoch:', epoch)
#        for input_img, disc_img in zip(train_loader, disc_loader):
#            counter += 1
#            print('#count:', counter)
#            print('target:', input_img.shape, input_img.min(), input_img.max())
#            print('source:', disc_img.shape, disc_img.min(), disc_img.max())
#        
#            # show images
#            plt.figure()
#            imshow(torchvision.utils.make_grid(input_img))
#            plt.figure()
#            imshow(torchvision.utils.make_grid(disc_img))
#            break
    
    for epoch in range(1):
        print('epoch:', epoch)
        for _, data in enumerate(val_loader):
            counter += 1
            input_img, disc_img = data
            print('#count:', counter)
            print('target:', input_img.shape, input_img.min(), input_img.max())
            print('source:', disc_img.shape, disc_img.min(), disc_img.max())
        
            # show images
            plt.figure()
            imshow(torchvision.utils.make_grid(input_img))
            plt.figure()
            imshow(torchvision.utils.make_grid(disc_img))
            break
