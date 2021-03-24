import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.wmad_estimator import Wmad_estimator

class SRResDNet(nn.Module):
    def __init__(self, model, scale):
        super(SRResDNet, self).__init__()
        
        self.model = model
        self.upscale_factor = scale
        self.noise_estimator = Wmad_estimator()
        self.alpha = nn.Parameter(torch.Tensor(np.linspace(np.log(2),np.log(1), 1)))
        self.bbproj = nn.Hardtanh(min_val = 0., max_val = 255.)

    def forward(self, input):        
        # reconstruction block
        input = F.interpolate(input, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        #print('input:', input.shape, input.min(), input.max())
        
        # estimate sigma 
        sigma = self.noise_estimator(input)
        sigma *= 255.
        #print('estimated sigma:', sigma.shape, sigma)
        
        # model  
        output = self.model(input, sigma, self.alpha)
        #print('output:', output.shape, output.min(), output.max())
        
        # residual ouput
        output = input - output
        #print('residual output:', output.shape, output.min(), output.max())
        
        # clipping layer
        output = self.bbproj(output)
        #print('clipping output:', output.shape, output.min(), output.max())
        
        return output