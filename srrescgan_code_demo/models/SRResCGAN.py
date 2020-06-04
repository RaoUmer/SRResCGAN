import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=True)

class BasicBlock(nn.Module):
    """
    Residual BasicBlock 
    """
    def __init__(self, inplanes, planes, stride=1, weightnorm=None, shortcut=True):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.relu1 = nn.PReLU(num_parameters=planes,init=0.1)
        self.relu2 = nn.PReLU(num_parameters=planes, init=0.1)
        self.conv2 = conv3x3(inplanes, planes, stride)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)

    def forward(self, x):
        out = self.relu1(x)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv1(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        out = self.relu2(out)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv2(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        if self.shortcut:
            out = x + out
        return out

class L2Proj(nn.Module):
    """
    L2Proj layer
    source link: https://github.com/cig-skoltech/deep_demosaick/blob/master/l2proj.py
    """
    def __init__(self):
        super(L2Proj, self).__init__()

    def forward(self, x, stdn, alpha):
        if x.is_cuda:
            x_size = torch.cuda.FloatTensor(1).fill_(x.shape[1] * x.shape[2] * x.shape[3])
        else:
            x_size = torch.Tensor([x.shape[1] * x.shape[2] * x.shape[3]])
        numX = torch.sqrt(x_size-1)
        if x.is_cuda:
            epsilon = torch.cuda.FloatTensor(x.shape[0],1,1,1).fill_(1) * (torch.exp(alpha) * stdn * numX)[:,None,None,None]
        else:
            epsilon = torch.zeros(x.size(0),1,1,1).fill_(1) * (torch.exp(alpha) *  stdn * numX)[:,None,None,None]
        x_resized = x.view(x.shape[0], -1)
        x_norm = torch.norm(x_resized, 2, dim=1).reshape(x.size(0),1,1,1)
        max_norm = torch.max(x_norm, epsilon)
        result = x * (epsilon / max_norm)
        
        return result

class Wmad_estimator(nn.Module):
    """
    Standard deviation estimator using MAD upon wavelet coefficients
    source link: https://github.com/cig-skoltech/deep_demosaick/blob/master/modules/wmad_estimator.py
    """
    def __init__(self):
        super(Wmad_estimator, self).__init__()

        # DB7 high pass decomposition filter
        self.db7_decomp_high = torch.Tensor([-0.07785205408506236, 0.39653931948230575, -0.7291320908465551,
                                                      0.4697822874053586, 0.14390600392910627, -0.22403618499416572,
                                                      -0.07130921926705004, 0.0806126091510659, 0.03802993693503463,
                                                      -0.01657454163101562,-0.012550998556013784, 0.00042957797300470274,
                                                      0.0018016407039998328,0.0003537138000010399])[:,None]
        self.db7_decomp_high = self.db7_decomp_high[None, None, :]


    def forward(self, x):
        if x.max() > 1:
            x  = x/255
        db7_decomp_high = self.db7_decomp_high
        if x.shape[1] > 1:
            db7_decomp_high = torch.cat([self.db7_decomp_high]*x.shape[1], dim=0)

        if x.is_cuda:
            db7_decomp_high = db7_decomp_high.cuda()

        diagonal = F.pad(x, (0,0,self.db7_decomp_high.shape[2]//2,self.db7_decomp_high.shape[2]//2), mode='reflect')
        diagonal = F.conv2d(diagonal, db7_decomp_high, stride=(2,1), groups=x.shape[1])
        diagonal = F.pad(diagonal, (self.db7_decomp_high.shape[2]//2,self.db7_decomp_high.shape[2]//2,0,0), mode='reflect')
        diagonal = F.conv2d(diagonal.transpose(2,3), db7_decomp_high, stride=(2,1), groups=x.shape[1])
        
        sigma = 0
        diagonal = diagonal.view(diagonal.shape[0],diagonal.shape[1],-1)
        for c in range(diagonal.shape[1]):
            d = diagonal[:,c]
            sigma += torch.median(torch.abs(d), dim=1)[0] / 0.6745
        sigma = sigma / diagonal.shape[1]
        sigma = sigma.detach()
        del db7_decomp_high
        
        return sigma

class ResCNet(nn.Module):
    """
    Residual Convolutional Net
    """
    def __init__(self, depth=5, color=True, weightnorm=True):
        self.inplanes = 64
        super(ResCNet, self).__init__()
        if color:
            in_channels = 3
        else:
            in_channels = 1

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=0,
                               bias=True)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)

        # Resnet blocks layer
        self.layer1 = self._make_layer(BasicBlock, 64, depth)
        self.conv_out = nn.ConvTranspose2d(64, in_channels, kernel_size=5, stride=1, padding=2, bias=True)
        if weightnorm:
            self.conv_out = weight_norm(self.conv_out)

        self.l2proj = L2Proj()
        self.zeromean()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, weightnorm=True, shortcut=False))
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, weightnorm=True, shortcut=True))
        return nn.Sequential(*layers)

    def zeromean(self):
        # Function zeromean subtracts the mean E(f) from filters f
        # in order to create zero mean filters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)

    def forward(self, x, stdn, alpha):
        self.zeromean()
        out = F.pad(x,(2,2,2,2),'reflect')
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.conv_out(out)
        out = self.l2proj(out, stdn, alpha)
        return out

class Generator(nn.Module):
    """
    SRResCGAN: Generator (G_{SR})
    """
    def __init__(self, scale):
        super(Generator, self).__init__()
        self.model = ResCNet()
        self.upscale_factor = scale
        self.noise_estimator = Wmad_estimator()
        self.alpha = nn.Parameter(torch.Tensor(np.linspace(np.log(2),np.log(1), 1)))
        self.bbproj = nn.Hardtanh(min_val = 0., max_val = 255.)

    def forward(self, input):        
        # upsampling
        input = F.interpolate(input, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        
        # estimate sigma 
        sigma = self.noise_estimator(input)
        sigma *= 255.
        
        # Rescnet model  
        output = self.model(input, sigma, self.alpha)
        
        # residual ouput
        output = input - output
        
        # clipping layer
        output = self.bbproj(output)
        
        return output

if __name__ == "__main__":    
    input = torch.randn(2,3,50,50).type(torch.FloatTensor)
    sf = 4

    model = Generator(sf)
    print(model)
    
    s = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Number of model params: %d' % s)
    
    output = model(input)
    print('output:', output.shape)