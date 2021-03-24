import torch
from torch import nn
import torch.nn.functional as F

class Wmad_estimator(nn.Module):
    """
    Standard deviation estimator using MAD upon wavelet coefficients
    """
    def __init__(self):
        super(Wmad_estimator, self).__init__()

        # DB7 high pass decomposition filter
        #self.db7_decomp_high = torch.Tensor([-0.48296291314469025, 0.836516303737469, -0.22414386804185735,
        #                                     -0.12940952255092145])[:,None]
        self.db7_decomp_high = torch.Tensor([-0.07785205408506236, 0.39653931948230575, -0.7291320908465551,
                                                      0.4697822874053586, 0.14390600392910627, -0.22403618499416572,
                                                      -0.07130921926705004, 0.0806126091510659, 0.03802993693503463,
                                                      -0.01657454163101562,-0.012550998556013784, 0.00042957797300470274,
                                                      0.0018016407039998328,0.0003537138000010399])[:,None]
        self.db7_decomp_high = self.db7_decomp_high[None, None, :]
        #self.db7_decomp_high = torch.stack([self.db7_decomp_high,self.db7_decomp_high,self.db7_decomp_high])


    def forward(self, x):
        if x.max() > 1:
            x  = x/255
        db7_decomp_high = self.db7_decomp_high
        if x.shape[1] > 1:
            db7_decomp_high = torch.cat([self.db7_decomp_high]*x.shape[1], dim=0)

        if x.is_cuda:
            db7_decomp_high = db7_decomp_high.cuda()

        #print('wmad x:', x.shape, x.min(), x.max())
        diagonal = F.pad(x, (0,0,self.db7_decomp_high.shape[2]//2,self.db7_decomp_high.shape[2]//2), mode='reflect')
        diagonal = F.conv2d(diagonal, db7_decomp_high, stride=(2,1), groups=x.shape[1])
        diagonal = F.pad(diagonal, (self.db7_decomp_high.shape[2]//2,self.db7_decomp_high.shape[2]//2,0,0), mode='reflect')
        diagonal = F.conv2d(diagonal.transpose(2,3), db7_decomp_high, stride=(2,1), groups=x.shape[1])
        #diagonal = diagonal.transpose(2,3)
        sigma = 0
        diagonal = diagonal.view(diagonal.shape[0],diagonal.shape[1],-1)
        for c in range(diagonal.shape[1]):
            d = diagonal[:,c]
            sigma += torch.median(torch.abs(d), dim=1)[0] / 0.6745
        sigma = sigma / diagonal.shape[1]
        sigma = sigma.detach()
        del db7_decomp_high
        return sigma
