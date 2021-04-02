import torch as th
import numpy as np
from scipy.fftpack import dct, dctn
from functools import reduce
import math


'''
modified from https://github.com/cig-skoltech

'''

def reverse(input,dim=0) :
    r"""Reverses the specified dimension of the input tensor."""
    Dims = input.dim()
    assert (dim < Dims), "The selected dimension (arg 2) exceeds the tensor's dimensions."
    idx = th.arange(input.size(dim)-1,-1,-1).type_as(input).long()
    return input.index_select(dim,idx)

def log10(input):
    return input.log().div(th.Tensor([10]).type_as(input).log())
    
def psnr(input,other,peakVal = None, average = False, nargout = 1):
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"The first two inputs "\
    +"must be tensors."
    
    if peakVal is None:
        peakVal = other.max()
    
    assert(input.shape == other.shape), "Dimensions mismatch between the two "\
    "input tensors."
    
    while input.dim() < 4:
        input = input.unsqueeze(0)
    while other.dim() < 4:
        other = other.unsqueeze(0)
    
    N = input.numel()
    batch = input.size(0)
    
    if N == 0:
        SNR = float('nan')
        MSE = 0
        return SNR, MSE
    
    MSE = (input-other).view(batch,-1).pow(2).mean(dim=1)
    
    #SNR = (10*th.log(peakVal**2/MSE)).div(math.log(10))
    SNR = 10*log10(peakVal**2/MSE)
    
    if average:
        return (SNR.mean(), MSE.mean()) if nargout == 2 else SNR.mean()
    else:
        return (SNR, MSE) if nargout == 2 else SNR


def periodicPad2D(input,pad = 0):
    r"""Pads circularly the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""
          
    # pad = [top,bottom,left,right]
    
    if isinstance(pad,int):
        assert(pad >= 0), """Pad must be either a non-negative integer 
        or a tuple."""
        pad = (pad,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)
             
    assert(isinstance(pad,tuple) and len(pad) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (pad[0] >= 0 and pad[1] >= 0 and pad[2] >= 0 and pad[3] >= 0), \
            "Padding must be non-negative in each dimension."
            
    assert(pad[0] <= sz[-2] and pad[1] <= sz[-2] and \
           pad[2] <= sz[-1] and pad[3] <= sz[-1]), \
    "The padding values exceed the tensor's dimensions."
    
    sz[-1] = sz[-1] + sum(pad[2::])
    sz[-2] = sz[-2] + sum(pad[0:2])
    
    out = th.empty(sz).type_as(input)
    
    # Copy the original tensor to the central part
    out[...,pad[0]:out.size(-2)-pad[1], \
        pad[2]:out.size(-1)-pad[3]] = input
    
    # Pad Top
    if pad[0] != 0:
        out[...,0:pad[0],:] = out[...,out.size(-2)-pad[1]-pad[0]:out.size(-2)-pad[1],:]
    
    # Pad Bottom
    if pad[1] != 0:
        out[...,out.size(-2)-pad[1]::,:] = out[...,pad[0]:pad[0]+pad[1],:]
    
    # Pad Left
    if pad[2] != 0:
        out[...,:,0:pad[2]] = out[...,:,out.size(-1)-pad[3]-pad[2]:out.size(-1)-pad[3]]
    
    # Pad Right
    if pad[3] != 0:
        out[...,:,out.size(-1)-pad[3]::] = out[...,:,pad[2]:pad[2]+pad[3]]    
    
    if sflag:
        out.squeeze_()
        
    return out

def periodicPad_transpose2D(input,crop = 0):
    r"""Adjoint of the periodicPad2D operation which amounts to a special type
    of cropping. CROP specifies the amount of cropping as [TOP, BOTTOM, LEFT, RIGHT].
    If crop is an integer then each direction is cropped by the same amount. In
    order to achieve a different amount of cropping in each direction of the 
    tensor, crop needs to be a tuple."""          
    
    # crop = [top,bottom,left,right]
    
    if isinstance(crop,int):
        assert(crop >= 0), """Crop must be either a non-negative integer 
        or a tuple."""
        crop = (crop,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)        
             
    assert(isinstance(crop,tuple) and len(crop) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (crop[0] >= 0 and crop[1] >= 0 and crop[2] >= 0 and crop[3] >= 0), \
            "Crop must be non-negative in each dimension."    
    
    assert (crop[0] + crop[1] <= sz[-2] and crop[2] + crop[3] <= sz[-1]), \
            "Crop does not have valid values."
    
    out = input.clone()
    
    # Top
    if crop[0] != 0:
        out[...,crop[0]:crop[0]+crop[1],:] += out[...,-crop[1]::,:]
    
    # Bottom 
    if crop[1] != 0:
        out[...,-crop[0]-crop[1]:-crop[1],:] += out[...,0:crop[0],:]
    
    # Left 
    if crop[2] != 0:
        out[...,crop[2]:crop[2]+crop[3]] += out[...,-crop[3]::]
    
    # Right
    if crop[3] != 0:
        out[...,-crop[2]-crop[3]:-crop[3]] += out[...,0:crop[2]]
    
    if crop[1] == 0:
        end_h = sz[-2]+1 
    else:
        end_h = sz[-2]-crop[1]
        
    if crop[3] == 0:
        end_w = sz[-1]+1
    else:
        end_w = sz[-1]-crop[3]
        
    out = out[...,crop[0]:end_h,crop[2]:end_w]
    
    if sflag:
        out.squeeze_()
        
    return out


def zeroPad2D(input,pad = 0):
    r"""Pads with zeros the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""

    # pad = [top,bottom,left,right]
    
    if isinstance(pad,int):
        assert(pad >= 0), """Pad must be either a non-negative integer 
        or a tuple."""
        pad = (pad,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)
        
    assert(isinstance(pad,tuple) and len(pad) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (pad[0] >= 0 and pad[1] >= 0 and pad[2] >= 0 and pad[3] >= 0), \
            "Padding must be non-negative in each dimension."
            
    assert(pad[0] <= sz[-2] and pad[1] <= sz[-2] and \
           pad[2] <= sz[-1] and pad[3] <= sz[-1]), \
    "The padding values exceed the tensor's dimensions."    
    
    sz[-1] = sz[-1] + sum(pad[2::])
    sz[-2] = sz[-2] + sum(pad[0:2])
    
    out = th.zeros(sz).type_as(input)
    out[...,pad[0]:sz[-2]-pad[1]:1,pad[2]:sz[-1]-pad[3]:1] = input
    
    if sflag:
        out.squeeze_()
    
    return out

def crop2D(input,crop):
    r"""Cropping the spatial dimensions (last two dimensions) of the 
    input tensor. This is the adjoint operation of zeroPad2D. Crop specifies 
    the amount of cropping as [TOP, BOTTOM, LEFT, RIGHT]. If crop is an integer 
    then each direction is cropped by the same amount. In order to achieve a 
    different amount of cropping in each direction of the  tensor, crop needs 
    to be a tuple."""    
    
    if isinstance(crop,int):
        assert(crop >= 0), """Crop must be either a non-negative integer 
        or a tuple."""
        crop = (crop,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)        
             
    assert(isinstance(crop,tuple) and len(crop) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())    
    
    assert (crop[0] >= 0 and crop[1] >= 0 and crop[2] >= 0 and crop[3] >= 0), \
            "Crop must be non-negative in each dimension."    
    
    assert (crop[0] + crop[1] <= sz[-2] and crop[2] + crop[3] <= sz[-1]), \
            "Crop does not have valid values."
    
    out = input[...,crop[0]:sz[-2]-crop[1]:1,crop[2]:sz[-1]-crop[3]:1]
    
    if sflag:
        out.unsqueeze_()
    
    return out
    
def symmetricPad2D(input,pad = 0):
    r"""Pads symmetrically the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""
          
    # pad = [top,bottom,left,right]
    
    if isinstance(pad,int):
        assert(pad >= 0), """Pad must be either a non-negative integer 
        or a tuple."""
        pad = (pad,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)
             
    assert(isinstance(pad,tuple) and len(pad) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (pad[0] >= 0 and pad[1] >= 0 and pad[2] >= 0 and pad[3] >= 0), \
            "Padding must be non-negative in each dimension."
            
    assert(pad[0] <= sz[-2] and pad[1] <= sz[-2] and \
           pad[2] <= sz[-1] and pad[3] <= sz[-1]), \
    "The padding values exceed the tensor's dimensions."
    
    sz[-1] = sz[-1] + sum(pad[2::])
    sz[-2] = sz[-2] + sum(pad[0:2])
    
    out = th.zeros(sz).type_as(input)
    
    # Copy the original tensor to the central part
    out[...,pad[0]:out.size(-2)-pad[1], \
        pad[2]:out.size(-1)-pad[3]] = input
    
    # Pad Top
    if pad[0] != 0:
        out[...,0:pad[0],:] = reverse(out[...,pad[0]:2*pad[0],:],-2)
    
    # Pad Bottom
    if pad[1] != 0:
        out[...,out.size(-2)-pad[1]::,:] = reverse(out[...,out.size(-2)
            -2*pad[1]:out.size(-2)-pad[1],:],-2)
    
    # Pad Left
    if pad[2] != 0:
        out[...,:,0:pad[2]] = reverse(out[...,:,pad[2]:2*pad[2]],-1)
    
    # Pad Right
    if pad[3] != 0:
        out[...,:,out.size(-1)-pad[3]::] = reverse(out[...,:,out.size(-1)
            -2*pad[3]:out.size(-1)-pad[3]],-1)    
    
    if sflag:
        out.squeeze_()
        
    return out


def symmetricPad_transpose2D(input,crop = 0):
    r"""Adjoint of the SymmetricPad2D operation which amounts to a special type
    of cropping. CROP specifies the amount of cropping as [TOP, BOTTOM, LEFT, RIGHT].
    If crop is an integer then each direction is cropped by the same amount. In
    order to achieve a different amount of cropping in each direction of the 
    tensor, crop needs to be a tuple."""          
    
    # crop = [top,bottom,left,right]
    
    if isinstance(crop,int):
        assert(crop >= 0), """Crop must be either a non-negative integer 
        or a tuple."""
        crop = (crop,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)        
             
    assert(isinstance(crop,tuple) and len(crop) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (crop[0] >= 0 and crop[1] >= 0 and crop[2] >= 0 and crop[3] >= 0), \
            "Crop must be non-negative in each dimension."    
    
    assert (crop[0] + crop[1] <= sz[-2] and crop[2] + crop[3] <= sz[-1]), \
            "Crop does not have valid values."
    
    out = input.clone()
    
    # Top
    if crop[0] != 0:
        out[...,crop[0]:2*crop[0],:] += reverse(out[...,0:crop[0],:],-2)
    
    # Bottom 
    if crop[1] != 0:
    # out[...,sz[-2]-2*crop[1]:sz[-2]-crop[1],:] += reverse(out[...,sz[-2]-crop[1]::,:],-2) 
        out[...,-2*crop[1]:-crop[1],:] += reverse(out[...,-crop[1]::,:],-2) 
    
    # Left 
    if crop[2] != 0:
        out[...,crop[2]:2*crop[2]] += reverse(out[...,0:crop[2]],-1)
    
    # Right
    if crop[3] != 0:
    # out[...,sz[-1]-2*crop[3]:sz[-1]-crop[3],:] += reverse(out[...,sz[-1]-crop[3]::,:],-1) 
        out[...,-2*crop[3]:-crop[3]] += reverse(out[...,-crop[3]::],-1) 
    
    if crop[1] == 0:
        end_h = sz[-2]+1 
    else:
        end_h = sz[-2]-crop[1]
        
    if crop[3] == 0:
        end_w = sz[-1]+1
    else:
        end_w = sz[-1]-crop[3]
        
    out = out[...,crop[0]:end_h,crop[2]:end_w]
    
    
    if sflag:
        out.squeeze_()
        
    return out

def pad2D(input,pad=0,padType='zero'):
    r"""Pads the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple. PadType specifies the type of padding.
    Valid padding types are "zero","symmetric" and "periodic". """
    
    pad = formatInput2Tuple(pad,int,4)
    
    if sum(pad) == 0:
        return input
    
    if padType == 'zero':
        return zeroPad2D(input,pad)
    elif padType == 'symmetric':
        return symmetricPad2D(input,pad)
    elif padType == 'periodic':
        return periodicPad2D(input,pad)
    else:
        raise NotImplementedError("Unknown padding type.")

def pad_transpose2D(input,pad=0,padType='zero'):
    r"""Transpose operation of pad2D. PAD specifies the amount of padding as 
    [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple. PadType specifies the type of padding.
    Valid padding types are "zero" and "symmetric". """    
    
    pad = formatInput2Tuple(pad,int,4)
    
    if sum(pad) == 0:
        return input
    
    if padType == 'zero':
        return crop2D(input,pad)
    elif padType == 'symmetric':
        return symmetricPad_transpose2D(input,pad)
    elif padType == 'periodic':
        return periodicPad_transpose2D(input,pad)
    else:
        raise NotImplementedError("Uknown padding type.")
    

def signum(input):
    out = - th.ones_like(input)
    out[input > 0] = 1
    return out

def dctmtx(support,dtype):
    D = np.eye(support,dtype = dtype)
    return th.from_numpy(dct(D, axis = 0, norm = 'ortho'))

def gen_dct2_kernel(support,dtype = 'f', GPU = False, nargout = 1):
    r"""If two output arguments are returned then:
    h1 has dimensions [Px,1,Px,1] and h2 has dimensions [Py,1,1,Py] where
    Px = support[0] and  Py = support[1].
    
    If a single output argument is returned then:
    h has dimensions [Px*Py 1 Px Py].
    
    support : Either an integer or a tuple
    
    Usage example :
    
    x = th.randn(1,1,16,16).double()
    h = utils.gen_dct2_kernel(8,'d',nargout = 1)
    Dx = th.conv2d(x,h,stride = (8,8)) % it computes the 2D DCT of each 
    % non-overlapping block of size 8x8
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = th.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())
    
    Usage example 2:
    
    x = th.randn(1,1,16,16).double()
    h1,h2 = utils.gen_dct2_kernel(8,'d',nargout = 2)
    Dx = th.conv2d(x,h1,stride = (8,1)) % it computes the 1D DCT of each 
    % non-overlapping block of size 8x1
    Dx = Dx.view(8,1,2,16)
    Dx = th.conv2d(Dx,h2,stride = (1,8)) % it computes the 1D DCT of each
    % non-overlapping block of size 1x8
    Dx = Dx.view(1,64,2,2)
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = th.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())    
    
    """
    assert(nargout == 1 or nargout ==2), "One or two output arguments "\
    "are expected."
    
    
    if isinstance(support,int):
        support = (support,support)
    
    if len(support) < 2:
        support = support * 2
        
    if nargout == 2:
        D = dctmtx(support[0],dtype)
        h1 = D.view(support[0],1,support[0],1)
        if support[1] != support[0]:
            D = dctmtx(support[1],dtype)
        h2 = D.view(support[1],1,1,support[1])
        
        if th.cuda.is_available and GPU:
            h1 = h1.cuda()
            h2 = h2.cuda()
                   
        return h1,h2
    else :
        h = np.zeros((reduce(lambda x, y: x*y, support[0:2]),1,support[0],support[1]),dtype)
        dirac = np.zeros(support[0:2],dtype)
        for k in np.arange(support[0]):
            for l in np.arange(support[1]):
                dirac[k,l] = 1;
                h[:,0,k,l] = dctn(dirac,norm = 'ortho').flatten()
                dirac[k,l] = 0
        
        h = th.from_numpy(h)
        if th.cuda.is_available and GPU:
            h = h.cuda()
        
        return h

def gen_dct3_kernel(support,dtype = 'f', GPU = False, nargout = 1):
    r"""If three output arguments are returned then:
    h1 has dimensions [Pz,Pz,1,1], h2 has dimensions [Px,1,Px,1] and 
    h3 has dimensions [Py,1,1,Py] where Pz = support[0], Px = support[1] and 
    Py = support[2].
    
    If two output arguments are returned then:
    h1 has dimensions [Pz,Pz,1,1] and h2 has dimensions [Px*Py,1,Px,Py].
    
    If a single output argument is returned then:
    h has dimensions [Px*Py*Pz Pz Px Py].
    
    support : Either an integer or a tuple 
    
    Usage example :
    from scipy.fftpack import dctn
    x = th.randn(1,3,16,16).double()
    h = utils.gen_dct3_kernel((3,8,8),'d',nargout = 1)
    Dx = th.conv2d(x,h,stride = (8,8)) % it computes the 3D DCT of each 
    % non-overlapping block of size 3x8x8
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = th.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())
    
    Usage example 2:
    
    x = th.randn(1,3,16,16).double()
    h1,h2 = utils.gen_dct3_kernel((3,8,8),'d',nargout = 2)
    Dx = th.conv2d(x,h1,stride = 1) % it computes the 1D DCT along the 3rd  
    % dimension.
    Dx = th.conv2d(Dx.view(3,1,16,16),h2,stride = (8,8)) % it computes the 2D 
    % DCT along the spatial dimensions
    Dx = Dx.view(1,3*64,2,2)
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = th.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())

    Usage example 3:
    
    x = th.randn(1,3,16,16).double()
    h1,h2,h3 = utils.gen_dct3_kernel((3,8,8),'d',nargout = 3)
    Dx = th.conv2d(x,h1,stride = 1) % it computes the 1D DCT along the 3rd 
    % dimension. 
    Dx = th.conv2d(Dx.view(3,1,16,16),h2,stride = (8,1)) % it computes the 1D 
    % DCT along the first spatial dimension
    Dx = th.conv2d(Dx.view(3*8,1,2,16),h3,stride = (1,8)) % it computes the 1D DCT along the channel
    % dimension (of size 3)
    Dx = Dx.view(1,24*8,2,2)
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = th.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())
    
    """
    assert(nargout == 1 or nargout ==2 or nargout == 3), "From one to three "\
    "output arguments are expected."
    
    
    if isinstance(support,int):
        support = (support,support,support)
    
    if len(support) < 2:
        support = support * 3
    
    if len(support) < 3:
        support = (1,)+support
    
    if nargout == 3:
        D = dctmtx(support[0],dtype)
        h1 = D.view(support[0],support[0],1,1)
        if support[1] != support[0]:
            D = dctmtx(support[1],dtype)
        h2 = D.view(support[1],1,support[1],1)
        if support[2] != support[1]:
            D = dctmtx(support[2],dtype)
        h3 = D.view(support[2],1,1,support[2])
        
        if th.cuda.is_available and GPU:
            h1 = h1.cuda()
            h2 = h2.cuda()
            h3 = h3.cuda()
                   
        return h1,h2,h3            
    elif nargout == 2:
        D = dctmtx(support[0],dtype)
        h1 = D.view(support[0],support[0],1,1)

        h2 = np.zeros((reduce(lambda x, y: x*y, support[1:3]),1,support[1],support[2]),dtype)
        dirac = np.zeros(support[1:3],dtype)
        for k in np.arange(support[1]):
            for l in np.arange(support[2]):
                dirac[k,l] = 1;
                h2[:,0,k,l] = dctn(dirac,norm = 'ortho').flatten()
                dirac[k,l] = 0
        
        h2 = th.from_numpy(h2)        
        
        if th.cuda.is_available and GPU:
            h1 = h1.cuda()
            h2 = h2.cuda()
                   
        return h1,h2
    else :
        h = np.zeros((reduce(lambda x, y: x*y, support[0:3]),support[0],support[1],support[2]),dtype)
        dirac = np.zeros(support[0:3],dtype)
        for k in np.arange(support[0]):
            for l in np.arange(support[1]):
                for m in np.arange(support[2]):
                    dirac[k,l,m] = 1;
                    h[:,k,l,m] = dctn(dirac,norm = 'ortho').flatten()
                    dirac[k,l,m] = 0
        
        h = th.from_numpy(h)
        if th.cuda.is_available and GPU:
            h = h.cuda()
        
        return h


def __shift(x,s,bc='circular'):
    """ Shift operator that can treat different boundary conditions. It applies 
    to a tensor of arbitrary dimensions. 
    ----------
    Usage: xs = shift(x,(0,1,-3,3),'reflexive')
    ----------
    Parameters
    ----------
    x : tensor.
    s : tuple that matches the dimensions of x, with the corresponding shifts.
    bc: String with the prefered boundary conditions (bc='circular'|'reflexive'|'zero')
        (Default: 'circular')
    """
    
    if not isinstance(bc, str):
        raise Exception("bc must be of type string")
       
    if not reduce(lambda x,y : x and y, [isinstance(k,int) for k in s]):
        raise Exception("s must be a tuple of ints")
           
    if len(s) < x.dim():
       s = s + (0,) * (x.dim()-len(s))        
    elif len(s) > x.dim():
        print("The shift values will be truncated to match the " \
        +"dimensions of the input tensor. The trailing extra elements will" \
        +" be discarded.")
        s = s[0:x.dim()]
    
    if reduce(lambda x,y : x or y, [ math.fabs(s[i]) > x.shape[i] for i in range(x.dim())]):
        raise Exception("The shift steps should not exceed in absolute values"\
        +" the size of the corresponding dimensions.")

    # use a list sequence instead of a tuple since the latter is an 
    # immutable sequence and cannot be altered         
    indices = [slice(0,x.shape[0])]
    for i in range(1,x.dim()):
        indices.append(slice(0,x.shape[i]))
        
    if bc == 'circular':
        xs = x[:] # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                m = x.shape[i]
                idx = indices[:]                
                idx[i] = (np.arange(0,m)-s[i])%m
                xs = xs[tuple(idx)]
    elif bc == 'reflexive':
        xs = x[:] # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                idx = indices[:]
                if s[i] > 0: # right shift                    
                    idx[i] = list(range(s[i]-1,-1,-1)) + list(range(0,x.shape[i]-s[i]))
                else: # left shift
                    idx[i] = list(range(-s[i],x.shape[i])) + \
                    list(range(x.shape[i]-1,x.shape[i]+s[i]-1,-1))
                
                xs = xs[tuple(idx)]
    elif bc == 'zero':
        xs=th.zeros_like(x)        
        idx_x=indices[:]
        idx_xs=indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:       
                if s[i] > 0: # right shift
                    idx_x[i] = slice(0,x.shape[i]-s[i])
                    idx_xs[i] = slice(s[i],x.shape[i])
                else: # left shift
                    idx_x[i] = slice(-s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]+s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]
        
    else:
        raise Exception("Unknown boundary conditions")
    
    return xs

def __shift_transpose(x,s,bc='circular'):
        
    r""" Transpose of the shift operator that can treat different boundary conditions. 
    It applies to a tensor of arbitrary dimensions. 
    ----------
    Usage: xs = shift_transpose(x,(0,1,-3,3),'reflexive')
    ----------
    Parameters
    ----------
    x : tensor.
    s : tuple that matches the dimensions of x, with the corresponding shifts.
    bc: String with the prefered boundary conditions (bc='circular'|'reflexive'|'zero')
        (Default: 'circular')
    """   
    
    if not isinstance(bc, str):
        raise Exception("bc must be of type string")
       
    if not reduce(lambda x,y : x and y, [isinstance(k,int) for k in s]):
        raise Exception("s must be a tuple of ints")
           
    if len(s) < x.dim():
       s = s + (0,)* (x.dim()-len(s))        
    elif len(s) > x.dim():
        print("The shift values will be truncated to match the " \
        +"dimensions of the input tensor. The trailing extra elements will" \
        +" be discarded.")
        s = s[0:x.dim()]
    
    if reduce(lambda x,y : x or y, [ math.fabs(s[i]) > x.shape[i] for i in range(x.dim())]):
        raise Exception("The shift steps should not exceed in absolute values"\
        +" the size of the corresponding dimensions.")
        
    # use a list sequence instead of a tuple since the latter is an 
    # immutable sequence and cannot be altered 
    indices=[slice(0,x.shape[0])]
    for i in range(1,x.dim()):
        indices.append(slice(0,x.shape[i]))
        
    if bc == 'circular':
        xs = x[:] # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                m = x.shape[i]
                idx = indices[:]                
                idx[i] = (np.arange(0,m)+s[i])%m
                xs = xs[tuple(idx)]
    elif bc == 'reflexive':
        y=x[:]
        for i in range(x.dim()):
            xs = th.zeros_like(x)
            idx_x_a = indices[:]
            #idx_x_b = indices[:]
            idx_xs_a = indices[:]
            idx_xs_b = indices[:]
            if s[i] == 0:
                xs = y[:]
            else:
                if s[i] > 0:
                    idx_xs_a[i] = slice(0,-s[i])
                    idx_xs_b[i] = slice(0,s[i])
                    idx_x_a[i] = slice(s[i],None)
                    #idx_x_b[i] = slice(s[i]-1,None,-1) #Pytorch does not 
                    # support negative steps
                else:
                    idx_xs_a[i] = slice(-s[i],None)
                    idx_xs_b[i] = slice(s[i],None)
                    idx_x_a[i] = slice(0,s[i])
                    #idx_x_b[i] = slice(-1,s[i]-1,-1) #Pytorch does not 
                    # support negative steps
                
                xs[tuple(idx_xs_a)] = y[tuple(idx_x_a)]
                xs[tuple(idx_xs_b)] += reverse(y[tuple(idx_xs_b)],dim = i)
                #xs[tuple(idx_xs_b)] += y[tuple(idx_x_b)]
                y = xs[:]
        
    elif bc == 'zero':
        xs = th.zeros_like(x)        
        idx_x = indices[:]
        idx_xs = indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                if s[i] < 0: 
                    idx_x[i] = slice(0,x.shape[i]+s[i])
                    idx_xs[i] = slice(-s[i],x.shape[i])
                else: 
                    idx_x[i] = slice(s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]-s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]
        
    else:
        raise Exception("Unknown boundary conditions")
    
    return xs

def shift(x,s,bc='circular'):
    """ Shift operator that can treat different boundary conditions. It applies 
    to a tensor of arbitrary dimensions. 
    ----------
    Usage: xs = shift(x,(0,1,-3,3),'reflexive')
    ----------
    Parameters
    ----------
    x : tensor.
    s : tuple that matches the dimensions of x, with the corresponding shifts.
    bc: String with the prefered boundary conditions (bc='circular'|'reflexive'|'zero')
        (Default: 'circular')
    """
    if not isinstance(bc, str):
        raise Exception("bc must be of type string")
       
    if not reduce(lambda x,y : x and y, [isinstance(k,int) for k in s]):
        raise Exception("s must be a tuple of ints")
           
    if len(s) < x.dim():
       s = s + (0,) * (x.dim()-len(s))        
    elif len(s) > x.dim():
        print("The shift values will be truncated to match the " \
        +"dimensions of the input tensor. The trailing extra elements will" \
        +" be discarded.")
        s = s[0:x.dim()]
    
    if reduce(lambda x,y : x or y, [ math.fabs(s[i]) > x.shape[i] for i in range(x.dim())]):
        raise Exception("The shift steps should not exceed in absolute values"\
        +" the size of the corresponding dimensions.")

    # use a list sequence instead of a tuple since the latter is an 
    # immutable sequence and cannot be altered         
    indices = [slice(0,x.shape[0])]
    for i in range(1,x.dim()):
        indices.append(slice(0,x.shape[i]))
        
    if bc == 'circular':
        xs = x.clone() # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                m = x.shape[i]
                idx = indices[:]                
                idx[i] = list((np.arange(0,m)-s[i])%m)
                xs = xs[tuple(idx)]
    elif bc == 'reflexive':
        xs = x.clone() # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                idx = indices[:]
                if s[i] > 0: # right shift                    
                    idx[i] = list(range(s[i]-1,-1,-1)) + list(range(0,x.shape[i]-s[i]))
                else: # left shift
                    idx[i] = list(range(-s[i],x.shape[i])) + \
                    list(range(x.shape[i]-1,x.shape[i]+s[i]-1,-1))
                
                xs = xs[tuple(idx)]
    elif bc == 'zero':
        xs=th.zeros_like(x)        
        idx_x=indices[:]
        idx_xs=indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:       
                if s[i] > 0: # right shift
                    idx_x[i] = slice(0,x.shape[i]-s[i])
                    idx_xs[i] = slice(s[i],x.shape[i])
                else: # left shift
                    idx_x[i] = slice(-s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]+s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]
        
    else:
        raise Exception("Unknown boundary conditions")
    
    return xs

def shift_transpose(x,s,bc='circular'):
        
    r""" Transpose of the shift operator that can treat different boundary conditions. 
    It applies to a tensor of arbitrary dimensions. 
    ----------
    Usage: xs = shift_transpose(x,(0,1,-3,3),'reflexive')
    ----------
    Parameters
    ----------
    x : tensor.
    s : tuple that matches the dimensions of x, with the corresponding shifts.
    bc: String with the prefered boundary conditions (bc='circular'|'reflexive'|'zero')
        (Default: 'circular')
    """   
    
    if not isinstance(bc, str):
        raise Exception("bc must be of type string")
       
    if not reduce(lambda x,y : x and y, [isinstance(k,int) for k in s]):
        raise Exception("s must be a tuple of ints")
           
    if len(s) < x.dim():
       s = s + (0,)* (x.dim()-len(s))        
    elif len(s) > x.dim():
        print("The shift values will be truncated to match the " \
        +"dimensions of the input tensor. The trailing extra elements will" \
        +" be discarded.")
        s = s[0:x.dim()]
    
    if reduce(lambda x,y : x or y, [ math.fabs(s[i]) > x.shape[i] for i in range(x.dim())]):
        raise Exception("The shift steps should not exceed in absolute values"\
        +" the size of the corresponding dimensions.")
        
    # use a list sequence instead of a tuple since the latter is an 
    # immutable sequence and cannot be altered 
    indices=[slice(0,x.shape[0])]
    for i in range(1,x.dim()):
        indices.append(slice(0,x.shape[i]))
        
    if bc == 'circular':
        xs = x.clone() # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                m = x.shape[i]
                idx = indices[:]                
                idx[i] = list((np.arange(0,m)+s[i])%m)
                xs = xs[tuple(idx)]
    elif bc == 'reflexive':
        y=x.clone()
        for i in range(x.dim()):
            xs = th.zeros_like(x)
            idx_x_a = indices[:]
            #idx_x_b = indices[:]
            idx_xs_a = indices[:]
            idx_xs_b = indices[:]
            if s[i] == 0:
                xs = y.clone()
            else:
                if s[i] > 0:
                    idx_xs_a[i] = slice(0,-s[i])
                    idx_xs_b[i] = slice(0,s[i])
                    idx_x_a[i] = slice(s[i],None)
                    #idx_x_b[i] = slice(s[i]-1,None,-1) #Pytorch does not 
                    # support negative steps
                else:
                    idx_xs_a[i] = slice(-s[i],None)
                    idx_xs_b[i] = slice(s[i],None)
                    idx_x_a[i] = slice(0,s[i])
                    #idx_x_b[i] = slice(-1,s[i]-1,-1) #Pytorch does not 
                    # support negative steps
                
                xs[tuple(idx_xs_a)] = y[tuple(idx_x_a)]
                xs[tuple(idx_xs_b)] += reverse(y[tuple(idx_xs_b)],dim = i)
                #xs[tuple(idx_xs_b)] += y[tuple(idx_x_b)]
                y = xs.clone()
        
    elif bc == 'zero':
        xs = th.zeros_like(x)        
        idx_x = indices[:]
        idx_xs = indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                if s[i] < 0: 
                    idx_x[i] = slice(0,x.shape[i]+s[i])
                    idx_xs[i] = slice(-s[i],x.shape[i])
                else: 
                    idx_x[i] = slice(s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]-s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]
        
    else:
        raise Exception("Unknown boundary conditions")
    
    return xs

def compute_patch_overlap(shape,patchSize,stride=1,padding=0,GPU=False,dtype = 'f'):
    r""" Returns a tensor whose dimensions are equal to 'shape' and it 
    indicates how many patches extracted from the image (the patches are of 
    size patchSize and are extracted using a specified stride) each pixel of 
    the image contributes. 

    For example below is the array which indicates how many times each pixel
    at the particular location of an image of size 16 x 16 has been found in 
    any of the 49 4x4 patches that have been extracted using a stride=2.

    T = 
     1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
     1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
     1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1


     Based on this table the pixel at the location (3,2) has been used in 4
     different patches while the pixel at the location (15,4) has been used in
     2 different patches."""
     
    assert(isinstance(shape,tuple)), "shape is expected to be a tuple."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    if len(shape) < 4:
        shape = (1,)*(4-len(shape)) + shape
    elif len(shape) > 4:
        shape = shape[0:3]
    
    if len(patchSize) < 2:
        patchSize = patchSize * 2
    
    if dtype == 'f' : 
        dtype = th.FloatTensor
    elif dtype == 'd' :
        dtype = th.DoubleTensor
    else:
        raise Exception("Supported data types are 'f' (float) and 'd' (double).")
    
    
    shape_ = (shape[0]*shape[1],1,shape[2],shape[3])
    
    
    Pn = reduce(lambda x,y : x*y, patchSize[0:2])
    h = th.eye(Pn).type(dtype)
    h = h.view(Pn,1,patchSize[0],patchSize[1])
    
    x = th.ones(shape_).type(dtype)
    
    if th.cuda.is_available() and GPU:
        x = x.cuda()
        h = h.cuda()
    
    T = th.conv2d(x,h,stride = stride, padding = padding)
    T = th.conv_transpose2d(T,h,stride = stride, padding = padding)
    
    return T.view(shape)

def formatInput2Tuple(input,typeB,numel,strict = True):
    assert(isinstance(input,(tuple,typeB))),"input is expected to be of type " \
        "tuple or of type " + str(typeB)[8:-2] + " but instead an input of "\
        +"type "+str(type(input))+" was provided."
    
    if isinstance(input,typeB):
        input = (input,)*numel
    
    if strict :
        assert(len(input) == numel), "An input of size "+str(numel)+" is expected "\
            "but instead input = "+str(input)+ " was provided."
    else:
        if len(input) < numel:
            input = input + (input[-1],)*(numel-len(input))
        elif len(input) > numel:
            input = input[0:numel]
        
    return tuple(typeB(i) for i in input)

def getPad2RetainShape(kernel_size,dilation = 1):
    r"""Returns the necessary padding in the format [TOP BOTTOM LEFT RIGHT] 
    so that the spatial dimensions of the output will remain the same with 
    the spatial dimensions of the input.
    Note: This function assumes that the conv2d is computed using stride = 1."""
    
    kernel_size = formatInput2Tuple(kernel_size,int,2)
    dilation = formatInput2Tuple(dilation,int,2)
    
    kernel_size = ((kernel_size[0]-1)*dilation[0]+1,(kernel_size[1]-1)*dilation[1]+1)
    Kc = th.Tensor(kernel_size).add(1).div(2).floor()
    return (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
                                        int(Kc[1])-1,kernel_size[1]-int(Kc[1]))    
    

def getPadSize(shape,patchSize,stride):
    r"""Computes the necessary padding so that an integer
    number of overlapping patches (of size patchSize) can be extracted from 
    the image using a specified stride.

    `padsize`:: Specifies the amount of padding of an image  as 
    [TOP, BOTTOM, LEFT, RIGHT].
    """
    assert(isinstance(shape,tuple)), "shape is expected to be a tuple."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    if len(shape) < 4:
        shape = (1,)*(4-len(shape)) + shape
    elif len(shape) > 4:
        shape = shape[0:3]
    
    if len(patchSize) < 2:
        patchSize = patchSize * 2
    
    if isinstance(stride,int):
        stride = (stride,)*2
    assert(isinstance(stride,tuple)), "stride is expected to be of type int "\
    "or of type tuple."
    
    shape = np.array(shape[2:4])
    patchSize = np.array(patchSize[0:2])
    stride = np.array(stride[0:2])
    
    assert(np.any(stride > 0)), "negative stride is not accepted."
    patchDims = (shape - patchSize)/stride + 1
    assert(np.any(patchDims > 0)), "The specified size of the patch is "\
    "greater than the spatial dimensions of the input tensor."
    
    usePad = patchDims - np.floor(patchDims)
    
    if usePad[0]:
        padSizeTB = np.floor(patchDims[0])*stride[0] + patchSize[0] - shape[0]
        padSizeTB = (np.floor(padSizeTB/2),np.ceil(padSizeTB/2))
    else:
        padSizeTB = (0,0)
    
    if usePad[1]:
        padSizeLR = np.floor(patchDims[1])*stride[1] + patchSize[1] - shape[1]
        padSizeLR = (np.floor(padSizeLR/2),np.ceil(padSizeLR/2))
    else:
        padSizeLR = (0,0)
    
    return tuple(int(i) for i in padSizeTB) + tuple(int(i) for i in padSizeLR)

def im2col(input,block_size,stride=1,pad=0,dilation=1):
    r""" im2col extracts all the valid patches from the input which is a 2D, 3D 
    or 4D tensor of size [B] x [C] x H x W. The extracted patches are of size 
    patchSize and they are extracted with an overlap equal to stride. The 
    output is of size B x C*P x BS where P is the total number of elements
    in the patch, while BS is the total number of extracted patches.
    """    
    while input.dim() < 4:
        input = input.unsqueeze(0)
        
    #Im2Col(input, kernel_size, dilation, padding, stride)
    block_size = formatInput2Tuple(block_size,int,2)
    dilation = formatInput2Tuple(dilation,int,2)
    pad = formatInput2Tuple(pad,int,2)
    stride = formatInput2Tuple(stride,int,2)
    
    return th.nn.functional.Im2Col.apply(input,block_size,dilation,pad,stride)

def col2im(input,output_size,block_size,stride=1,pad=0,dilation=1):
    r""" col2im is the transpose operation of im2col.
    
    output_size : is the size of the original tensor from which the patches 
    where extracted.
    """    
    #Col2Im(input, output_size, kernel_size, dilation, padding, stride)
    assert(input.dim()==3),"The first input argument must be a 3D tensor."
    output_size = formatInput2Tuple(output_size,int,2)
    block_size = formatInput2Tuple(block_size,int,2)
    dilation = formatInput2Tuple(dilation,int,2)
    pad = formatInput2Tuple(pad,int,2)
    stride = formatInput2Tuple(stride,int,2)    
    
    return th.nn.functional.Col2Im.apply(input,output_size,block_size,dilation,pad,stride)

def im2patch(input,patchSize,stride=1) :
    r""" im2patch extracts all the valid patches from the input which is a 3D 
    or 4D tensor of size B x C x H x W. The extracted patches are of size 
    patchSize and they are extracted with an overlap equal to stride. The 
    output is of size B x C*P x PH x PW where P is the total number of elements
    in the patch, while PH and PW is the number of patches in the horizontal and
    vertical axes, respectively.
    """
    assert(input.dim() >= 3 and input.dim() < 5), "A 3D or 4D tensor is expected."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    
    if len(patchSize) < 2:
        patchSize  *=  2
    
    if input.dim() == 3:
        input = input.unsqueeze(0)
        
    Pn = reduce(lambda x,y : x*y, patchSize[0:2])
    h = th.eye(Pn).type(input.type())
    h = h.view(Pn,1,patchSize[0],patchSize[1])
    
    batch, Nc = input.shape[0:2] 
    
    if Nc != 1:
        input = input.view(batch*Nc,1,input.shape[2],input.shape[3])
    
    P = th.conv2d(input,h,stride = stride)
    
    if Nc != 1:
        P = P.view(batch,Nc*Pn,P.shape[2],P.shape[3])
    
    return P

def patch2im(input,shape,patchSize,stride=1) :
    r""" patch2im is the transpose operation of im2patch.
    
    shape : is the size of the original tensor from which the patches where 
    extracted.
    """
    assert(input.dim() == 4), "A 4D tensor is expected."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    
    if len(patchSize) < 2:
        patchSize  *=  2
    if len(shape) < 4:
        shape = (1,)*(4-len(shape)) + shape
    elif len(shape) > 4:
        shape = shape[0:3]

        
    Pn = reduce(lambda x,y : x*y, patchSize[0:2])
    batch = shape[0]
    Nc = math.floor(input.shape[1]/Pn);
    if Nc != 1:
        input = input.view(batch*Nc,input.shape[1]/Nc,input.shape[2],input.shape[3])
    
    h = th.eye(Pn).type(input.type())
    h = h.view(Pn,1,patchSize[0],patchSize[1])
    
    out = th.conv_transpose2d(input,h,stride = stride)
        
    if Nc != 1:
        out = out.view(batch,Nc*out.shape[1],out.shape[2],out.shape[3])
    
    if reduce(lambda x,y : x or y,[out.shape[i] < shape[i] for i in range(4)]):
        out = th.nn.functional.pad(out,(0,shape[3]-out.shape[3],0,shape[2]-out.shape[2]))
    
    return out

def im2patch_sinv(input,shape,patchSize,stride=1) :
    r""" im2patch_sinv is the pseudo inverse of im2patch.
    
    shape : is the size of the original tensor from which the patches where 
    extracted.
    """
    assert(input.dim() == 4), "A 4D tensor is expected."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    
    if len(patchSize) < 2:
        patchSize  *=  2
    if len(shape) < 4:
        shape = (1,)*(4-len(shape)) + shape
    elif len(shape) > 4:
        shape = shape[0:3]

        
    Pn = reduce(lambda x,y : x*y, patchSize[0:2])
    batch = shape[0]
    Nc = math.floor(input.shape[1]/Pn);
    if Nc != 1:
        input = input.view(batch*Nc,input.shape[1]/Nc,input.shape[2],input.shape[3])
    
    h = th.eye(Pn).type(input.type())
    h = h.view(Pn,1,patchSize[0],patchSize[1])
    
    out = th.conv_transpose2d(input,h,stride = stride)
        
    if Nc != 1:
        out = out.view(batch,Nc*out.shape[1],out.shape[2],out.shape[3])
    
    D = compute_patch_overlap(shape,patchSize,stride)
    D = D.type(input.type())
    out = out.div(D)
    
    if reduce(lambda x,y : x or y,[out.shape[i] < shape[i] for i in range(4)]):
        out = th.nn.functional.pad(out,(0,shape[3]-out.shape[3],0,shape[2]-out.shape[2]))
    
    return out
        
def odctdict(n,L,dtype = 'f',GPU = False):
    D = th.zeros(n,L)
    if dtype == 'f':
        D = D.float()
    else:
        D = D.double()
    
    D[:,0] = 1/math.sqrt(n)
    for k in range(1,L): 
        o = th.arange(0,n)*math.pi*k/L
        v = th.cos(o.float()); 
        v -= v.mean();
        D[:,k] = v.div(v.norm(p=2))
    
    if th.cuda.is_available() and GPU:
        D = D.cuda()
    
    return D

def odctndict(n,L,p = None, dtype = 'f', GPU = False):
    r"""  D = ODCTNDICT((N1 N2 ... Np),(L1 L2 ... Lp)) returns an overcomplete 
    DCT dictionary for p-dimensional signals of size N1xN2x...xNp. The number 
    of DCT atoms in the i-th dimension is Li, so the combined dictionary is of
    size (N1*N2*...*Np) x (L1*L2*...*Lp).

    D = ODCTNDICT([N1 N2 ... Np],L) specifies the total number of atoms in
    the dictionary instead of each of the Li's individually. The Li's in
    this case are selected so their relative sizes are roughly the same as
    the relative sizes of the Ni's. Note that the actual number of atoms in
    the dictionary may be larger than L, as rounding might be required for
    the computation of the Li's.

    D = ODCTNDICT(N,L,P) is shorthand for the call ODCTNDICT(N*ones(1,P),L),
    and returns the overcomplete DCT dictionary for P-dimensional signals of
    size NxNx...xN. L is the required size of the overcomplete dictionary,
    and is rounded up to the nearest integer with a whole P-th root.
    """
    assert(isinstance(n,int) or isinstance(n,tuple)), " n should be either of "\
    "type int or of type tuple."
    assert(isinstance(L,int) or isinstance(L,tuple)), " L should be either of "\
    "type int or of type tuple."
    assert(isinstance(p,int) or p is None), " p should be either of "\
    "type int or being omitted."
         
    n = np.asarray(n)
    L = np.asarray(L)

    if p is None:
        p = n.size

    if n.size == 1 :
        n = n*np.ones((1,p))
    if L.size == 1 :
        L = L*np.ones((1,))
        

    if L.size ==1 and p > 1 :
        N = np.prod(n)
        L = np.ceil((L*(np.power(n,p)/N)**(1/(p-1)))**(1/p))
    
    n = tuple(int(i) for i in n)
    L = tuple(int(i) for i in L)
    
    D = odctdict(n[0],L[0],dtype,GPU)
    for i in range(1,p):
        D = kron(D,odctdict(n[i],L[i],dtype,GPU))
    
    return D

def odct2dict(n,L,dtype = 'f', GPU = False):
    return odctndict(n,L,2,dtype,GPU)

def odct3dict(n,L,dtype = 'f', GPU = False):
    return odctndict(n,L,3,dtype,GPU)

def kron(x,y):
    r""" Kronecker tensor product.
    KRON(X,Y) is the Kronecker tensor product of X and Y.
    The result is a large matrix formed by taking all possible
    products between the elements of X and those of Y. For
    example, if X is 2 by 3, then KRON(X,Y) is
 
       [ X[0,0]*Y  X[0,1]*Y  X[0,2]*Y
         X[1,0]*Y  X[1,1]*Y  X[1,2]*Y ]
    """
    assert(x.dim() == 1 or x.dim() == 2), "x must be either a 1D or 2D tensor."
    assert(y.dim() == 1 or y.dim() == 2), "x must be either a 1D or 2D tensor."
    
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    x_size = x.shape
    y_size = y.shape
    
    x = x.t().contiguous().view(-1)
    y = y.t().contiguous().view(-1)
    
    z = y.ger(x)
    
    D = th.Tensor().type_as(x)
    for m in range(0,x_size[1]):
        d = th.Tensor().type_as(x)
        for k in range(x_size[0]*m,x_size[0]*(m+1)):
            d = th.cat((d,z[:,k].contiguous().view(y_size[1],y_size[0]).t()),dim=0)
        if m == 0:
            D = th.cat((D,d))
        else:
            D = th.cat((D,d),dim=1)
    
    return D

def sub2ind(shape,*args):
    r"""Linear index from multiple subscripts.
    SUB2IND is used to determine the equivalent single index
    corresponding to a given set of subscript values.
 
    IND = SUB2IND(shape,I,J) returns the linear index equivalent to the
    row and column subscripts in the arrays I and J for a matrix of
    size SIZ. 
 
    IND = SUB2IND(shape,I1,I2,...,IN) returns the linear index
    equivalent to the N subscripts in the arrays I1,I2,...,IN for an
    array of size SIZ.
 
    I1,I2,...,IN must have the same size, and IND will have the same size
    as I1,I2,...,IN. For a tensor A, if IND = SUB2IND(A.shape,I1,...,IN),
    then A.take(IND[k])=A(I1[k],...,IN[k]) for all k.
    
    The subscript arguments must be of type np.ndarrays.
    """
    for k in args:
        assert(isinstance(k,np.ndarray)),"All the subscript arguments are "\
        "expected to be of type 'ndarray'."
    
    assert(len(shape) == len(args)), "%d subscript arguments are expected."\
    %len(shape)
    
    s = args[0].shape
    assert(args[0].min() >= 0 and args[0].max() < shape[0]), "Invalid values for the "\
    "subscript arguments."
    for k in range(1,len(args)) :
        assert(s == args[k].shape),"The dimensions of all the subscript arguments "\
        "must match."
        assert(args[k].min() >= 0 and args[k].max() < shape[k]), "Invalid "\
        "values for the subscript arguments."
        
    p = th.LongTensor(np.hstack((np.cumprod(np.array(shape[1:])[-1::-1])[-1::-1],np.array(1))))
    
    idx = th.zeros(args[0].shape).long()
    for k in range(0,len(args)):
        idx += th.from_numpy(args[k]).long().mul(p[k])
        
    return idx

def meshgrid(*args):
    r""" Y,X,Z = meshgrid(np.arange(-3,4),np.arange(-2,3),np.arange(2,5))"""
    s_ind = list()
    siz =  len(args)
    for t in args:
        assert(isinstance(t,np.ndarray) and t.ndim == 1),"Input arguments must be 1D ndarrays."    
    
    for k in range(0,siz):
        s_ind.append(args[k])
        s_ind[k].shape = (1,)*k + (args[k].size,) + (1,)*(siz-1-k)
    
    for k in range(0,siz):
        for m in range(0,siz):
            if k != m :
                s_ind[k] = s_ind[k].repeat(args[m].size,axis = m)
    
    return s_ind    
    
def meshgrid_(shape) :
    
    s_ind = list()
    
    siz = len(shape)
    for k in range(0,siz):
        s_ind.append(np.arange(0,shape[k]))
        s_ind[k].shape = (1,)*k + (shape[k],) + (1,)*(siz-1-k)
    
    for k in range(0,siz):
        for m in range(0,siz):
            if k != m :
                s_ind[k] = s_ind[k].repeat(shape[m],axis = m)
    
    return s_ind

def TicTocGenerator():
    import time
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        #print("Elapsed time: {:f} seconds\n".format(tempTimeInterval))
        return tempTimeInterval

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)  


def im2Tensor(img,dtype = th.FloatTensor):
    assert(isinstance(img,np.ndarray) and img.ndim in (2,3,4)), "A numpy "\
    "nd array of dimensions 2, 3, or 4 is expected."
    
    if img.ndim == 2:
        return th.from_numpy(img).unsqueeze_(0).unsqueeze_(0).type(dtype)
    elif img.ndim == 3:
        return th.from_numpy(img.transpose(2,0,1)).unsqueeze_(0).type(dtype)
    else:
        return th.from_numpy(img.transpose((3,2,0,1))).type(dtype)    

def tensor2Im(img,dtype = np.float32):
    assert(isinstance(img,th.Tensor) and img.ndimension() == 4), "A 4D "\
    "torch.Tensor is expected."
    if img.size(0) == 1:
        img = img.squeeze(0)
        fshape = (1,2,0)
    else:
        fshape = (0,2,3,1)
    
    return img.numpy().transpose(fshape).astype(dtype)
    
def imnormalize(input):
    out = input - input.min()
    out = out/out.max()
    return out


def gen_imdb_BSDS500_fromList(\
        listPath = "/home/stamatis/Documents/Work/repos/datasets/BSDS500/BSDS_validation_list.txt",\
        imdbPath = "/home/stamatis/Documents/Work/repos/datasets/BSDS500/",\
        color = True, savePath = None, shape = (128,128),img_ext = '.jpg', \
        dtype = 'f', train = 0.8, test = 0.2, data = 'both'):


    def randomCropImg(img,output_shape):
        input_shape = img.shape[0:2]
        row_start = np.random.randint(0,input_shape[0]-output_shape[0]+1)
        col_start = np.random.randint(0,input_shape[1]-output_shape[1]+1)
        
        return img[row_start:row_start+output_shape[0]:1,col_start:col_start+output_shape[1]:1,...]    
    
    import os
    from matplotlib import image as Img
    import numpy as np
    
    # read all the images in the dataset
    if color:
        imdbPath =  os.path.join(imdbPath,'color')
    else:
        imdbPath = os.path.join(imdbPath,'gray')
    
    l = os.listdir(imdbPath)
    l = [os.path.join(imdbPath,f) for f in l if f.endswith(img_ext)]
    N = len(l) # number of images in the dataset
    
    # read all the images that should be excluded from the training set
    f = open(listPath,'r')
    test_samples = f.readlines(-1); f.close()
    test_samples = [os.path.join(imdbPath,k.replace('\n','')) for k in test_samples]
       
    # images in the dataset excluding the images defined in the list
    lx = [f for f in l if f not in test_samples]

    train = train/(train+test)
    test = test/(train+test)
  
    Ntrain = int(np.round(N*train)) # number of train images
    Ntest = N - Ntrain # number of test images
    
    assert(Ntest >= len(test_samples)),"The list contains more test samples\
    than the specified percentage of the total number of available images."
    
    l_test = list(np.random.choice(lx,Ntest-len(test_samples),replace = False))
    test_samples.extend(l_test)
    train_samples = [f for f in l if f not in test_samples]
    
    
    im_shape = Img.imread(train_samples[0]).shape
    if len(im_shape) == 2:
        H,W = im_shape
        Nchannels = 1
    else:
        H,W,Nchannels = im_shape
    
    assert(max(shape) <= min(H,W)),"The specified shape is incompatible with "\
    "the dimensions of the images in the dataset."
    
    imdb_train = np.zeros((shape)+(Nchannels,Ntrain),dtype = dtype)
    for i in range(Ntrain):
        img = np.array(Img.imread(train_samples[i]),dtype = dtype)
        if img.ndim == 2:
            img.shape = img.shape + (1,)
            
        imdb_train[...,i] = randomCropImg(img,shape)
    
    imdb_test = np.zeros((shape)+(Nchannels,Ntest),dtype = dtype)
    for i in range(Ntest):
        img = np.array(Img.imread(test_samples[i]),dtype=dtype)
        if img.ndim == 2:
            img.shape = img.shape + (1,)
            
        imdb_test[...,i] = randomCropImg(img,shape)        
    
    if savePath is not None:
        np.savez(savePath, train_set = imdb_train, test_set  = imdb_test)        
    
    if data == 'both':
        return imdb_train,imdb_test
    elif data == 'train':
        return imdb_train
    elif data == 'test':
        return imdb_test
    
def wmad_estimator(x,wname='db7',mode='symmetric',multichannel=False):
    r"""Accepts either a torch tensor or an ndarray and provides an estimate
    of the standard deviation of the noise degrading the input. It can operate
    on a batch of multichannel images. If the input is a torch Tensor we assume
    that the dimensions are B x C x H x W otherwise H x W x C x B, where H, W 
    are the spatial dimensions, C the image channels and B the number of images.
    It returns a torch tensor or an ndarray of size B x 1 (B x C if multichannel
    is set to True) with the respective estimated standard deviations."""
    from pywt import dwtn
    
    assert(isinstance(x,np.ndarray) or th.is_tensor(x)),"The first input "\
    +"argument must be either a ndarray or a tensor."
    
    assert(isinstance(wname,str)),"The second input argument must be a string "\
    +"indicating the wavelet basis to be used for the decomposition."

    tensor = False if type(x) is np.ndarray else True
    
    if tensor: 
        assert(x.dim() < 5),"Input is expected to be at most a 4D tensor."
        while x.dim() != 4:
            x = x.unsqueeze(0)
        cuda = x.is_cuda
        x = x.cpu().numpy().transpose((2,3,1,0))        
    else:
        assert(x.ndim < 5),"Input is expected to be at most a 4D ndarray."
        while x.ndim != 4:
            x.shape += (1,)
    
    
    c = dwtn(x,wname,mode=mode,axes = (0,1))['dd']
    prod = lambda z: reduce(lambda x,y: x*y,z)
    c = c.reshape((prod(c.shape[0:2]),)+c.shape[2:])
    
    if multichannel:
        sigma = np.median(np.abs(c)/.6745,axis = 0)
    else:
        sigma = np.median(np.abs(c)/.6745,axis = 0).mean(axis = 0, keepdims=True)
    
    sigma = sigma.transpose(1,0)
    if tensor:
        sigma = th.from_numpy(sigma).cuda() if cuda else th.from_numpy(sigma)
    
    return sigma 


def block_wmad_estimator(x,wname='db7',blckSize=(8,8),blckStride=(4,4),\
                         mode='symmetric',multichannel=False,upsample = False,\
                         upsample_mode = 'bilinear'):
    r"""Accepts a torch tensor and provides an estimate of the standard deviation 
    of the noise degrading blocks of the input. The overlap of the extracted 
    blocks is defined through blckStride. The function can be applied on a
    a batch of multichannel images. The tensor's dimensions are B x C x H x W, 
    where H, W are the spatial dimensions, C the image channels and B is the 
    number of images. It returns a torch tensor of size B x C x PH x PW 
    with the respective estimated standard deviations for all blocks P = PH x PW."""
    
    assert(th.is_tensor(x) and x.dim()==4),"The first input argument must be "\
    +"a 4D tensor."
    
    assert(isinstance(wname,str)),"The second input argument must be a string "\
    +"indicating the wavelet basis to be used for the decomposition."

    patchDims = np.floor((np.asarray(x.shape[2:])-np.asarray(blckSize))/\
                         np.asarray(blckStride)+1)
    patchDims = patchDims.astype(np.int64)

    from pywt import Wavelet
    # Retrieve the high-pass decomposition filter for the wavelet transform
    dec_hi = np.asarray(Wavelet(wname).filter_bank[1])
    
    kernel = th.from_numpy(dec_hi).type_as(x)
    while kernel.dim() < 4:
        kernel.unsqueeze_(0)
    
    kernel = reverse(kernel,dim=3)
    
    batch,channels,H,W = x.shape
    
    P = im2col(x,blckSize,blckStride).permute(2,0,1)
    numPatches = P.size(0)
    
    assert(numPatches == patchDims[0]*patchDims[1]),"Something wrong happened."
    
    P = P.contiguous().view(numPatches*batch*channels,1,blckSize[0],blckSize[1])
    
    padding = getPad2RetainShape((kernel.shape[-1],)*2,dilation = 1)
    
    P = pad2D(P,padding,mode)
    stdn_est = th.conv2d(P,kernel,stride=(1,2))
    kernel = kernel.squeeze(0).unsqueeze(3)
    stdn_est = th.conv2d(stdn_est,kernel,stride=(2,1))
    
    stdn_est = stdn_est.view(numPatches,batch,channels,-1).abs().div(.6745).median(dim=3)[0]
    
    if not multichannel:
        stdn_est = stdn_est.mean(dim=2,keepdim=True)
        stdn_est = stdn_est.expand(-1,-1,channels)
    
    stdn_est = stdn_est.permute(1,2,0).view(batch,channels,patchDims[0],patchDims[1]) 
    
    if upsample:
        return th.nn.functional.upsample(stdn_est,x.shape[2:],mode=upsample_mode)
    else:
        return stdn_est
    
def loadmat(filename):
    import scipy.io as spio
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
    
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    
    return _check_keys(data)


def imfilter2D_FrequencyDomain(input,kernel,padType="symmetric",mode="conv"):
    r"""If the input and the kernel are both multichannel tensors then each
    channel of the input is filtered by the corresponding channel of the 
    kernel.Otherwise, if kernel has a single channel each channel of the input
    is filtered by the same channel of the kernel."""
    from pydl.cOps import cmul

    assert(mode in ("conv","corr")), "Valid filtering modes are"\
    +" 'conv' and 'corr'."
    assert(padType in ("periodic","symmetric","zero","valid")), "Valid padType"\
    +" values are 'periodic'|'symmetric'|'zero'|'valid'."
    
    assert(input.dim() < 5),"The input must be at most a 4D tensor."
    
    while input.dim() < 3:
        input = input.unsqueeze(0)
    
    assert(kernel.dim() < 4),"The filtering kernel must be at most a 3D tensor."
    
    while kernel.dim() < 3:
        kernel = kernel.unsqueeze(0)
    
    assert(kernel.size(0) == 1 or kernel.size(0) == input.size(-3))," Invalid "\
    +"filtering kernel dimensions."
    
    if input.dim() == 4:
        kernel = kernel.unsqueeze(0)

    if mode != "conv":
        kernel = reverse(reverse(kernel,dim=-1),dim=-2)
    
    shape = tuple(kernel.shape)
    
    if padType == "symmetric" or padType == "zero":
        padding = getPad2RetainShape(shape[-2:],dilation = 1)
        input = pad2D(input,padding,padType)
    
    if padType == "valid":
        padding = getPad2RetainShape(shape[-2:],dilation = 1)
    
    if input.dim() == 4:
        kernel_pad = th.zeros(shape[0],shape[1],input.size(2),input.size(3)).type_as(kernel)        
        kernel_pad[...,0:shape[2],0:shape[3]] = kernel
        del kernel
    else:
        kernel_pad = th.zeros(shape[0],input.size(1),input.size(2)).type_as(kernel)        
        kernel_pad[...,0:shape[1],0:shape[2]] = kernel
        del kernel
        
    s = tuple(int(i) for i in -(np.asarray(shape[-2:])//2))
    if input.dim() == 4:
        s = (0,0) + s
    else:
        s = (0,) + s    
    kernel_pad = shift(kernel_pad,s,bc='circular')

    K = th.rfft(kernel_pad,2)   
        
    out = th.irfft(cmul(th.rfft(input,2),K),2,signal_sizes=input.shape[-2:])
    
    if padType != "periodic":
        out = crop2D(out,padding)
    
    return out

def imfilter_transpose2D_FrequencyDomain(input,kernel,padType="symmetric",mode="conv"):
    
    from pydl.cOps import cmul, conj
    
    assert(mode in ("conv","corr")), "Valid filtering modes are"\
    +" 'conv' and 'corr'."
    assert(padType in ("periodic","symmetric","zero","valid")), "Valid padType"\
    +" values are 'periodic'|'symmetric'|'zero'|'valid'."
    
    assert(input.dim() < 5),"The input must be at most a 4D tensor."
    
    while input.dim() < 3:
        input = input.unsqueeze(0)
    
    assert(kernel.dim() < 4),"The filtering kernel must be at most a 3D tensor."
    
    while kernel.dim() < 3:
        kernel = kernel.unsqueeze(0)
    
    assert(kernel.size(0) == 1 or kernel.size(0) == input.size(-3))," Invalid "\
    +"filtering kernel dimensions."
    
    if input.dim() == 4:
        kernel = kernel.unsqueeze(0)
    
    if mode != "conv":
        kernel = reverse(reverse(kernel,dim=-1),dim=-2)    
    
    shape = tuple(kernel.shape)
    
    if padType != "periodic":
        padding = getPad2RetainShape(shape[-2:],dilation = 1)
        input = pad2D(input,padding,"zero")
   
    if input.dim() == 4:
        kernel_pad = th.zeros(shape[0],shape[1],input.size(2),input.size(3)).type_as(kernel)        
        kernel_pad[...,0:shape[2],0:shape[3]] = kernel
        del kernel
    else:
        kernel_pad = th.zeros(shape[0],input.size(1),input.size(2)).type_as(kernel)        
        kernel_pad[...,0:shape[1],0:shape[2]] = kernel
        del kernel
    
    s = tuple(int(i) for i in -(np.asarray(shape[-2:])//2))
    if input.dim() == 4:
        s = (0,0) + s
    else:
        s = (0,) + s    
    kernel_pad = shift(kernel_pad,s,bc='circular')

    K = conj(th.rfft(kernel_pad,2))
        
    out = th.irfft(cmul(th.rfft(input,2),K),2,signal_sizes=input.shape[-2:])
    
    if padType == "symmetric" or padType == "zero":
        out = pad_transpose2D(out,padding,padType)
    
    return out

def imfilter2D_SpatialDomain(input,kernel,padType="symmetric",mode="conv"):
    r"""If the input and the kernel are both multichannel tensors then each
    channel of the input is filtered by the corresponding channel of the 
    kernel.Otherwise, if kernel has a single channel each channel of the input
    is filtered by the same channel of the kernel."""
    
    assert(mode in ("conv","corr")), "Valid filtering modes are"\
    +" 'conv' and 'corr'."
    assert(padType in ("periodic","symmetric","zero","valid")), "Valid padType"\
    +" values are 'periodic'|'symmetric'|'zero'|'valid'."    
   
    assert(input.dim() < 5),"The input must be at most a 4D tensor."
    
    while input.dim() <  4:
        input = input.unsqueeze(0)
    
    assert(kernel.dim() < 4),"The filtering kernel must be at most a 3D tensor."
    
    while kernel.dim() < 4:
        kernel = kernel.unsqueeze(0)
    
    channels = input.size(1)     
    assert(kernel.size(1) == 1 or kernel.size(1) == channels),"Invalid "\
    +"filtering kernel dimensions."
    
    if channels != 1:
        kernel = kernel.expand(1,channels,*kernel.shape[-2:])
    
    if mode == "conv":
        kernel = reverse(reverse(kernel,dim=-1),dim=-2)
    
    if padType == "valid":
        padding = 0
    else:
        padding = getPad2RetainShape(kernel.shape[-2:])
    
    input = pad2D(input,padding,padType)
    
    groups = 1    
    if kernel.size(1) != 1:
        kernel = kernel.view(channels,1,*kernel.shape[-2:])
        groups = channels
    
    out = th.conv2d(input,kernel,groups = groups)
        
    return out

def imfilter_transpose2D_SpatialDomain(input,kernel,padType="symmetric",mode="conv"):
    
    assert(mode in ("conv","corr")), "Valid filtering modes are"\
    +" 'conv' and 'corr'."
    assert(padType in ("periodic","symmetric","zero","valid")), "Valid padType"\
    +" values are 'periodic'|'symmetric'|'zero'|'valid'."
    
    assert(input.dim() < 5),"The input must be at most a 4D tensor."
    
    while input.dim() <  4:
        input = input.unsqueeze(0)
    
    assert(kernel.dim() < 4),"The filtering kernel must be at most a 3D tensor."
    
    while kernel.dim() < 4:
        kernel = kernel.unsqueeze(0)
    
    channels = input.size(1)     
    assert(kernel.size(1) == 1 or kernel.size(1) == channels),"Invalid "\
    +"filtering kernel dimensions."
    
    if channels != 1:
        kernel = kernel.expand(1,channels,*kernel.shape[-2:])
    
    if mode == "conv":
        kernel = reverse(reverse(kernel,dim=-1),dim=-2)
    
    if padType == "valid":
        padding = 0
    else:
        padding = getPad2RetainShape(kernel.shape[-2:])
        
    groups = 1    
    if kernel.size(1) != 1:
        kernel = kernel.view(channels,1,*kernel.shape[-2:])
        groups = channels
    
    out = th.conv_transpose2d(input,kernel,groups = groups)

    return pad_transpose2D(out,padding,padType)

def wiener_deconv(input,blurKernel,regKernel,alpha):
    r"""Multi Multichannel Deconvolution Wiener Filter for a batch of input
    images. (Filtering is taking place in the Frequency domain under the 
    assumption of periodic boundary conditions for the input image.)
    
    input :: tensor of size batch x channels x height x width.
    blurKernel :: tensor of size batch x channels x b_height x b_width
    regKernel :: tensor of size N x D x channels x r_height x r_width
    alpha :: tensor of size batch x N x channels.
    
    output : batch x N x channels x height x width"""
    
    from pydl.cOps import cmul, cabs, conj
    
    assert(input.dim() < 5),"The input must be at most a 4D tensor."    
    while input.dim() < 4:
        input = input.unsqueeze(0)

    batch = input.size(0)
    channels = input.size(1)

    assert(blurKernel.dim() < 5),"The blurring kernel must be at most a 4D tensor."
    while blurKernel.dim() < 4:
        blurKernel = blurKernel.unsqueeze(0)
    
    bshape = tuple(blurKernel.shape)
    assert(bshape[0] in (1,batch) and bshape[1] in (1,channels)),"Invalid blurring kernel dimensions."
            
    assert(regKernel.dim() < 6),"The regularization kernel must be at most a 5D tensor."    
    while regKernel.dim() < 5:
        regKernel = regKernel.unsqueeze(0)    
    
    rshape = tuple(regKernel.shape)
    assert(rshape[2] in (1,channels)),"Invalid regularization kernel dimensions."    
    
    N = rshape[0] # Number of wiener filters applied to each input image of size channels x height x width
    
    assert(alpha.shape == (batch,N,channels)),"Invalid dimensions for "\
    +"alpha parameter. The expected shape of the tensor is {} x {} x {}".format(batch,N,channels)
    
    K = th.zeros(bshape[0],bshape[1],input.size(2),input.size(3)).type_as(blurKernel)
    K[...,0:bshape[2],0:bshape[3]] = blurKernel
    del blurKernel

    bs = tuple(int(i) for i in -(np.asarray(bshape[-2:])//2))
    bs = (0,0) + bs
    K = shift(K,bs,bc='circular')    
    K = th.rfft(K,2) # batch x channels x height x width x 2

    G = th.zeros(rshape[0],rshape[1],rshape[2],input.size(2),input.size(3)).type_as(regKernel)
    G[...,0:rshape[3],0:rshape[4]] = regKernel
    del regKernel

    rs = tuple(int(i) for i in -(np.asarray(rshape[-2:])//2))
    rs = (0,0,0) + rs
    G = shift(G,rs,bc='circular')    
    G = th.rfft(G,2) # N x D x channels x height x width x 2     

    Y = cmul(conj(K),th.rfft(input,2)).unsqueeze(1) # batch x 1 x channels x height x width x 2
    
    K = cabs(K).pow(2).unsqueeze(-1) # batch x channels x height x width x 1
    G = cabs(G).pow(2).sum(dim=1).unsqueeze(0) # 1 x N x channels x height x width
    # batch x N x channels x height x width x 1
    G = G.mul(alpha.unsqueeze(-1).unsqueeze(-1)).unsqueeze(-1)  
    G += K.unsqueeze(1) # batch x N x channels x height x width x 1 
    del K
    return th.irfft(Y.div(G),2,signal_sizes=input.shape[-2:]) # batch x N x channels x height x width
        
 
def fftshift(x,dim = None):
    r"""FFTSHIFT Shift zero-frequency component to the center of the spectrum.
    For 1D complex tensors, FFTSHIFT(X) swaps the left and right halves of
    X.  For 2D complex tensors, FFTSHIFT(X) swaps the first and third
    quadrants and the second and fourth quadrants.  For N-D complex tensors
    FFTSHIFT(X) swaps "half-spaces" of X along each dimension. 
    A complex N-D tensor is a (N+1)-D tensor whose last dimension must be equal 
    to 2.

    FFTSHIFT(X,DIM) applies the FFTSHIFT operation along the 
    dimension DIM.

    FFTSHIFT is useful for visualizing the Fourier transform with
    the zero-frequency component in the middle of the spectrum."""
   
    assert(th.is_tensor(x) and x.size(-1) == 2),"input must be a complex tensor."
    
    if dim is not None:
        assert(isinstance(dim,int) and dim > -1 and dim < x.dim()),"Invalid specified dimension."
        s = (0,)*x.dim()
        s[dim] = x.size(dim)//2
    else:
        s = th.tensor(x.shape[0:-1],dtype = th.float32).div(2).floor()
        s = tuple(int(i) for i in s) + (0,)
    
    
    return shift(x,s,bc='circular')


def gaussian_filter(shape,std):
    
    shape = formatInput2Tuple(shape,float,2)
    siz = tuple((k-1)/2 for k in shape)
    
    [y,x] = meshgrid(np.arange(-siz[0],siz[0]+1),np.arange(-siz[1],siz[1]+1))
    arg = -(x**2+y**2)/(2*std**2)
    
    h = np.exp(arg)
    eps = np.spacing(1)
    h[h < eps*np.max(h)]  = 0

    if np.sum(h) != 0:
        h = h/np.sum(h)
    
    return h

def power_iteration(x0,A,numiter=20):
    r"""Compute the largest eigenvalue of the operator A."""

    for i in range(numiter):
        x = A(x0)
        x /= x.norm(p=2)
        x0 = x
    
    return A(x0).mul(x0).sum()/x0.norm(p=2).pow(2)

def getSubArrays(start,end,numSubArrays,length,dformat=lambda x:np.array(x)):
    r""" We want to split the array A=[start:end] in numSubArrays arrays of 
    equal  length so that all the elements in A are included in at least one of 
    the created sub-arrays.
    
    Returns the starting (s) and ending points (e) of each sub-array.
    
    The last argument 'dformat' is a lambda expression which can be defined to
    specify the type of the structure that will hold the data (default :
        lambda x:np.array(x))
    
    """
    
    assert(numSubArrays*length >= end), "The combination of the selected "\
    +"number of sub-arrays and length cannot cover completely all the elements "\
    +"between start and end."

    p = end-length
    s = np.int32(np.linspace(start,p,numSubArrays))
    e = s + length
    
    mask = {}
    for i in range(numSubArrays):
        mask[i] = dformat(range(s[i],e[i]))
    
    
    return mask


def psf2otf(psf,otfSize):
    r"""Transforms a given 2D psf (point spread function) to a 2D otf (optical 
    transfer) function of a specified size"""
    
    assert(psf.dim() == 2 and len(otfSize) >= 2),"Invalid input for psf and/or otfSize."
    assert(psf.size(0) <= otfSize[-2] and psf.size(1) <= otfSize[-1]),"The "\
    +"spatial support of the otf must be equal or larger to that of the psf."
    
    otf = th.zeros(otfSize).type_as(psf)
    otf[...,0:psf.size(0),0:psf.size(1)] = psf
    
    s = tuple(int(i) for i in -(np.asarray(psf.shape[0:])//2))
    s = (0,)*(len(otfSize)-2)+s
    otf = shift(otf,s,bc='circular')
    otf = th.rfft(otf,2)
    
    return otf

def edgetaper(input,psf):
    
    from pydl.cOps import cmul, conj
    
    assert(th.is_tensor(input) and th.is_tensor(psf)),"The inputs must be "\
    +"pytorch tensors."
    
    assert(input.dim() < 5), "The input is expected to be at most a 4D tensor."
    
    assert(psf.dim()==2),"Only 2D psfs are accepted."
    
    beta = {}
    
    if psf.size(0) != 1:
        psfProj = psf.sum(dim=1)
        z = th.zeros(input.size(-2)-1).type_as(psf)
        z[0:psf.size(0)] = psfProj
        z = th.rfft(z,1,onesided=True)
        z = th.irfft(cmul(z,conj(z)),1,onesided=True,signal_sizes=(input.size(-2)-1,))
        z = th.cat((z,z[0:1]),dim=0).div(z.max())
        beta['dim0'] = z.unsqueeze(-1)
    
    if psf.size(1) != 1:
        psfProj = psf.sum(dim=0)
        z = th.zeros(input.size(-1)-1).type_as(psf)
        z[0:psf.size(1)] = psfProj
        z = th.rfft(z,1,onesided=True)
        z = th.irfft(cmul(z,conj(z)),1,onesided=True,signal_sizes=(input.size(-1)-1,))
        z = th.cat((z,z[0:1]),dim=0).div(z.max())
        beta['dim1'] = z.unsqueeze(0)

    if len(beta.keys()) == 1:
        alpha = 1 - beta[list(beta.keys())[0]]
    else:
        alpha = (1-beta['dim0'])*(1-beta['dim1'])
    
    while alpha.dim() < input.dim():
        alpha = alpha.unsqueeze(0)
            
    otf = psf2otf(psf,input.shape)
    
    blurred_input = th.irfft(cmul(th.rfft(input,2),otf),2,signal_sizes = input.shape[-2:])
    
    output = alpha*input + (1-alpha)*blurred_input
    
    return output.clamp(input.min(),input.max()),alpha    

def imGrad(input,bc='reflexive'):
    r"""Computes the discrete gradient of an input batch of images of size 
    B x C x H x W. The output is of size B x 2*C x H x W where in the 2nd 
    dimension from 0:C the gradient with respect to the y-axis of each channel
    of the image is stored, while in C:2C the gradient with respect to the 
    x-axis is stored."""
    assert(th.is_tensor(input) and input.dim() == 4), "A 4D tensor is expected "\
    +"as input."
    
    return th.cat((shift(input,(0,0,-1,0),bc)-input,\
                       shift(input,(0,0,0,-1),bc)-input),dim = 1)

def imDivergence(input,bc='reflexive'):
    r"""Computes the discrete divergence ( adjoint of the gradient) of an input 
    batch of gradient images of size B x 2*C x H x W. The output is of size 
    B x C x H x W."""
    
    assert(th.is_tensor(input) and input.dim() == 4), "A 4D tensor is expected "\
    +"as input."
    
    assert(input.size(1)%2 == 0),"Invalid input dimensions: the second "\
    +"dimension must be even sized."
    idx = input.size(1)//2
    
    return shift_transpose(input[:,0:idx,...],(0,0,-1,0),bc)-input[:,0:idx,...]\
                + shift_transpose(input[:,idx:,...],(0,0,0,-1),bc)-input[:,idx:,...]\

def init_dct(tensor):
    r"""Initializes the input tensor with weights from the dct basis or dictionary."""
    assert(tensor.ndimension() == 4),"A 4D tensor is expected."
    output_features,input_channels,H,W = tensor.shape
    
    if H*W*input_channels == output_features+1:
        tensor.data.copy_(gen_dct3_kernel(tensor.shape[1:]).type_as(tensor)[1:,...])
    else:
        if input_channels == 1:
            weights = odctndict((H,W),output_features+1)
        else:
            weights = odctndict((H,W,input_channels),output_features+1)
        weights = weights[:,1:output_features+1].type_as(tensor).view(H,W,input_channels,output_features)
        weights = weights.permute(3,2,0,1)
        tensor.data.copy_(weights)

def init_dctMultiWiener(tensor):
    r"""Initializes the input tensor with weights from the dct basis or dictionary."""
    assert(tensor.dim() in (4,5)),"A 4D or 5D tensor is expected."
    if tensor.dim() == 4:
        output_features,input_channels,H,W = tensor.shape
    else:
        numFilters,output_features,input_channels,H,W = tensor.shape
    
    if H*W == output_features+1:
        weights = gen_dct2_kernel((H,W)).type_as(tensor)[1:,...]
        if tensor.dim() == 4:
            weights = weights.repeat(1,input_channels,1,1)
        else:
            weights = weights.unsqueeze_(0).repeat(numFilters,1,input_channels,1,1)
    else:
        if input_channels == 1:
            weights = odctndict((H,W),output_features+1)
        else:
            weights = odctndict((H,W,input_channels),output_features+1)
        weights = weights[:,1:output_features+1].type_as(tensor).view(H,W,input_channels,output_features)
        weights = weights.permute(3,2,0,1)        
        if tensor.dim() == 5:
            weights = weights.unsqueeze_(0).repeat(numFilters,1,1,1,1)

    tensor.data.copy_(weights)        
        
    
def init_rbf_lut(centers,sigma,start,end,step):
    r"""Computes necessary data for the Look-up table of rbf computation."""
    data_samples = th.range(start,end,step).type_as(centers)
    data_samples = data_samples.unsqueeze(1)
    data_samples = (data_samples - centers)/sigma
    return data_samples

def init_msra(tensor):
    r"""Initializes the input tensor with weights according to He initialization."""
    output_channels,input_channels,H,W = tensor.shape
    tensor.data.copy_(th.randn_like(tensor).\
        mul(th.sqrt(th.Tensor([2])).type_as(tensor).div(H*W*input_channels)))

def init_convWeights(tensor,init_type = 'dct'):
    if init_type == 'dct':
        init_dct(tensor)
    elif init_type == 'msra':
        init_msra(tensor)
    else: 
        raise NotImplementedError
    
    