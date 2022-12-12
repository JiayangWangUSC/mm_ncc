# %%
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet
from varnet import *
import pathlib
import numpy as np
import torch.optim as optim
from fastmri.data import  mri_data
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import math

# %% data loader
from my_data import *

nc = 16
nx = 384
ny = 396

def data_transform(kspace, ncc_effect, image_svd):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    kspace = torch.cat((kspace[torch.arange(nc),:,:].unsqueeze(-1),kspace[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    image_svd = transforms.to_tensor(image_svd)
    return kspace, image_svd

test_data = SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_copy/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_clean/train/'),
    transform=data_transform,
    challenge='multicoil'
)

def KtoIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

def MtoIm(im):
    Im = fastmri.rss(fastmri.complex_abs(im),dim=1)
    return Im

# %% sampling mask
mask = torch.zeros(ny)
mask[torch.arange(132)*3] = 1
mask[torch.arange(186,210)] =1
mask = mask.bool().unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(nc,nx,1,2)

# %% define loss
L1Loss = torch.nn.L1Loss()
L2Loss = torch.nn.MSELoss()

import scipy.special as ss

def NccLoss(x1,x2,sigma,nc):
    x = x1*x2/(sigma*sigma/2)
    y = torch.sum(torch.square(x1)/(sigma*sigma)-(torch.log(ss.ive(nc-1,x))+x)+(nc-1)*torch.log(x1))
    return y/torch.sum(torch.ones_like(x))


# %% imnet loader
imnet = torch.load('/home/wjy/Project/mm_ncc_model/imnet_mse',map_location=torch.device('cpu'))

# %%
with torch.no_grad():
    kspace, image_svd = test_data[0]
    kspace = kspace.squeeze()
    noise = math.sqrt(0.5)*torch.randn_like(kspace)
    kspace_noise = kspace + noise

    gt = KtoIm(kspace)
    gt_noise = KtoIm(kspace_noise)

    Mask = mask.unsqueeze(0)
    kspace_undersample = torch.mul(kspace_noise,Mask)

    image = fastmri.ifft2c(kspace_undersample)
    image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
    image_output = imnet(image_input)
    recon = torch.cat((image_output[:,torch.arange(nc),:,:].unsqueeze(4),image_output[:,torch.arange(nc,2*nc),:,:].unsqueeze(4)),4)

    recon = MtoIm(recon)

# %% varnet loader
epoch = 100
#sigma = 1
cascades = 12
chans = 16
varnet = torch.load("/home/wjy/Project/mm_ncc_model/varnet_mae_acc3_cascades"+str(cascades)+"_channels"+str(chans),map_location = 'cpu')
#varnet = torch.load("/home/wjy/Project/refnoise_model/varnet_mse_acc4_cascades"+str(cascades)+"_channels"+str(chans)+"_epoch160",map_location = 'cpu')

# %%
with torch.no_grad():
    kspace, image_svd = test_data[0]
    kspace = kspace.unsqueeze(0)

    gt = KtoIm(kspace)

    Mask = mask.unsqueeze(0)
    kspace_undersample = torch.mul(kspace,Mask)
    
    recon_M = varnet(kspace_undersample, Mask, 24)

    recon = MtoIm(recon_M)

# %%
sp = torch.ge(image_svd,0.03*torch.max(image_svd))
print(L2Loss(recon,image_svd))
print(L2Loss(torch.mul(recon,sp),torch.mul(image_svd,sp)))
print(L1Loss(recon,image_svd))
print(L1Loss(torch.mul(recon,sp),torch.mul(image_svd,sp)))
#print(NccLoss(recon,gt,sigma,nc)-NccLoss(gt,gt,sigma,nc))

# %%
up = 220
bottom = 270
left = 140
right = 190
patch = image_svd
patch = patch[:,torch.arange(up,bottom),:]
patch = patch[:,:,torch.arange(left,right)]
patch = F.interpolate(patch.unsqueeze(0),size=[256,256],mode='nearest')
save_image(patch.squeeze()/50,'/home/wjy/Project/mm_ncc_result/sense_patch6_slice1.png')

# %%
