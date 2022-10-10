# %%
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet
import pathlib
import numpy as np
import torch.optim as optim
from fastmri.data import  mri_data
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from torch.utils.data import Dataset
import h5py
import math


# %% data loader
from my_data import *

nc = 16
nx = 384
ny = 396

def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    kspace = torch.cat((kspace[torch.arange(nc),:,:].unsqueeze(-1),kspace[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    return kspace

train_data = SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/miniset_brain_clean/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_clean/train/'),
    transform=data_transform,
    challenge='multicoil'
)

def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

# %% unet loader

recon_model = Unet(
  in_chans = 32,
  out_chans = 32,
  chans = 256,
  num_pool_layers = 4,
  drop_prob = 0.0
)
recon_model = torch.load("/project/jhaldar_118/jiayangw/refnoise/model/imnet_mse")
#print(sum(p.numel() for p in recon_model.parameters() if p.requires_grad))
# %% training settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size)
recon_model.to(device)
recon_optimizer = optim.Adam(recon_model.parameters(),lr=3e-4)
L2Loss = torch.nn.MSELoss()
L1Loss = torch.nn.L1Loss()
# %% sampling mask
mask = torch.zeros(ny)
mask[torch.arange(132)*3] = 1
mask[torch.arange(186,210)] =1
mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(1,nc,nx,1,2)

# %%
max_epochs = 200
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    batch_count = 0    
    for train_batch in train_dataloader:
        batch_count = batch_count + 1
        Mask = mask.repeat(train_batch.size(0),1,1,1,1).to(device)
    
        torch.manual_seed(batch_count)
    
        noise = math.sqrt(0.5)*torch.randn_like(train_batch)
        kspace = (train_batch + noise).to(device)
        gt = toIm(kspace)
    
        image = fastmri.ifft2c(torch.mul(Mask.to(device),kspace.to(device))).to(device)   
        image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1).to(device) 
        image_output = recon_model(image_input).to(device)
        recon = torch.cat((image_output[:,torch.arange(nc),:,:].unsqueeze(4),image_output[:,torch.arange(nc,2*nc),:,:].unsqueeze(4)),4).to(device)
        recon = fastmri.rss(fastmri.complex_abs(recon),dim=1)

        loss = L2Loss(recon.to(device),gt.to(device))
    
        if batch_count%100 == 0:
            print("batch:",batch_count,"train loss:",loss.item())
        
        loss.backward()
        recon_optimizer.step()
        recon_optimizer.zero_grad()
    if (epoch + 1)%20 == 0:
        torch.save(recon_model,"/project/jhaldar_118/jiayangw/refnoise/model/imnet_mse_epoch"+str(epoch+1))
