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
import scipy.special as ss

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

# %% varnet loader
from varnet import *
cascades = 8
chans = 16
recon_model = VarNet(
    num_cascades = cascades,
    sens_chans = 16,
    sens_pools = 4,
    chans = chans,
    pools = 4,
    mask_center= True
)

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
mask[torch.arange(66)*6] = 1
mask[torch.arange(186,210)] =1
mask = mask.bool().unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(nc,nx,1,2)

sigma =1
# %%
max_epochs = 50
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    batch_count = 0    
    for train_batch in train_dataloader:
        batch_count = batch_count + 1
        Mask = mask.unsqueeze(0).repeat(train_batch.size(0),1,1,1,1).to(device)
    
        kspace_input = torch.mul(Mask,train_batch.to(device)).to(device)   
        recon = recon_model(kspace_input, Mask, 24).to(device)
        recon = fastmri.rss(fastmri.complex_abs(recon),dim=1)

        with torch.no_grad():
            gt = toIm(train_batch)
            x = recon.cpu()*gt/(sigma*sigma/2)
            y = gt*(ss.ive(nc,x)/ss.ive(nc-1,x))

        loss = L2Loss(recon.to(device),y.to(device))

        if batch_count%100 == 0:
            print("batch:",batch_count,"train loss:",loss.item())
        
        loss.backward()
        recon_optimizer.step()
        recon_optimizer.zero_grad()

    #if (epoch + 1)%20 == 0:
    torch.save(recon_model,"/project/jhaldar_118/jiayangw/mm_ncc/model/varnet_ncc_cascades"+str(cascades)+"_channels"+str(chans))


# %%
