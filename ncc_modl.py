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

def data_transform(kspace, ncc_effect, sense_maps):
    # Transform the kspace to tensor format
    ncc_effect = transforms.to_tensor(ncc_effect)
    kspace = transforms.to_tensor(kspace)
    kspace = torch.cat((kspace[torch.arange(nc),:,:].unsqueeze(-1),kspace[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    #kspace_clean = transforms.to_tensor(kspace_clean)
    #kspace_clean = torch.cat((kspace_clean[torch.arange(nc),:,:].unsqueeze(-1),kspace_clean[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    sense_maps = transforms.to_tensor(sense_maps)
    sense_maps = torch.cat((sense_maps[torch.arange(nc),:,:].unsqueeze(-1),sense_maps[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)

    return kspace, ncc_effect, sense_maps

train_data = SliceDataset(
    #root=pathlib.Path('/home/wjy/Project/fastmri_dataset/brain_copy/'),
    root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_copy/train/'),
    transform=data_transform,
    challenge='multicoil'
)

def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

# %% MoDL loader
from modl_model import *
layers = 5
iters = 10
recon_model = MoDL(
    n_layers = layers,
    k_iters = iters
)
recon_model = torch.load("/project/jhaldar_118/jiayangw/mm_ncc/model/modl_mse_acc6")
# %% training settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size)
recon_model.to(device)
recon_optimizer = optim.Adam(recon_model.parameters(),lr=3e-3)
L2Loss = torch.nn.MSELoss()
L1Loss = torch.nn.L1Loss()

# %% sampling mask
mask = torch.zeros(ny,dtype=torch.int8)
mask[torch.arange(66)*6] = 1
mask[torch.arange(186,210)] =1
mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(nc,nx,1,2)

# %%
max_epochs = 100
#print("acc3")
for epoch in range(max_epochs):
    print("epoch:",epoch+1)
    batch_count = 0    
    for kspace, ncc_effect, sense_maps in train_dataloader:
        batch_count = batch_count + 1
        Mask = mask.unsqueeze(0).repeat(kspace.size(0),1,1,1,1)

        image_zf = fastmri.complex_mul(fastmri.complex_conj(sense_maps),fastmri.ifft2c(torch.mul(Mask,kspace))) 
        image_zf = torch.permute(torch.sum(image_zf, dim=1),(0,3,1,2)).to(device)
        maps = torch.complex(sense_maps[:,:,:,:,0],sense_maps[:,:,:,:,1]).to(device)
        recon = recon_model(image_zf, maps, Mask[:,0,:,:,0].squeeze().to(device)).to(device)
        recon = fastmri.complex_abs(torch.permute(recon,(0,2,3,1)))

        with torch.no_grad():
            gt = toIm(kspace)
            x = 2*recon.cpu()*gt/(ncc_effect[:,1,:,:])
            y = gt*(ss.ive(ncc_effect[:,0,:,:],x)/ss.ive(ncc_effect[:,0,:,:]-1,x))

        loss = torch.sum(torch.square(recon.to(device)-y.to(device))/ncc_effect[:,1,:,:].to(device))


        if batch_count%100 == 0:
            print("batch:",batch_count,"train loss:",loss.item())
        
        loss.backward()
        recon_optimizer.step()
        recon_optimizer.zero_grad()

    
        torch.save(recon_model,"/project/jhaldar_118/jiayangw/mm_ncc/model/modl_ncc_acc6")

# %%
