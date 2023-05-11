%% sense maps
addpath(genpath('SPIRiT_v0.3'))
%datapath = '/project/jhaldar_118/jiayangw/dataset/brain_copy/train/';
datapath = '/home/wjy/Project/fastmri_dataset/brain_copy/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16; Ns = 8;

%%
%newdatapath = '/project/jhaldar_118/jiayangw/dataset/brain_clean/train/';
for dir_num = 3:length(dirname)
    h5create([datapath,dirname(dir_num).name],'/sense_maps',[N2,N1,2*Nc,Ns],'Datatype','single');
end

%%
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x,1)*size(x,2));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x,1)*size(x,2)); 

%%
for dir_num = 1:length(dirname)
%% slice selection, undersampling and whitening 
kData = h5read([datapath,dirname(dir_num).name],'/kspace');
kspace = complex(kData.r,kData.i)*2e5;
kspace = permute(kspace,[4,2,1,3]);

sense_maps = zeros(Ns,N1,N2,Nc);

for s = 1:Ns
    % undersample
    im = ifft2c(reshape(kspace(s,:,:,:),2*N1,N2,Nc));
    im = im(192:575,:,:);
    kdata = fft2c(im);
    
    % sense reconstruction
    [sx,sy,Nc] = size(kdata);
    ncalib = 24; % use 24 calibration lines to compute compression
    ksize = [6,6]; % kernel size
    eigThresh_1 = 0.1;
    eigThresh_2 = 0.95;
    calib = crop(kdata,[ncalib,ncalib,Nc]);
    
    [k,S] = dat2Kernel(calib,ksize);
    idx = max(find(S >= S(1)*eigThresh_1));
    [M,W] = kernelEig(k(:,:,:,1:idx),[sx,sy]);
    maps = M(:,:,:,end).*repmat(W(:,:,end)>eigThresh_2,[1,1,Nc]);
    sense_maps(s,:,:,:) = maps;
end
sense_maps = permute(sense_maps,[3,2,4,1]);


%% new dataset
csm = zeros(N2,N1,2*Nc,Ns);
csm(:,:,1:Nc,:) = real(sense_maps);
csm(:,:,Nc+1:2*Nc,:) = imag(sense_maps);
csm = single(csm);
h5write([datapath,dirname(dir_num).name],'/sense_maps',csm);
end
