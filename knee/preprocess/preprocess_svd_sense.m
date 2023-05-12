%%
addpath(genpath('/project/jhaldar_118/jiayangw/mm_ncc/SPIRiT_v0.3'));
datapath = '/project/jhaldar_118/jiayangw/dataset/knee_copy/train/';
%addpath(genpath('/home/wjy/Project/mm_ncc/SPIRiT_v0.3'));
%datapath = '/home/wjy/Project/fastmri_dataset/knee_copy/';
dirname = dir(datapath);
N1 = 320; N2 = 368; Nc = 15; Ns = 15;

%%
%newdatapath = '/project/jhaldar_118/jiayangw/dataset/brain_clean/train/';
for dir_num = 3:length(dirname)
    h5create([datapath,dirname(dir_num).name],'/image_svd',[N2,N1,1,Ns],'Datatype','single');
end

%%
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x,1)*size(x,2));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x,1)*size(x,2)); 

%%
for dir_num = 3 : length(dirname)
%% slice selection, undersampling and whitening 
kData = h5read([datapath,dirname(dir_num).name],'/kspace');
kspace = complex(kData.r,kData.i)*2e5;
kspace = permute(kspace,[4,2,1,3]);

image_svd = zeros(Ns,N1,N2,1);

for s = 1:Ns
    % undersample
    im = ifft2c(reshape(kspace(s+15,:,:,:),2*N1,N2,Nc));
    im = im(161:480,:,:);
    
    % whiten
    patch = [reshape(im(:,1:10,:),[],Nc);reshape(im(:,end-10:end,:),[],Nc)];
    cov = patch'*patch/size(patch,1);
    cov_inv = inv(cov);
    [~,S,V] = svd(cov_inv);
    W = V*sqrt(S);
    W_back = inv(sqrt(S))*V';
    im = reshape(im,[],Nc)*W;
    
    % svd compression
    cov = im'*im/size(im,1);
    [~,S,V] = svd(cov);
    V = V(:,1:6);
    im = reshape(im * V * V'* W_back,N1,N2,Nc);
    %image_svd(s,:,:,1) = sqrt(sum(abs(im).^2,3));
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
    image_svd(s,:,:,:) = abs(sum(reshape(im,N1,N2,Nc).*conj(maps),3));
end
image_svd = permute(image_svd,[3,2,4,1]);


%% new dataset
image_svd = single(image_svd);
h5write([datapath,dirname(dir_num).name],'/image_svd',image_svd);
end
