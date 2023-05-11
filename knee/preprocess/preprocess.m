%%
datapath = '/project/jhaldar_118/jiayangw/dataset/knee_copy/train/';
%datapath = '/home/wjy/Project/fastmri_dataset/knee_copy/';
dirname = dir(datapath);
N1 = 320; N2 = 368; Nc = 15; Ns = 15; %brain

%%
for dir_num = 3:length(dirname)
    h5create([datapath,dirname(dir_num).name],'/kspace_central',[N2,N1,2*Nc,Ns],'Datatype','single');
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

kspace_new = zeros(Ns,N1,N2,Nc);
for s = 1:Ns
    im = ifft2c(reshape(kspace(s,:,:,:),2*N1,N2,Nc));
    im = im(161:480,:,:);
    kspace_new(s,:,:,:) = fft2c(im);
end
kspace_new = permute(kspace_new,[3,2,4,1]);
%% new dataset
kdata = zeros(N2,N1,2*Nc,Ns);
kdata(:,:,1:Nc,:) = real(kspace_new);
kdata(:,:,Nc+1:2*Nc,:) = imag(kspace_new);
kdata = single(kdata);
h5write([datapath,dirname(dir_num).name],'/kspace_central',kdata);
end
