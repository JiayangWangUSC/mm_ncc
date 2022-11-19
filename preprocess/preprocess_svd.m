%%
%datapath = '/project/jhaldar_118/jiayangw/dataset/brain_copy/train/';
addpath('../')
datapath = '/home/wjy/Project/fastmri_dataset/brain_copy/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16; Ns = 8;

%%
%newdatapath = '/project/jhaldar_118/jiayangw/dataset/brain_clean/train/';
%for dir_num = 3:length(dirname)
%    h5create([datapath,dirname(dir_num).name],'/image_svd',[N2,N1,1,Ns],'Datatype','single');
%end

%%
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x,1)*size(x,2));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x,1)*size(x,2)); 

%%
for dir_num = 3:length(dirname)
%% slice selection, undersampling and whitening 
kData = h5read([datapath,dirname(dir_num).name],'/kspace');
kspace = complex(kData.r,kData.i)*2e5;
kspace = permute(kspace,[4,2,1,3]);

image_svd = zeros(Ns,N1,N2,1);

for s = 1:Ns
    im = ifft2c(reshape(kspace(s,:,:,:),2*N1,N2,Nc));
    im = im(192:575,:,:);
    im = reshape(im,[],Nc);
    cov = im'*im/size(im,1);
    [U,S,V] = svd(cov);
    V = V(:,1:6);
    im = reshape(im * V*V',N1,N2,Nc);
    image_svd(s,:,:,1) = sqrt(sum(abs(im).^2,3));
end
image_svd = permute(image_svd,[3,2,4,1]);


%% new dataset
image_svd = single(image_svd);
h5write([datapath,dirname(dir_num).name],'/image_svd',image_svd);
end
