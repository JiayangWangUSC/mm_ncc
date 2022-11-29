clear;
close all;
clc;

addpath(genpath('/home/haldar/001.Matlab/MatlabPath'));
load('3thumant2.mat');

figure;
imagesc3d(abs(fftshift(ifft2(kData))));
im = fftshift(ifft2(ifftshift(kData)));

im = im(257+[-128:127],:,:); % remove oversampled A/D

noise_only = im(50:70,1:60,:);
figure;
imagesc3d(abs(noise_only));
%%
var_ch = sum(sum(abs(noise_only).^2,1),2)/prod(size(noise_only(:,:,1)));

kData = fftshift(fft2(ifftshift(im)))/sqrt(prod(size(im(:,:,1)))); % Using normalization so that the FFT is unitary and preserves variance without scaling
figure;
imagesc3d(abs(kData)./sqrt(var_ch)); % using broadcasting
colorbar;
caxis([0,10]);
colormap([0,0,0;0.2,0.2,0.2;jet(8)]);

rgb = ind2rgb(round(imagesc3d(abs(kData)./sqrt(var_ch))/10*10), [0,0,0;0.2,0.2,0.2;jet(8)]);
im = imagesc3d(abs(kData)/1e-6);
im(im>1)=1;
figure;
imagesc(rgb);
figure;
imagesc(im);colormap(gray);

imwrite(im,'channels.png');
imwrite(rgb,'colorcoded.png');
figure;
colorbar;
colormap([0,0,0;0.2,0.2,0.2;jet(8)]);
caxis([0,10]);
saveas(gca,'colorbar.pdf'); % Just need to crop this to get the colorbar