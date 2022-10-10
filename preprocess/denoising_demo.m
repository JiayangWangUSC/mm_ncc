%DENOISING_DEMO: Denoising demonstration based on the multichannel SURE-LET
%   principle applied to interscale orthonormal wavelet thresholding.
%
%   See also OWT_MC_SURELET_denoise.
%
%   Author: Florian Luisier
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/
%
%   Reference:
%   [1] F. Luisier, T. Blu, "SURE-LET Multichannel Image Denoising:
%   Interscale Orthonormal Wavelet Thresholding," IEEE Transactions on
%   Image Processing, vol. 17, no. 4, pp. 482-492, April 2008.
%   [2] F. Luisier, "The SURE-LET Approach to Image Denoising," Swiss
%   Federal Institute of Technology Lausanne, EPFL Thesis no. 4566 (2010),
%   232 p., January 8, 2010.

clear all;
close all;
addpath(genpath('./Images/'));

% Some parameters
%----------------
wtype   = 'sym8';       % Orthonormal Wavelet filter
DISPLAY = 'on';         % If 'on', the results are displayed

% Load the original noise-free image
%-----------------------------------
filename = 'parrots.tif';
original = aux_stackread(filename);

[nx,ny,C] = size(original);
sigma     = 20*ones(C,1);   % Set to 0 to input a different noise level
                            % inside each channel
                            
if(max(sigma)==0)
    % Noise standard deviation in the various channels
    %-------------------------------------------------
    sigma = zeros(C,1);
    for c=1:C
        sigma(c) = str2double(input(...
        ['Noise Standard Deviation in Channel ' num2str(c) ': '],'s'));
    end
    fprintf('\n');
end
   
% Create input noisy image
%-------------------------
%RandStream.setDefaultStream(RandStream('mt19937ar','seed',0));
noise = randn(nx,ny,C);
noise = noise/std(noise(:));
input = noise;
for c=1:C
    input(:,:,c) = original(:,:,c)+sigma(c)*noise(:,:,c);
end

% Noise covariance matrix
%------------------------
B = reshape(input-original,nx*ny,C);
R = cov(B);

% Denoise
%--------
start  = clock;
output = OWT_MC_SURELET_denoise(input,wtype,R);
time   = etime(clock,start);       

% PSNR computation
%-----------------
MSE_0  = mean((input(:)-original(:)).^2);
MSE_D  = mean((output(:)-original(:)).^2);
PSNR_0 = 10*log10(255^2/MSE_0);
PSNR_D = 10*log10(255^2/MSE_D);

% Display results
%----------------
fprintf(['\nInput PSNR   : ' num2str(PSNR_0,'%.2f') '[dB]']);
fprintf(['\nOutput PSNR  : ' num2str(PSNR_D,'%.2f') '[dB]']);
fprintf(['\nElapsed time : ' num2str(time,'%.2f') '[s]\n\n']);

% Plot results
%-------------
if(strcmp(DISPLAY,'on'))
    h = figure('Units','normalized','Position',[0 0.4 1 0.5]);
    set(h,'name','OWT MULTICHANNEL SURE-LET');
    if(C==1 || C==3)
        subplot(1,3,1);imagesc(uint8(original),[0 255]);
        axis image;colormap gray(256);axis off;
        title('Original','fontsize',16,'fontweight','bold');drawnow;
        subplot(1,3,2);imagesc(uint8(input),[0 255]);
        axis image;colormap gray(256);axis off;
        title(['Noisy: PSNR = ' num2str(PSNR_0,'%.2f') ' dB'],...
            'fontsize',16,'fontweight','bold');drawnow;
        subplot(1,3,3);imagesc(uint8(output),[0 255]);
        axis image;colormap gray(256);axis off;
        title(['Denoised: PSNR = ' num2str(PSNR_D,'%.2f') ' dB'],...
            'fontsize',16,'fontweight','bold');drawnow;
    else
        subplot(1,3,1);imagesc(uint8(original(:,:,1)),[0 255]);
        axis image;colormap gray(256);axis off;
        title('Original','fontsize',16,'fontweight','bold');drawnow;
        subplot(1,3,2);imagesc(uint8(input(:,:,1)),[0 255]);
        axis image;colormap gray(256);axis off;
        title(['Noisy: PSNR = ' num2str(PSNR_0,'%.2f') ' dB'],...
            'fontsize',16,'fontweight','bold');drawnow;
        subplot(1,3,3);imagesc(uint8(output(:,:,1)),[0 255]);
        axis image;colormap gray(256);axis off;
        title(['Denoised: PSNR = ' num2str(PSNR_D,'%.2f') ' dB'],...
            'fontsize',16,'fontweight','bold');drawnow; 
    end
end