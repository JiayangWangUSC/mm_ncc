function output = OWT_MC_SURELET_denoise(input,wtype,R)
%OWT_MC_SURELET_DENOISE: Removes additive white Gaussian noise using the
%   multichannel interscale SURE-LET principle applied in an orthonormal
%   wavelet transform (OWT).
% 	
%   output = OWT_MC_SURELET_denoise(input,wtype,R) performs a multichan-
%   nel interscale orthonormal wavelet thresholding based on the
%   algorithm described in [1].
%
%   Input:
%   - input : noisy signal of size [nx,ny].
%   - (OPTIONAL) wtype : orthonormal wavelet filter (see the Matlab
%   function 'wfilters' to find all the available orthonormal filters).
%   Default is 'sym8'.
%   - (OPTIONAL) R : noise interchannel covariance matrix. If not provided,
%   'R' is assumed to be diagonal, and its diagonal elements are estimated
%   using the procedure described in [2].
% 	
%   Output:
%   - output : denoised signal of the same size as 'input'.
%
%   See also fcn_mc_denoise.
% 
%   Author: Florian Luisier
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/
%
%   References:
%   [1] F. Luisier, T. Blu, "SURE-LET Multichannel Image Denoising:
%   Interscale Orthonormal Wavelet Thresholding," IEEE Transactions on
%   Image Processing, vol. 17, no. 4, pp. 482-492, April 2008. 
%   [2] F. Luisier, "The SURE-LET Approach to Image Denoising," Swiss
%   Federal Institute of Technology Lausanne, EPFL Thesis no. 4566 (2010),
%   232 p., January 8, 2010.

[nx,ny,nz] = size(input);
output     = zeros(nx,ny,nz);

if(~exist('wtype','var'))
    wtype = 'sym8';
end

% Compute the Most Suitable Number of Iterations
%-----------------------------------------------
J = aux_num_of_iters([nx,ny]);
if(max(J)==0)
    disp(['The size of the signal is too small to perform a reliable '...
    'statistical denoising.']);
    return;
end

% Orthonormal Wavelet Transform
%------------------------------
WT  = zeros(nx,ny,nz);
WTp = zeros(nx,ny,nz);
for z=1:nz
    WT(:,:,z)  = fft_wavedec(input(:,:,z),wtype,J);
    % Construction of the Interscale Predictor
    %-----------------------------------------
    WTp(:,:,z) = fft_wavedec(input(:,:,z),wtype,J,1);
end
WTden = WT;

% Estimation of the Noise Standard Deviation (if not provided) 
%-------------------------------------------------------------
if(~exist('R','var'))
    R = zeros(nz,nz);
    for z=1:nz
        % Eigenfilter-based Estimator
        %----------------------------
        sigma  = aux_noise_estim(input(:,:,z));        
        R(z,z) = sigma^2;        
        fprintf(['Estimated Noise Standard Deviation in Channel '...
                  num2str(z) ': ' num2str(sigma,'%.2f') '\n']);
    end
end

% Denoising Part
%---------------
for i=1:min(J)
    N1 = nx/2^i;
    N2 = ny/2^i;
    %%%%%%
    % HL %
    %%%%%%
    Y  = WT(1:N1,N2+1:2*N2,:);
    Yp = WTp(1:N1,N2+1:2*N2,:);
    Yp = aux_gaussian_smoothing(abs(Yp),1);
    WTden(1:N1,N2+1:2*N2,:) = fcn_mc_denoise(Y,Yp,R);
    %%%%%%
    % LH %
    %%%%%%
    Y  = WT(N1+1:2*N1,1:N2,:);
    Yp = WTp(N1+1:2*N1,1:N2,:);
    Yp = aux_gaussian_smoothing(abs(Yp),1);
    WTden(N1+1:2*N1,1:N2,:) = fcn_mc_denoise(Y,Yp,R);    
    %%%%%%
    % HH %
    %%%%%%
    Y  = WT(N1+1:2*N1,N2+1:2*N2,:);
    Yp = WTp(N1+1:2*N1,N2+1:2*N2,:);
    Yp = aux_gaussian_smoothing(abs(Yp),1);
    WTden(N1+1:2*N1,N2+1:2*N2,:) = fcn_mc_denoise(Y,Yp,R);
end

% Inverse Orthonormal Wavelet Transform
%--------------------------------------
for z=1:nz
    output(:,:,z) = fft_waverec(WTden(:,:,z),wtype,J);
end