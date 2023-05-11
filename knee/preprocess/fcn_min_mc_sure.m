function xhat = fcn_min_mc_sure(Y,ICpred,ISpred,band,R)
%FCN_MIN_MC_SURE: Removes additive white Gaussian noise using the multi-
%   channel interscale SURE-LET principle.
%
%   output = fcn_min_mc_sure(Y,ICpred,ISpred,band,R) denoises the
%   wavelet coefficients of a given subband 'Y(:,:,band)' using its cor-
%   responding interchannel predictor 'ICpred' and interscale predictor 
%   'ISpred'. 
% 	
%   Input:
%   - Y      : wavelet coefficients of the noisy signal.
%   - ICpred : interchannel predictor.
%   - ISpred : interscale predictor.
%   - band   : index of the band to be denoised.
%   - R      : interchannel noise covariance matrix.
% 	
%   Output:
%   - xhat   : denoised wavelet coefficients associated with 'Y(:,:,band)'.
%
%   See also OWT_MC_SURELET_denoise, fcn_mc_denoise.
% 
%   Author: Florian Luisier
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/
%
%   References:
%   [1] F. Luisier, T. Blu, "SURE-LET Multichannel Image Denoising:
%   Interscale Orthonormal Wavelet Thresholding," IEEE Transactions on
%   Image Processing, vol. 17, no. 4, pp. 482-492, April 2008. 
%   [2] F. Luisier, T. Blu, M. Unser, "A New SURE Approach to Image
%   Denoising: Interscale Orthonormal Wavelet Thresholding," IEEE
%   Transactions on Image Processing, vol. 16, no. 3, pp. 593-606,
%   March 2007. 

[nx,ny,nz] = size(Y);
N          = nx*ny;
Y          = reshape(Y,N,nz);
theta      = zeros(N,4*nz);
div        = zeros(4*nz,1);
rho        = R(:,band);
weights    = pinv(6*sqrt(nz)*R);
dICpred    = -Y*(weights*rho).*ICpred;
k          = 0;
for z = 1:nz
    k = k+1;
    theta(:,k) = Y(:,z);
    div(k)     = rho(z);
    k = k+1;
    theta(:,k) = Y(:,z).*ICpred(:);
    div(k)     = mean(rho(z).*ICpred(:)+Y(:,z).*dICpred(:));    
    k = k+1;
    theta(:,k) = Y(:,z).*ISpred(:);
    div(k)     = rho(z)*mean(ISpred(:));
    k = k+1;
    theta(:,k) = Y(:,z).*ISpred(:).*ICpred;
    div(k)     = mean(ISpred(:).*(rho(z).*ICpred(:)+Y(:,z).*dICpred(:)));
end
K = k;
M = theta'*theta/N;

% SURE minimization:
%-------------------
Y = repmat(Y(:,band),1,K);
C = mean(Y.*theta)'-div;
% C = max(0,C); % E{x'F(y)} should be positive

% Set of SURE-optimized LET parameters
%-------------------------------------
A = pinv(M'*M+1e-4*eye(K))*M'*C;

% Estimate of the noise-free wavelet coefficients
%------------------------------------------------
xhat = reshape(theta*A,nx,ny);