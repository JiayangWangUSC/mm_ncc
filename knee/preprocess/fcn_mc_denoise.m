function output = fcn_mc_denoise(input,parent,R)
%FCN_MC_DENOISE: Removes additive white Gaussian noise using the
%   multichannel interscale SURE-LET principle.
%
%   output = fcn_mc_denoise(input,parent,R,central) denoises the wavelet
%   coefficients of a given multichannel subband 'input' using its
%   corresponding interscale prediction 'parent'. 
% 	
%   Input:
%   - input  : wavelet coefficients of the noisy signal.
%   - parent : wavelet coefficients of the interscale prediction (parent)
%   associated to 'input'.
%   - R      : interchannel noise covariance matrix.
% 	
%   Output:
%   - output : denoised wavelet coefficients.
%
%   See also OWT_MC_SURELET_denoise, fcn_min_mc_sure.
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

[nx,ny,nz] = size(input);
output     = zeros(nx,ny,nz);
weights    = pinv(6*sqrt(nz)*R);

% Building the discriminative interchannel predictor:
%----------------------------------------------------
Y          = reshape(input,nx*ny,nz);
WeightedY2 = sum((Y*weights).*Y,2);
ICpred     = exp(-WeightedY2(:)/2);

% Building the discriminative interscale predictor:
%--------------------------------------------------
Yp          = reshape(parent,nx*ny,nz);
WeightedYp2 = sum((Yp*weights).*Yp,2);
ISpred      = exp(-WeightedYp2(:)/2);

for z = 1:nz
    output(:,:,z) = fcn_min_mc_sure(input,ICpred,ISpred,z,R);
end