function sigma = aux_noise_estim(y,filtersize,blocksize)
%AUX_NOISE_ESTIM: Estimates the standard deviation of the additive white
%   Gaussian noise using a robust eigenfilter procedure.
%	
%   Input:
%   - y                    : noisy input image.
%   - (OPTIONAL) filtersize: support of the 2D filter for the eigenfilter
%                            design.
%   - (OPTIONAL) blocksize : size of the neighborood for local estimation
%                            of the AWGN standard deviation.
% 	
%   Output:
%   - sigma: estimate of the AWGN standard deviation.
%
%   Author: Florian Luisier
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/
%
%   References:
%   [1] F. Luisier, "The SURE-LET Approach to Image Denoising," Swiss
%   Federal Institute of Technology Lausanne, EPFL Thesis no. 4566 (2010),
%   232 p., January 8, 2010.

if(~exist('filtersize','var'))
    filtersize = [3,3];
end
if(~exist('blocksize','var'))
    blocksize = round(0.1*size(y));
end

% Eigenfilter computation & application
%--------------------------------------
Y          = im2col(y,filtersize,'sliding')';
A          = cov(Y);
opts.disp  = 0;
[h,lambda] = eigs(A,1,'SM',opts); %#ok<NASGU>
Hy         = reshape(Y*h,size(y)-filtersize+1);

% Local AWGN std computation
%---------------------------
By         = sqrt(pi/2)*conv2(ones(blocksize(1),1)/blocksize(1),...
                              ones(blocksize(2),1)/blocksize(2),...
                              abs(Hy),'valid');

% Histogram computation
%----------------------
bins       = 100;
edges      = linspace(min(By(:)),max(By(:))+1,bins);
N          = histc(By(:),edges);

% Mode of the smoothed histogram
%-------------------------------
[aux,Ns]   = spaps(edges,N,1e-1*median(N).^2);
[M,m]      = max(Ns);
sigma      = edges(m);