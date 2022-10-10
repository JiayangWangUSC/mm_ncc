function y = aux_gaussian_smoothing(x,sigma)
%AUX_GAUSSIAN_SMOOTHING: Applies a normalized 2D Gaussian kernel on each 
%   z-slice of a 3D signal.
%
%   y = aux_gaussian_smoothing(x,sigma) applies a normalized 2D Gaussian
%   kernel with standard deviation 'sigma' on each z-slice of the 3D signal
%   'x'.
% 	
%   Input:
%   - x     : input 3D signal.
%   - sigma : standard deviation of the normalized 2D Gaussian kernel.
% 	
%   Output:
%   - y : smoothed 3D signal of the same size as 'x'.
% 
%   Authors: Florian Luisier
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/

nz     = size(x,3);
M      = ceil(3*sigma);
t      = -M:M;
kernel = 1/(sqrt(2*pi)*sigma)*exp(-t.^2/(2*sigma^2));
y      = x;
for z=1:nz
    y(:,:,z) = conv2(kernel,kernel,...
                     padarray(x(:,:,z),[M,M],'circular'),'valid');
end