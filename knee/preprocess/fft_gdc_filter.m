function W = fft_gdc_filter(G,wtype)
%FFT_GDC_FILTER: Provides the 1D frequency response of the Group Delay
%   Compensation (GDC) filter associated to the wavelet filter G.
%
%   W = fft_gdc_filter(G,wtype) computes the 1D frequency response of the
%   Group Delay Compensation (GDC) filter associated to the wavelet filter
%   G, as presented in [1].
% 	
%   Input:
%   - G : frequency response of the analysis wavelet filter.
%   - wtype : wavelet filter (see the Matlab function 'wfilters'
%   to find all the available filters). For wavelet filters like the Symlets,
%   the Coiflets or the discrete Meyer wavelet, the shortest GDC filter is
%   chosen.
% 	
%   Output:
%   - W : frequency response of the GDC filter associated to the wavelet
%   filter 'G'.
%
%   See also fft_wavedec, wfilters.
% 
%   Authors: Florian Luisier and Thierry Blu, March 2007
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/
%
%   References:
%   [1] F. Luisier, T. Blu, M. Unser, "A New SURE Approach to Image
%   Denoising: Interscale Orthonormal Wavelet Thresholding," 
%   IEEE Transactions on Image Processing, vol. 16, no. 3, pp. 593-606, 
%   March 2007.

M        = length(G);
nu       = (0:(M/2-1))/(M/2);
z        = exp(2*1i*pi*nu);
LO_D     = wfilters(wtype,'l');
[aux,n0] = max(abs(fliplr(LO_D)));
symmetry = 0;
epsilon  = 1;
if(strcmp(wtype(1:3),'sym'))
    N = str2double(wtype(4:end));  
    if(N==1)
        symmetry = 0;
    else
        symmetry = 1;
    end
elseif(strcmp(wtype(1:3),'coi'))
    symmetry = 1;
elseif(strcmp(wtype(1:3),'dme'))
    symmetry = 1;
end
if(symmetry)
    W = z.^(-n0).*(z-1);
else
    W = conj(G(1:M/2)).*conj(G(M/2+1:M)).*(1+epsilon*z.^(-1));
end
% Filter Normalization
%---------------------
W = W/sqrt(mean(abs(W(:)).^2));