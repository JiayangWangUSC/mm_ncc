function Jmax = aux_dyadic_max_scales(N)
%AUX_DYADIC_MAX_SCALES: Computes the maximum number of dyadic scales.
%
%   Jmax = aux_dyadic_max_scales(N) computes the maximum number of dyadic
%   scales for a given size of signal N.
%   For example, if N=[100,136], Jmax=[2,3].
% 	
%   Input:
%   - N : 2D vector containing the signal size.
% 	
%   Output:
%   - Jmax : 2D vector containing the maximum number of dyadic scales along
%   the dimension X (Jmax(1)) and Y (Jmax(2)).
% 
%   Authors: Florian Luisier
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/

i = 1;
while(~mod(N(1),2^i))
    i = i+1;
end;
Jmax(1) = i-1;
i = 1;
while(~mod(N(2),2^i))
    i = i+1;
end;
Jmax(2) = i-1;