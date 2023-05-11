function J = aux_num_of_iters(N)
%AUX_NUM_OF_ITERS: Computes the most suitable number of iterations to be
%   performed in the SURE-LET algorithm.
%
%   J = aux_num_of_iters(N) computes the most suitable number of iterations
%   to be performed in the SURE-LET algorithm for a given size of
%   signal N.
%   The idea is to prevent denoising in subbands where the number of
%   samples is below 256 to ensure a reliable statistical analysis.
% 	
%   Input:
%   - N : 2D vector containing the signal size.
% 	
%   Output:
%   - J : 2D vector containing the most suitable number of dyadic scales
%   along the dimension X (J(1)) and Y (J(2)).
%
%   See also OWT_MULTI_SURELET_denoise, aux_dyadic_max_scales.
% 
%   Author: Florian Luisier
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/

J = zeros(size(N));

% Maximum Number of Possible Iterations
%--------------------------------------
Jmax = aux_dyadic_max_scales(N);
if(max(Jmax)==0)
    return;
end

% Compute the Most Suitable Number of Iterations
%-----------------------------------------------
if(Jmax(1)==0)
    J(2) = min(max(floor(log2(N(2)/256)),0),Jmax(2));
elseif(Jmax(2)==0)
    J(1) = min(max(floor(log2(N(1)/256)),0),Jmax(1));
else
    J(1) = min(max(floor(log2(N(1)/16)),0),Jmax(1));
    J(2) = min(max(floor(log2(N(2)/16)),0),Jmax(2));
end
if(max(J)==0)
    return;
end
res = prod(N)/2^sum(J);
while(res>=512)
    if(~mod(N(1)/2^J(1),2))
        J(1) = J(1)+1;
        res = res/2;
    elseif(~mod(N(2)/2^J(2),2))
        J(2) = J(2)+1;
        res = res/2;
    else
        res = 0;
    end
end