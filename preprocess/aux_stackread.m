function [stack,info] = aux_stackread(filename)
%AUX_STACKREAD: Read a stack of images and convert it to a double 3D
%   matrix.
% 	
%   [stack,info] = aux_stackread(filename) reads the stack named 'filename'
%   and returns a 3D double matrix 'stack' and the stack info 'info'.
%
%   Input:
%   - filename: string containing the name of the stack.
% 	
%   Output:
%   - stack: the extracted double 3D matrix.
%   - info : informations about the stack.
% 
%   Authors: Florian Luisier, December 2007
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/

info = imfinfo(filename);
nz   = size(info,1);
nx   = info(1).Height;
ny   = info(1).Width;
type = info(1).ColorType;
if(strcmp(type,'truecolor'))
    C = 3;
else
    C = 1;
end
stack = zeros(nx,ny,C*nz);
for z=1:nz
    stack(:,:,(z-1)*C+1:z*C) = double(imread(filename,z));
end