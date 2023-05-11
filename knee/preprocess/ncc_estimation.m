%% effective NCC noise parameters estimation
datapath = '/project/jhaldar_118/jiayangw/dataset/knee_copy/train/';
%datapath = '/home/wjy/Project/fastmri_dataset/knee_copy/';
dirname = dir(datapath);
N1 = 320; N2 = 368; Nc = 15; Ns = 15;

%%
for dir_num = 3:length(dirname)
    h5create([datapath,dirname(dir_num).name],'/ncc_effect',[N2,N1,2,Ns],'Datatype','single');
end

%%
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x,1)*size(x,2));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x,1)*size(x,2)); 
vect = @(x) x(:);

%% non-central chi negative likelihood loss
F = @(x,y,L,sigma) (L-1)*log(x) + log(sigma/2) - mean(L*log(y)) + mean(x^2+y.^2)/sigma ...
                 - mean(log(besseli(L-1,2*x*y/sigma,1))+2*x*y/sigma);

%%
for dir_num=3:length(dirname)
    % load data
    ncc_effect = zeros(N1,N2,2,Ns);
    kData = h5read([datapath,dirname(dir_num).name],'/kspace');
    kspace = complex(kData.r,kData.i)*2e5;
    kspace = permute(kspace,[4,2,1,3]);
    
    for s = 1:Ns
        kdata = reshape(kspace(s+15,:,:,:),2*N1,N2,Nc);
        im = ifft2c(kdata);
        im = im(161:480,:,:);
        patch = [reshape(im(:,1:10,:),[],Nc);reshape(im(:,end-10:end,:),[],Nc)].';
        cov = patch*patch'/size(patch,2);
        
        Im = sqrt(sum(abs(im).^2,3));
        noise = [vect(Im(:,1:10));vect(Im(:,end-10:end))];
        noise = double(noise);
        
        func = @(x) F(eps,noise,x(1),x(2));
        x0 = double([Nc/2 ; trace(cov)/(Nc/2)]);
        options = optimoptions('fminunc', 'Algorithm','quasi-newton','SpecifyObjectiveGradient',false);
        x = fminunc(func,x0,options);
        
        ncc_effect(:,:,1,s) = x(1);
        ncc_effect(:,:,2,s) = x(2);
    end
    
    ncc_effect = permute(ncc_effect,[2,1,3,4]);
    h5write([datapath,dirname(dir_num).name],'/ncc_effect',single(ncc_effect));

end




