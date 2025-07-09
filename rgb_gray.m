clc
clear
close all

img_direct='D:\MangoLeafBD Dataset\Anthracnose';
filenames=dir(fullfile(img_direct,'*.jpg'));
total_images=numel(filenames);
file={filenames.name};

% gray image conversion
gimg_direct='D:\Gray_MangoLeaf Dataset\Anthracnose';
for i=1:total_images
    f{i}=fullfile(img_direct,file{i});
    Img{i}=imread(f{i});
    gray_image=rgb2gray(Img{i});
    basefileName=sprintf('Anth%d',i);
    fullFileName=fullfile(gimg_direct, basefileName);
    imwrite(gray_image,fullfile(gimg_direct,[basefileName '.jpg']));
end
