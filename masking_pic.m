clc
clear 
close all

%% load all file names
img_direct='D:\Gray_MangoLeaf Dataset\Anthracnose';
filenames=dir(fullfile(img_direct,'*.jpg'));
total_images=numel(filenames);
file={filenames.name};

%% read images
for i=1:total_images
    f{i}=fullfile(img_direct,file{i});
    Img{i}=imread(f{i});
end

% krisch compass mask values
mask_value={[-3 -3 5;-3 0 5;-3 -3 5],[-3 5 5;-3 0 5;-3 -3 -3]...
    [5 5 5;-3 0 -3;-3 -3 -3],[5 5 -3;5 0 -3;-3 -3 -3]...
    [5 -3 -3;5 0 -3;5 -3 -3],[-3 -3 -3;5 0 -3;5 5 -3]...
    [-3 -3 -3;-3 0 -3;5 5 5],[-3 -3 -3;-3 0 5;-3 5 5]};

[row,col]=size(Img{1});
result=zeros(row-2,col-2);

% save the masking values in a directory
folder='D:\Gray_Image_Values_MangoLeaf\Anthracnose';
for i=1:total_images
    for j=1:8
        baseFileName=sprintf('Anth%d%d.xlsx',i,j);
        % calling 'edge' function to calculate directional values
        result=edge(Img{i},mask_value{j}); 
        fullFilename=fullfile(folder,baseFileName);
        xlswrite(fullFilename,result);
    end
end

%% load 8 directional values in cell  
for i=1:total_images
    for j=1:8
    baseFileName=sprintf('Anth%d%d.xlsx',i,j);
    fullFilename=fullfile(folder,baseFileName);
    cell{i,j}=xlsread(fullFilename); 
    end
end



