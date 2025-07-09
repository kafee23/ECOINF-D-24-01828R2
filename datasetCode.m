clc
clear
close all

% load excel file
img_diect1='D:\Gray_Image_Values_MangoLeaf\Anthracnose\Anth_LDP';
img_diect2='D:\Gray_Image_Values_MangoLeaf\Healthy\Heal_LDP';
n=1000;
%%
for i=1:n
    if i<=500
        baseFileName=sprintf('Anth_LDP%d.xlsx',i);
        fullFilename=fullfile(img_diect1,baseFileName);
        file{i,1}=xlsread(fullFilename);
    else
        baseFileName=sprintf('Heal_LDP%d.xlsx',(i-500));
        fullFilename=fullfile(img_diect2,baseFileName);
        file{i,1}=xlsread(fullFilename);
    end
end           
[row,col]=size(file);

% split the matrix in 3 by 3 submatrix
[row1,col1]=size(file{1,1});
nr=floor(row1/3);
nc=floor(row1/3);

%% mean based feature extraction
for k=1:n
    matA=file{k,1};
    for i=1:nr
        for j=1:nc
            subMat{i,j}=matA(((i-1)*3+1):3*i,((j-1)*3+1):3*j);
        end
    end 
    matB=zeros(3,3);
    for x=1:nr
        for y=1:nc
            matB=matB+subMat{x,y};
        end
    end
    matB=matB';
    matB=matB(:);
    matB=matB';
    featureMat_Anth3_LDP(k,:)=matB/(nr*nc);
end



