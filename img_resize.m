clc
clear 
close all

% image resizing
srcFiles = dir('D:\MangoLeaf_Dataset\Anthracnose\*.jpg'); 
for i = 1 : length(srcFiles) 
    filename = strcat('D:\MangoLeaf_Dataset\Anthracnose\',srcFiles(i).name);
    im = imread(filename); k=imresize(im,[320,320]); 
    newfilename=strcat('D:\MangoLeaf_Dataset\Anthracnose\',srcFiles(i).name); 
    imwrite(k,newfilename,'jpg');
end