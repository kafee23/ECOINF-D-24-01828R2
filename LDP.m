clc
n=total_images;
direct='D:\Gray_Image_Values_MangoLeaf\Anthracnose\Anth_LDP';
[row,col]=size(cell{1,1});
result_1=zeros(row,col);

% calculating LDP values
for k=1:n
    for i=1:row
        for j=1:col
            temp=[cell{k,1}(i,j) cell{k,2}(i,j) cell{k,3}(i,j) cell{k,4}(i,j) cell{k,5}(i,j) cell{k,6}(i,j) cell{k,7}(i,j) cell{k,8}(i,j)];
            [out,idx] = sort(temp,'descend');
            result_1(i,j)=calculation(idx(1),idx(2),idx(3)); 
        end
    end
    baseFileName=sprintf('Anth_LDP%d.xlsx',k);
    fullFilename=fullfile(direct,baseFileName);
    xlswrite(fullFilename,result_1);

end


  
          