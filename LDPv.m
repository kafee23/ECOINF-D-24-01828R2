clc
n=total_images;
folder='D:\Gray_Image_Values_MangoLeaf\Anthracnose\Anth_LDPv';
[row,col]=size(cell{1,1});
result_2=zeros(row,col);

% calculating LDPv values
for k=1:n
    for i=1:row
        for j=1:col
            temp=[cell{k,1}(i,j) cell{k,2}(i,j) cell{k,3}(i,j) cell{k,4}(i,j) cell{k,5}(i,j) cell{k,6}(i,j) cell{k,7}(i,j) cell{k,8}(i,j)];
            aver_value=mean(temp);
            temp=temp-aver_value;
            sqr_value=temp.^2;
            total=sum(sqr_value);
            result_2(i,j)=total/8;
        end
    end
    baseFileName=sprintf('Anth_LDPv%d.xlsx',k);
    fullFilename=fullfile(folder,baseFileName);
    xlswrite(fullFilename,result_2);
    
end
