function c =edge(Img,mask_value)
I=Img;
East=mask_value;
Im = double(I);
[row,col]=size(Im);
c = zeros(row-2,col-2);
for i=1:row
    if(i<=row-2)
        for j=1:col
            if(j<=col-2)
            n=1;
            sum =0;
            for k=i:i+2
                 m=1;
                for l=j:j+2
                    sum=sum+East(n,m).*Im(k,l);
                    m=m+1;
                end
                n=n+1;
            end
            c(i,j)=sum;
            fprintf('\n');
            end
        end
    end
end
end