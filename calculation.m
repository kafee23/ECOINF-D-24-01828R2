function decimal=calculation(f_count,s_count,th_count)
a=f_count;
b=s_count;
c=th_count;
data=zeros(1,8);

for i=1:8
    if(i==b||i==a||i==c)
    data(1,i)=1;
    end
end
binary=sprintf('%d',data);
decimal=bin2dec(binary);
end