function [T,area_ration,r,a,b]=certificate_circle(BW)
%************************************
%    函数功能为判断区域是否为圆形
%    BW是只含有一个区域的二值图像
%    如果判断是圆形，则返回值T=1，否则返回值为T=0
%************************************
T=0;
se=strel('square',3);
erode_image=imerode(BW,se);
bound_image=BW-erode_image;
[y,x]=find(bound_image==1);
Npoint=sum(sum(BW));   %Npoint是区域像素点数目（区域面积）
[r,a,b]=(nihe(x,y));%坐标取整数
r=round(r);
a=round(a);
b=round(b);
%通过包含的像素点（面积）判断是否为圆形,每个像素点的宽定为单位1
s_area=pi*r*r;
goal_area=Npoint;
area_ration=s_area/goal_area;
if area_ration>1
    area_ration=1/area_ration;
end
if area_ration>=0.2&&area_ration<=1.9%判定为圆形的条件
    T=1;
end
if Npoint<800
    T=0;    %若交通标志太小，则放弃识别
end

