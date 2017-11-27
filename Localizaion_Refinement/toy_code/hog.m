im = imread('1.jpg');
if size(im,3) == 3
    im = rgb2gray(im);
end
im = double(im);
rows = size(im,1);
cols = size(im,2);
%h = fspecial('gaussian');
%im = imfilter(im,h);
%figure,imshow(im,[]);
im = im.^0.5;   % gamma calibration
%figure,imshow(im,[]);
Ix = im;
Iy = im;
for i = 1:rows-2   % Calculate the gradient in X and Y direction
    Iy(i,:) = im(i+2,:)-im(i,:);
end
    Iy(rows-1,:) = Iy(rows-2,:);
    Iy(rows,:) = Iy(rows-2,:);
for j = 1:cols-2
    Ix(:,j) = im(:,j+2)-im(:,j);
end
    Ix(:,cols-1) = Ix(:,cols-2);
    Ix(:,cols) = Ix(:,cols-2);
angle = atand(Iy./Ix);% Calculate the angle and magnitude of 
angle = imadd(angle,90);
magtu = sqrt(Ix.^2+Iy.^2);
angle(isnan(angle)) = 0;% Remove NaN
magtu(isnan(magtu)) = 0;
%figure,imshow(angle,[]);
%figure,imshow(magtu,[]);
hog_feature = [];
%Find every block
for i = 1:rows/8-1
    for j = 1:cols/8-1
        angle_block = angle((i-1)*8+1:(i-1)*8+16,(j-1)*8+1:(j-1)*8+16);
        magtu_block = magtu((i-1)*8+1:(i-1)*8+16,(j-1)*8+1:(j-1)*8+16);
        %magtu_block = imfilter(mag_block,gaussian);
        block_feature = [];
        %Find every cell in a block
        for ii = 1:2
            for jj = 1:2
                angle_cell = angle_block((ii-1)*8+1:(ii-1)*8+8,(jj-1)*8+1:(jj-1)*8+8);
                magtu_cell = magtu_block((ii-1)*8+1:(ii-1)*8+8,(jj-1)*8+1:(jj-1)*8+8);
                %Find every pixel's belonged bins
                hist_cell = zeros(1,9);
                for p = 1:8
                    for q = 1:8
                        % Bi_linear Interpolation
                        if angle_cell(p,q)>10 && angle_cell(p,q)<=30
                            hist_cell(1)=hist_cell(1)+magtu_cell(p,q)*(30-angle_cell(p,q))/20;
                            hist_cell(2)=hist_cell(2)+magtu_cell(p,q)*(angle_cell(p,q)-10)/20;
                        elseif angle_cell(p,q)>30 && angle_cell(p,q)<=50
                            hist_cell(2)=hist_cell(2)+magtu_cell(p,q)*(50-angle_cell(p,q))/20;
                            hist_cell(3)=hist_cell(3)+magtu_cell(p,q)*(angle_cell(p,q)-30)/20;
                        elseif angle_cell(p,q)>50 && angle_cell(p,q)<=70
                            hist_cell(3)=hist_cell(3)+magtu_cell(p,q)*(70-angle_cell(p,q))/20;
                            hist_cell(4)=hist_cell(4)+magtu_cell(p,q)*(angle_cell(p,q)-50)/20;
                        elseif angle_cell(p,q)>70 && angle_cell(p,q)<=90
                            hist_cell(4)=hist_cell(4)+magtu_cell(p,q)*(90-angle_cell(p,q))/20;
                            hist_cell(5)=hist_cell(5)+magtu_cell(p,q)*(angle_cell(p,q)-70)/20;
                        elseif angle_cell(p,q)>90 && angle_cell(p,q)<=110
                            hist_cell(5)=hist_cell(5)+magtu_cell(p,q)*(110-angle_cell(p,q))/20;
                            hist_cell(6)=hist_cell(6)+magtu_cell(p,q)*(angle_cell(p,q)-90)/20;
                        elseif angle_cell(p,q)>110 && angle_cell(p,q)<=130
                            hist_cell(6)=hist_cell(6)+magtu_cell(p,q)*(130-angle_cell(p,q))/20;
                            hist_cell(7)=hist_cell(7)+magtu_cell(p,q)*(angle_cell(p,q)-110)/20;
                        elseif angle_cell(p,q)>130 && angle_cell(p,q)<=150
                            hist_cell(7)=hist_cell(7)+magtu_cell(p,q)*(150-angle_cell(p,q))/20;
                            hist_cell(8)=hist_cell(8)+magtu_cell(p,q)*(angle_cell(p,q)-130)/20;
                        elseif angle_cell(p,q)>150 && angle_cell(p,q)<=170
                            hist_cell(8)=hist_cell(8)+magtu_cell(p,q)*(170-angle_cell(p,q))/20;
                            hist_cell(9)=hist_cell(9)+magtu_cell(p,q)*(angle_cell(p,q)-150)/20;
                        elseif angle_cell(p,q)>170 && angle_cell(p,q)<=180
                            hist_cell(9)=hist_cell(9)+magtu_cell(p,q)*(190-angle_cell(p,q))/20;
                            hist_cell(1)=hist_cell(1)+magtu_cell(p,q)*(angle_cell(p,q)-170)/20;
                        elseif angle_cell(p,q)>0 && angle_cell(p,q)<=10
                            hist_cell(1)=hist_cell(1)+magtu_cell(p,q)*(angle_cell(p,q)+10)/20;
                            hist_cell(9)=hist_cell(9)+magtu_cell(p,q)*(10-angle_cell(p,q))/20;
                        end
                    end
                end
                block_feature = [block_feature hist_cell];
            end
        end
                % Normalize block_feature using L2-norm
                block_feature = block_feature/sqrt(norm(block_feature)^2+0.00001);
                hog_feature = [hog_feature block_feature];
    end 
end
hog_feature(isnan(hog_feature)) = 0;
%Normalize the hog_feature using L2-Hys
hog_feature = hog_feature/sqrt(norm(hog_feature)^2+0.00001);
for z = 1:length(hog_feature)
    if hog_feature(z)>0.2
        hog_feature(z)=0.2
    end
end
hog_feature = hog_feature/sqrt(norm(hog_feature)^2+0.00001);