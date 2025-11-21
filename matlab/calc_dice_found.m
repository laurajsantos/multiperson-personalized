function [dice, raio, vector] = calc_dice_found(curr_im,selectedkinr,t_i,tag, raio, xalt, yalt)


        
[y, x] = ndgrid(1:size(curr_im, 1), 1:size(curr_im, 2));
centroid = mean([x(logical(curr_im)), y(logical(curr_im))]);
% rgb= insertMarker(curr_im, centroid,'x','color','red','size',10);
%         imshow(rgb)
centroidkin = mean(selectedkinr(:,:,t_i),2);
imblack= curr_im*0;
vector = [];

% full_im =desenhaskelframe2d(imblack,selectedkinr,t_i,tag);
% tot_im = desenhaskelframe2d(rgb,selectedkinr,t_i,tag);
% skel_curr_im = bwskel(logical(curr_im));
% centroid = mean([x(logical(skel_curr_im)), y(logical(skel_curr_im))]);
% figure(1);
% imshow(curr_im)
% figure(2);
% imshow(skel_curr_im)
if ~isinf(centroidkin(1)) && ~isinf(centroidkin(2))
%     rgb_skel= insertMarker(tot_im, centroidkin','x','color','red','size',10);

    if t_i==1 && tag==1
        
%         figure(1);
%         imshow(curr_im)
%         [xalt,yalt]=getpts;
        raio= sqrt((xalt(1)-xalt(2))^2 + (yalt(1)-yalt(2))^2);
    end
    
    new_skin = selectedkinr(:,:,t_i);%+ (centroid-centroidkin')';
    vector = (centroid-centroidkin');
%     im_fin=desenhaskelframe2d(rgb_skel,new_skin,-1,tag);
    mask_fin=desenhaskelframe2d(imblack,new_skin,-1,tag);
    se=strel('disk',round(raio/2));
    mask_fin_dil = imdilate(mask_fin,se);
%     if t_i==1 && tag==1
%         figure(2);
%         imshow(mask_fin_dil)
%         figure(3)
%         imshow(mask_fin)
%     end
    
    % figure(2);
    % imshow(im_fin)
    
    mask1=curr_im;
    mask2=mask_fin_dil;
    
    unionMask = mask1 | mask2;
    andMask = mask1 & mask2;
    dice = (nnz(andMask) / nnz(unionMask));%*(nnz(mask1) / nnz(mask2)) ;
else
    dice = 0;
end