clear all
close all
direc = 'C:\Users\LauraSantos\Documents\Data\CADIn\New_folder';
lista = dir(direc);
nomes = extractfield(lista,'name');
foldersind = find(contains(nomes,'_'));
foldersnames = nomes(1, foldersind);
fA= [];
fC = [];
sumA = [];
sumC = [];
nameA= {};
nameC = {};
lastfA = 1;
lastfC = 1;
c = containers.Map;
c('SL') = 1.57;
c('PJ') = 1.7;
c('AR') = 1.68;
c('HC') = 1.67;
c('LV') = 1.6;
c('MJ') = 1.63;
cumulateh = [];
horder = [];
distorder = [];
cumulatedist = [];
medias = [];
for ifold=1:length(foldersnames)-1
    focal_file = strcat(direc,'\',foldersnames{ifold},'\','focal_length.txt');
    A = textread(focal_file,'%s');
    indices = 1:9:size(A,1)-8;
    labels = A(indices);
    A(indices) = [];
    nA = cellfun(@str2num,A(:),'un',0);
    newnA = cell2mat(nA);
    rA = reshape(newnA,8,size(newnA,1)/8);
    indA = find(ismember(labels,'adult'));
    indC = find(ismember(labels,'child'));
    if ~isempty(indA)
        selrA = rA(:,indA);
        dist = selrA(2,:);
        medA = median(selrA(3,:));
        mediaA = mean(selrA(3,:));
        p25c = prctile(selrA(3,:), 25);
        p75c = prctile(selrA(3,:), 75);
        medbA = median(selrA(5,:));
        p25cb = prctile(selrA(5,:), 25);
        p75cb = prctile(selrA(5,:), 75);
        medcA = median(selrA(7,:));
        p25cc = prctile(selrA(7,:), 25);
        p75cc = prctile(selrA(7,:), 75);
        nameA{end+1} = foldersnames{ifold}(6:7);
%         if ifold==6
%             medA = median(selrA(4,:));
%             p25c = prctile(selrA(4,:), 25);
%             p75c = prctile(selrA(4,:), 75);
%             medbA = median(selrA(6,:));
%             p25cb = prctile(selrA(6,:), 25);
%             p75cb = prctile(selrA(6,:), 75);
%             medcA = median(selrA(8,:));
%             p25cc = prctile(selrA(8,:), 25);
%             p75cc = prctile(selrA(8,:), 75);
%             addfA = selrA([4,6,8],:);
%             fA = [fA selrA([4,6,8],:)];
%             
%         else
            addfA = selrA([3,5,7],:);
            fA = [fA selrA([3,5,7],:)];
%         end
        
        cumulateh = [cumulateh ones(1,size(selrA,2))*c(nameA{end})];
        horder = [horder c(nameA{end})];
        cumulatedist = [cumulatedist dist];
        distorder = [distorder dist(1)];
        medias = [medias mediaA];
        sumA = [sumA; [medA p25c p75c selrA(2,1) medbA p25cb p75cb medcA p25cc p75cc]];
        figure(1)
%         subplot 211
%         scatter(lastfA:lastfA+length(addfA(1,:))-1, addfA(1,:)./addfA(2,:))
%         ylabel ('focal-length/bbox')
%         ylim([0 2])
%         title('focal-length/bbox - adult')
%         hold on 

        subplot 211
        scatter(lastfA:lastfA+ length(addfA(1,:))-1, addfA(1,:)./addfA(3,:))
        ylabel ('focal-length/s')
        ylim([140 460])
        title('focal-length/s - adult')
        hold on 
        lastfA = lastfA + length(addfA(1,:));
        
    end

    if ~isempty(indC)
        listafiles = dir(strcat(direc,'\',foldersnames{ifold}));
        nomes = extractfield(listafiles,'name');
        anind = find(contains(nomes,'anagra'));
        filean = nomes(1,anind);
        B = textread(strcat(direc,'\',foldersnames{ifold},'\',filean{:}),'%s');
        nameC{end+1} =  B{1};
        selrA = rA(:,indC);
        medA = median(selrA(4,:));
        p25c = prctile(selrA(4,:), 25);
        p75c = prctile(selrA(4,:), 75);
        medbA = median(selrA(6,:));
        p25cb = prctile(selrA(6,:), 25);
        p75cb = prctile(selrA(6,:), 75);
        medcA = median(selrA(8,:));
        p25cc = prctile(selrA(8,:), 25);
        p75cc = prctile(selrA(8,:), 75);
        sumC = [sumC; [medA p25c p75c selrA(2,1) medbA p25cb p75cb medcA p25cc p75cc]];
        fC = [fC selrA([4,6,8],:)];
        addfC = selrA([4,6,8],:);
%         subplot 212
%         scatter(lastfC:lastfC+ length(addfC(1,:))-1, addfC(1,:)./addfC(2,:))
%         ylabel ('focal-length/bbox')
%         ylim([0 2])
%         title('focal-length/bbox - child')

        subplot 212
        scatter(lastfC:lastfC+ length(addfC(1,:))-1, addfC(1,:)./addfC(3,:))
        ylabel ('focal-length/s')
        ylim([140 460])
        title('focal-length/s - child')
        hold on 
        lastfC = lastfC + length(addfC(1,:));
    end
    
    
    

%     figure(2)
%     subplot 211
%     scatter(lastfA:lastfA+ length(addfA(1,:))-1, addfA(1,:)./addfA(3,:))
%     ylabel ('focal-length/s')
%     ylim([140 460])
%     title('focal-length/s - adult')
%     hold on 
%     
%     subplot 212
%     scatter(lastfC:lastfC+ length(addfC(1,:))-1, addfC(1,:)./addfC(3,:))
%     ylabel ('focal-length/s')
%     ylim([140 460])
%     title('focal-length/s - child')
%     hold on 

    
    



end
x = 1: length(fA(1,:));
xc = 1: length(fC(1,:));
figure;
subplot 211
scatter(x, fA(1,:)./fA(2,:))
ylabel ('focal-length/bbox')
ylim([0 2])
title('focal-length/bbox - adult')

subplot 212
scatter(xc, fC(1,:)./fC(2,:))
ylabel ('focal-length/bbox')
ylim([0 2])
title('focal-length/bbox - child')
saveas(gcf,strcat('C:\Users\LauraSantos\Pictures\focal-length\division_bbox.png'))

figure;
subplot 211
scatter(x, fA(1,:)./fA(3,:))
ylabel ('focal-length/s')
ylim([140 460])
title('focal-length/s - adult')

subplot 212
scatter(xc, fC(1,:)./fC(3,:))
ylabel ('focal-length/s')
ylim([140 460])
title('focal-length/s - child')
saveas(gcf,strcat('C:\Users\LauraSantos\Pictures\focal-length\division_s.png'))

figure;
subplot 211
scatter(x, fA(1,:)./(fA(3,:).*fA(2,:)))
ylabel ('focal-length/(bbox*s)')
ylim([1 1.6])
title('focal-length/(bbox*s) - adult')

subplot 212
scatter(xc, fC(1,:)./(fC(3,:).*fC(2,:)))
ylim([1 1.6])
ylabel ('focal-length/(bbox*s)')
title('focal-length/(bbox*s) - child')
saveas(gcf,strcat('C:\Users\LauraSantos\Pictures\focal-length\division_bbox_s.png'))

%% RANSAC method to reject the outliers
figure;
scatter(cumulateh, fA(1,:));
figure;
scatter(horder, sumA(:,1));
Parameters = [];

%% Considering all points for the RANSAC method
close all
figure;
scatter(cumulateh, fA(1,:));
hold on
points = [cumulateh' fA(1,:)'];
modelLeastSquares = polyfit(points(:,1),points(:,2),1);
x = [min(points(:,1)) max(points(:,1))];
y = modelLeastSquares(1)*x + modelLeastSquares(2);
model_bef = modelLeastSquares;

yfit = polyval(modelLeastSquares, points(:,1));          % Estimated  Regression Line
SStot = sum((points(:,2)-mean(points(:,2))).^2);                    % Total Sum-Of-Squares
SSres = sum((points(:,2)-yfit).^2);                       % Residual Sum-Of-Squares
Rsq_bef = 1-SSres/SStot;
SSres_bef = SSres;

plot(points(:,1),yfit,'r-')
hold on
sampleSize = 2; % number of points to sample per trial
maxDistance = 230; % max allowable distance for inliers
cumsq = 1./(cumulatedist.^2);
fitLineFcn = @(points) polyfit(points(:,1),points(:,2),1); % fit function using polyfit
evalLineFcn = ...   % distance evaluation function
  @(model, points) sum(cumsq.*((points(:, 2) - polyval(model, points(:,1))).^2),2);

[modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
  sampleSize,maxDistance, 'Confidence',99.6);
modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);

inlierPts = points(inlierIdx,:);
% x = [min(inlierPts(:,1)) max(inlierPts(:,1))];
x = inlierPts(:,1);
y = modelInliers(1)*x + modelInliers(2);
model_aft = modelInliers;

yfit = polyval(modelInliers, inlierPts(:,1));          % Estimated  Regression Line
SStot = sum((inlierPts(:,2)-mean(inlierPts(:,2))).^2);                    % Total Sum-Of-Squares
SSres = sum((inlierPts(:,2)-yfit).^2);                       % Residual Sum-Of-Squares
Rsq_aft = 1-SSres/SStot;
SSres_aft = SSres;
SStot_aft = SStot;
Parameters = [Parameters; [Rsq_aft SSres_aft SStot_aft model_aft]];
hold on
scatter(x,y,'m')
hold on
plot(x, y, 'g-')
ylim([340,480])
legend('Initial points','Least squares fit','inlinerpoints','Robust fit');
xlabel('Height')
ylabel('Focal length')
hold off
%% Considering just the medians for the RANSAC method
Parameters = [];

close all
figure;
scatter(horder, sumA(:,1));
hold on
points = [horder' sumA(:,1)];
modelLeastSquares = polyfit(points(:,1),points(:,2),1);
x = [min(points(:,1)) max(points(:,1))];
y = modelLeastSquares(1)*x + modelLeastSquares(2);
model_bef = modelLeastSquares;

yfit = polyval(modelLeastSquares, points(:,1));          % Estimated  Regression Line
SStot = sum((points(:,2)-mean(points(:,2))).^2);                    % Total Sum-Of-Squares
SSres = sum((points(:,2)-yfit).^2);                       % Residual Sum-Of-Squares
Rsq_bef = 1-SSres/SStot;
SStot_bef = SStot;
SSres_bef = SSres;
plot(x,y,'r-')
hold on
sampleSize = 2; % number of points to sample per trial
maxDistance = 230; % max allowable distance for inliers
cumsq = 1./(distorder.^2);
fitLineFcn = @(points) polyfit(points(:,1),points(:,2),1); % fit function using polyfit
evalLineFcn = ...   % distance evaluation function
  @(model, points) sum(cumsq.*((points(:, 2) - polyval(model, points(:,1))).^2),2);

[modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
  sampleSize,maxDistance,'Confidence',99.6);
modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);

inlierPts = points(inlierIdx,:);
% x = [min(inlierPts(:,1)) max(inlierPts(:,1))];
x = inlierPts(:,1);
y = modelInliers(1)*x + modelInliers(2);
model_aft = modelInliers;

yfit = polyval(modelInliers, inlierPts(:,1));          % Estimated  Regression Line
SStot = sum((inlierPts(:,2)-mean(inlierPts(:,2))).^2);                    % Total Sum-Of-Squares
SSres = sum((inlierPts(:,2)-yfit).^2);                       % Residual Sum-Of-Squares
Rsq_aft = 1-SSres/SStot;
SStot_aft = SStot;
SSres_aft = SSres;
Parameters = [Parameters; [Rsq_aft SSres_aft SStot_aft model_aft]];


hold on
scatter(x,y,'m')
hold on
plot(x, y, 'g-')
ylim([340,480])
legend('Initial points','Least squares fit','inlinerpoints','Robust fit');
xlabel('Height')
ylabel('Focal length')
hold off

%% Considering just the means for the RANSAC method
close all
figure;
scatter(horder, medias);
hold on
points = [horder' medias'];
modelLeastSquares = polyfit(points(:,1),points(:,2),1);
x = [min(points(:,1)) max(points(:,1))];
y = modelLeastSquares(1)*x + modelLeastSquares(2);
model_bef = modelLeastSquares;

yfit = polyval(modelLeastSquares, points(:,1));          % Estimated  Regression Line
SStot = sum((points(:,2)-mean(points(:,2))).^2);                    % Total Sum-Of-Squares
SSres = sum((points(:,2)-yfit).^2);                       % Residual Sum-Of-Squares
Rsq_bef = 1-SSres/SStot;
SStot_bef = SStot;
SSres_bef = SSres;
plot(x,y,'r-')
hold on
sampleSize = 2; % number of points to sample per trial
maxDistance = 230; % max allowable distance for inliers
cumsq = 1./(distorder.^2);
fitLineFcn = @(points) polyfit(points(:,1),points(:,2),1); % fit function using polyfit
evalLineFcn = ...   % distance evaluation function
  @(model, points) sum(cumsq.*((points(:, 2) - polyval(model, points(:,1))).^2),2);

[modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
  sampleSize,maxDistance,'Confidence',99.6);
modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);

inlierPts = points(inlierIdx,:);
% x = [min(inlierPts(:,1)) max(inlierPts(:,1))];
x = inlierPts(:,1);
y = modelInliers(1)*x + modelInliers(2);
model_aft = modelInliers;

yfit = polyval(modelInliers, inlierPts(:,1));          % Estimated  Regression Line
SStot = sum((inlierPts(:,2)-mean(inlierPts(:,2))).^2);                    % Total Sum-Of-Squares
SSres = sum((inlierPts(:,2)-yfit).^2);                       % Residual Sum-Of-Squares
Rsq_aft = 1-SSres/SStot;
SStot_aft = SStot;
SSres_aft = SSres;
Parameters = [Parameters; [Rsq_aft SSres_aft SStot_aft model_aft]];

hold on
scatter(x,y,'m')
hold on
plot(x, y, 'g-')
ylim([340,480])
legend('Initial points','Least squares fit','inlinerpoints','Robust fit');
xlabel('Height')
ylabel('Focal length')
hold off

%% Read image
figure;
A = imread('C:\Users\LauraSantos\Pictures\2815430.jpg');
imshow 'C:\Users\LauraSantos\Pictures\2815430.jpg'
pixels = ginput(5);

%%
close all
newy = 468-pixels(:,2);
z = [1.9; 2.5; 3.1; 3.7; 4.3];
figure;
scatter(newy,z)
xlabel('pixels')
ylabel('z (m)')
Z_centrale = 3.7;
h = -0.895;
f = 468/2/h*Z_centrale;
yest = (f./z)*h;
deltay = yest-pixels(:,2);
figure;
plot(z,deltay);
xlabel('z (m)')
ylabel('deltay')
