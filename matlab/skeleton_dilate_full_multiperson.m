clear all
close all
direc = 'C:\Users\LauraSantos\Documents\Data\CADIn\3Dcomparison';
% lista = dir(direc);
% nomes = extractfield(lista,'name');
tot_metric = [];
sd_metric = [];
centroid_t_kin = [];
centroid_t_mul = [];
centroid_p_kin = [];
centroid_p_mul = [];
javan = 1;
load('variables.mat')
nomes = {'202MC-17','203MR-08','305MR-02','310MC-11'};
folders = {'0608_MJC','0607_PJ','0620_SF','0531_ARG'};
for inomes =1:4%length(nomes)
    folderdir = dir(strcat(direc,'\',nomes{inomes}));
    nomefolder = folders{inomes};
    listafolder = dir(strcat(direc,'\',nomes{inomes}, '\',nomefolder));
    filesemfolder = extractfield(listafolder,'name');
    
    jointsfilesind = find(contains(filesemfolder,'jointpoints') & ~contains(filesemfolder,'jointpointschild'));
    jointsfiles = filesemfolder(1,jointsfilesind);
    jointsfilesindchild = find(contains(filesemfolder,'jointpointschild'));
    jointsfileschild = filesemfolder(1,jointsfilesindchild);
    tkin = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',jointsfiles{1}),'%f');
    pkin = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',jointsfileschild{1}),'%f');
    tkinr = reshape(tkin,52,size(tkin,1)/52);
    pkinr = reshape(pkin,52,size(pkin,1)/52);
    
    mulfilesind = find(contains(filesemfolder,'alt_results_adult2d'));
    mulfiles = filesemfolder(1,mulfilesind);
    mulfilesindchild = find(contains(filesemfolder,'alt_results_child2d'));
    mulfileschild = filesemfolder(1,mulfilesindchild);
    tmul = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',mulfiles{1}),'%f');
    pmul = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',mulfileschild{1}),'%f');
    tmulr = reshape(tmul,49,size(tmul,1)/49);
    pmulr = reshape(pmul,49,size(pmul,1)/49);

    nomechild = nomes{inomes};
    splitfolder = split(nomefolder,'_');
    nometherapist = splitfolder{2};

    tfilesind = find(contains(filesemfolder,'tempos'));
    tfiles = filesemfolder(1,tfilesind);
    tempos = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',tfiles{1}),'%f');
    tmulr(1,:) = tempos(tmulr(1,:)+1);
    pmulr(1,:) = tempos(pmulr(1,:)+1);

    tfilesind = find(contains(filesemfolder,'ther_times'));
    tfiles = filesemfolder(1,tfilesind);
    tgtr = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',tfiles{1}),'%f')';

    pfilesind = find(contains(filesemfolder,'pat_times'));
    pfiles = filesemfolder(1,pfilesind);
    pgtr = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',pfiles{1}),'%f')';

    %Calculation of RMSE for the therapist considering ground truth,
    %multiperson, kinect.
    [Cgtmul,iagtmul,ibgtmul] = intersect(tgtr(1,:),tmulr(1,:));
    frames = iagtmul;

    selectedgt  = tgtr(1,iagtmul);
    selectedmul = tmulr(2:end,ibgtmul);

    selectedmulr= reshape(selectedmul,2,size(selectedmul,1)/2,size(selectedmul,2));

    selectedmulr(1,:,:) = selectedmulr(1,:,:)*1980/832;
    selectedmulr(2,:,:) = selectedmulr(2,:,:)*1080/480;


    full_t = strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\', 'therapist\');
    listafolder_t = dir(full_t);
    filesemfolder_t = extractfield(listafolder_t,'name');
    filesemfolder_t = filesemfolder_t(3:end);
    splitfiles= split(filesemfolder_t,'.');
    names=str2mat(splitfiles(1,:,1));
    namesc=str2num(names);
    sortnames = sort(namesc);
    ther_kin = [];
    ther_mul = [];
    
    raio = 0;
    x = gtxyther(javan:javan+1,1);
    y = gtxyther(javan:javan+1,2);
    for t_i =1:length(selectedgt)
        curr_im = imread(strcat(full_t,num2str(sortnames(frames(t_i))),'.jpg'))./128 * 255;
        
        [dice_mul, raio, vector] = calc_dice_found(curr_im,selectedmulr,t_i,1, raio, x, y);
        centroid_t_mul = [centroid_t_mul; vector];
        ther_mul =[ther_mul dice_mul];
    end

    %Calculation of RMSE for the patient considering ground truth,
    %multiperson, kinect.
    [Cgtmul,iagtmul,ibgtmul] = intersect(pgtr(1,:),pmulr(1,:));
    frames = iagtmul;

    selectedgt  = pgtr(1,iagtmul);
    selectedmul = pmulr(2:end,ibgtmul);

    selectedmulr= reshape(selectedmul,2,size(selectedmul,1)/2,size(selectedmul,2));

    selectedmulr(1,:,:) = selectedmulr(1,:,:)*1980/832;
    selectedmulr(2,:,:) = selectedmulr(2,:,:)*1080/480;


    full_t = strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\', 'patient\');
    listafolder_t = dir(full_t);
    filesemfolder_t = extractfield(listafolder_t,'name');
    filesemfolder_t = filesemfolder_t(3:end);
    splitfiles= split(filesemfolder_t,'.');
    names=str2mat(splitfiles(1,:,1));
    namesc=str2num(names);
    sortnames = sort(namesc);
    pat_kin = [];
    pat_mul = [];
    
    raio = 0;
    x = gtxypat(javan:javan+1,1);
    y = gtxypat(javan:javan+1,2);
    for t_i =1:length(selectedgt)

        curr_im = imread(strcat(full_t,num2str(sortnames(frames(t_i))),'.jpg'))./128 * 255;
        

        [dice_mul, raio, vector] = calc_dice_found(curr_im,selectedmulr,t_i,1, raio, x, y);
        centroid_p_mul = [centroid_p_mul; vector];
        pat_mul =[pat_mul dice_mul];
    end

    tot_metric=[tot_metric; mean(ther_mul) mean(pat_mul)];
    sd_metric=[sd_metric; std(ther_mul) std(pat_mul)];
    javan=javan+2;

    
end
