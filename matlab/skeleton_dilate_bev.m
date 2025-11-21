clear all
close all
direc = 'C:\Users\LauraSantos\Documents\Data\CADIn\3Dcomparison';
lista = dir(direc);
nomes = extractfield(lista,'name');
tot_metric = [];
sd_metric = [];
centroid_t_kin = [];
centroid_t_mul = [];

centroid_p_kin = [];
centroid_p_mul = [];
load("variables.mat")
javan= 1;
nomes = {'202MC-17','203MR-08','305MR-02','310MC-11'};
folders = {'0608_MJC','0607_PJ','0620_SF','0531_ARG'};
for inomes =1:4%length(nomes)
    folderdir = dir(strcat(direc,'\',nomes{inomes}));
    nomefolder = folders{inomes};
    listafolder = dir(strcat(direc,'\',nomes{inomes}, '\',nomefolder));
    filesemfolder = extractfield(listafolder,'name');
    listafolderbev = dir(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\bev'));
    filesemfolderbev = extractfield(listafolderbev,'name');
    
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


    bevfilesind = find(contains(filesemfolderbev,'results_2d_bev_adult'));
    bevfiles = filesemfolderbev(1,bevfilesind);
    bevfilesindchild = find(contains(filesemfolderbev,'results_2d_bev_child'));
    bevfileschild = filesemfolderbev(1,bevfilesindchild);
    tbev = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\bev\',bevfiles{1}),'%f');
    pbev = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\bev\',bevfileschild{1}),'%f');
    tbevr = reshape(tbev,49,size(tbev,1)/49);
    pbevr = reshape(pbev,49,size(pbev,1)/49);

    nomechild = nomes{inomes};
    splitfolder = split(nomefolder,'_');
    nometherapist = splitfolder{2};

    tfilesind = find(contains(filesemfolder,'tempos'));
    tfiles = filesemfolder(1,tfilesind);
    tempos = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',tfiles{1}),'%f');
    tmulr(1,:) = tempos(tmulr(1,:)+1);
    pmulr(1,:) = tempos(pmulr(1,:)+1);
    tbevr(1,:) = tempos(tbevr(1,:)+1);
    pbevr(1,:) = tempos(pbevr(1,:)+1);


    tfilesind = find(contains(filesemfolder,'ther_times'));
    tfiles = filesemfolder(1,tfilesind);
    tgtr = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',tfiles{1}),'%f')';

    pfilesind = find(contains(filesemfolder,'pat_times'));
    pfiles = filesemfolder(1,pfilesind);
    pgtr = textread(strcat(direc,'\',nomes{inomes}, '\',nomefolder,'\',pfiles{1}),'%f')';

    %Calculation of RMSE for the therapist considering ground truth,
    %multiperson, kinect.
    [Cgtkin,igt,ikin] = intersect(tgtr(1,:),tkinr(1,:));
    [Cgtkinmul,igt1,imul] = intersect(Cgtkin,tmulr(1,:));

    igt=igt(igt1);
    ikin=ikin(igt1);

    [Cgtkinmulbev,igt2,ibev] = intersect(Cgtkinmul,tbevr(1,:));

    igt=igt(igt2);
    ikin=ikin(igt2);
    imul=imul(igt2);

    selectedgt  = tgtr(1,igt);
    selectedmul = tmulr(2:end,imul);
    selectedkin = tkinr(3:end,ikin);
    selectedbev = tbevr(2:end,ibev);
    frames = igt;

    selectedmulr= reshape(selectedmul,2,size(selectedmul,1)/2,size(selectedmul,2));
    selectedkinr= reshape(selectedkin,2,size(selectedkin,1)/2,size(selectedkin,2));
    selectedbevr= reshape(selectedbev,2,size(selectedbev,1)/2,size(selectedbev,2));

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
    ther_bev = [];

    raio = 0;
    x = gtxyther(javan:javan+1,1);
    y = gtxyther(javan:javan+1,2);
    for t_i =1:length(selectedgt)
        
        curr_im = imread(strcat(full_t,num2str(sortnames(frames(t_i))),'.jpg'))./128 * 255;
        [dice_kin, raio, vector] = calc_dice_found(curr_im,selectedkinr,t_i,1, raio, x, y);
%         [dice_kin, raio, vector,x,y] = calc_dice(curr_im,selectedkinr,t_i,1, raio);
%         centroid_t_kin = [centroid_t_kin; vector];
        ther_kin=[ther_kin dice_kin];

%         gtxyther=[gtxyther; x y];

        [dice_mul, raio, vector] = calc_dice_found(curr_im,selectedmulr,t_i,0, raio, x, y);
%         [dice_mul, raio, vector,~,~] = calc_dice(curr_im,selectedmulr,t_i,0, raio);
%         centroid_t_mul = [centroid_t_mul; vector];
        ther_mul =[ther_mul dice_mul];

        [dice_bev, raio, vector] = calc_dice_found(curr_im,selectedbevr,t_i,2, raio, x, y);
%         [dice_mul, raio, vector,~,~] = calc_dice(curr_im,selectedmulr,t_i,0, raio);
%         centroid_t_mul = [centroid_t_mul; vector];
        ther_bev =[ther_bev dice_bev];
    end

    %Calculation of RMSE for the patient considering ground truth,
    %multiperson, kinect.
    [Cgtkin,igt,ikin] = intersect(pgtr(1,:),pkinr(1,:));
    [Cgtkinmul,igt1,imul] = intersect(Cgtkin,pmulr(1,:));

    igt=igt(igt1);
    ikin=ikin(igt1);

    [Cgtkinmulbev,igt2,ibev] = intersect(Cgtkinmul,pbevr(1,:));

    igt=igt(igt2);
    ikin=ikin(igt2);
    imul=imul(igt2);

    selectedgt  = pgtr(1,igt);
    selectedmul = pmulr(2:end,imul);
    selectedkin = pkinr(3:end,ikin);
    selectedbev = pbevr(2:end,ibev);
    frames = igt;

    selectedmulr= reshape(selectedmul,2,size(selectedmul,1)/2,size(selectedmul,2));
    selectedkinr= reshape(selectedkin,2,size(selectedkin,1)/2,size(selectedkin,2));
    selectedbevr= reshape(selectedbev,2,size(selectedbev,1)/2,size(selectedbev,2));

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
    pat_bev = [];
    
    raio = 0;
    x = gtxypat(javan:javan+1,1);
    y = gtxypat(javan:javan+1,2);
    for t_i =1:length(selectedgt)
        

        curr_im = imread(strcat(full_t,num2str(sortnames(frames(t_i))),'.jpg'))./128 * 255;
        [dice_kin, raio, vector] = calc_dice_found(curr_im,selectedkinr,t_i,1,raio, x, y);
%         [dice_kin, raio, vector, x, y] = calc_dice(curr_im,selectedkinr,t_i,1,raio);
%         centroid_p_kin = [centroid_p_kin; vector];
        pat_kin=[pat_kin dice_kin];


        [dice_mul, raio, vector] = calc_dice_found(curr_im,selectedmulr,t_i,0, raio, x, y);
%         [dice_mul, raio, vector] = calc_dice(curr_im,selectedmulr,t_i,0, raio);
%         centroid_p_mul = [centroid_p_mul; vector];
        pat_mul =[pat_mul dice_mul];
        
        [dice_bev, raio, vector] = calc_dice_found(curr_im,selectedbevr,t_i,2, raio, x, y);
%         [dice_mul, raio, vector] = calc_dice(curr_im,selectedmulr,t_i,0, raio);
%         centroid_p_mul = [centroid_p_mul; vector];
        pat_bev =[pat_bev dice_bev];
    end

    tot_metric=[tot_metric; mean(ther_kin) mean(ther_mul) mean(ther_bev) mean(pat_kin) mean(pat_mul) mean(pat_bev)];
    sd_metric=[sd_metric; std(ther_kin) std(ther_mul) std(ther_bev) std(pat_kin) std(pat_mul) std(pat_bev)];
    javan = javan+2;
    
end
