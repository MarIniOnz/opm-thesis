function condMat=createCondMat(cfg,recPos1,recPos2,recPos3)

tmp1=repmat(recPos1,cfg.trialnum/2,1,(cfg.blocknum+cfg.pract));tmp1=permute(tmp1,[3 1 2]);%30 trials within each block with left hand side presentation
tmp2=repmat(recPos2,cfg.trialnum/2,1,(cfg.blocknum+cfg.pract));tmp2=permute(tmp2,[3 1 2]);%30 trials within each block with right hand side presentation
catchtr=floor((cfg.trialnum/2)/10);%3 catch trials for each condition (left, right);
% catchtr=3;%3 catch trials for each condition (left, right);
tmp3=tmp1(:,1:catchtr,:);tmp4=tmp2(:,1:catchtr,:);

condMat=cat(2,tmp1,tmp2,tmp3,tmp4);%combine the matrices
clear tmp1 tmp2 tmp3 tmp4

catch1Start=((cfg.trialnum+(2*catchtr))-((2*catchtr)-1));%indexes of the catchtrial rows

for bl=1:size(condMat,1)
    for i=1:size(condMat,2)
        tmp=(squeeze(condMat(bl,i,1:4)))';
        if isequal(tmp,recPos1)
            condMat(bl,i,5)=cfg.cueLeftTex;
            condMat(bl,i,7:10)=recPos2;
%             condMat(bl,i,11)=cfg.LPT_tr.trig_left;
            condMat(bl,i,11)=cfg.LPT_tr.trig_left_gratings_on;
        elseif isequal(tmp,recPos2)
            condMat(bl,i,5)=cfg.cueRightTex;
            condMat(bl,i,7:10)=recPos1;
%             condMat(bl,i,11)=cfg.LPT_tr.trig_right;
            condMat(bl,i,11)=cfg.LPT_tr.trig_right_gratings_on;
        end
        clear tmp
    end
end

for i=1:(cfg.blocknum+cfg.pract) %add the switchtime (random)    
    swtime=cfg.switchStartT+(cfg.switchEndT-cfg.switchStartT).*rand(cfg.trialnum/2,1);
    condMat(i,1:cfg.trialnum/2,6)=swtime;condMat(i,cfg.trialnum/2+1:cfg.trialnum,6)=swtime;
    clear swtime
    condMat(i,catch1Start:end,6)=99;
end %the created 3D matrix = 1st dim - block, 2nd dim - trialnum, 3rd dim 1-4 - position, 3rd dim 5-fixation dot presentation, 3rd dim 6-switchtime

for i=1:(cfg.blocknum+cfg.pract) %add the switchtime (random)    
while true
    othswtime1=(cfg.othswtime1_end-cfg.othswtime1_start).*rand(cfg.trialnum/2,1,'double')+cfg.othswtime1_start;
    condMat(i,1:cfg.trialnum/2,12)=othswtime1;condMat(i,cfg.trialnum/2+1:cfg.trialnum,12)=othswtime1;
    clear othswtime1
    othswtime2=(cfg.othswtime2_end-cfg.othswtime2_start).*rand(cfg.trialnum/2,1,'double')+cfg.othswtime2_start;
    condMat(i,1:cfg.trialnum/2,13)=othswtime2;condMat(i,cfg.trialnum/2+1:cfg.trialnum,13)=othswtime2;
    clear othswtime2
    
    othswtime1=(cfg.othswtime1_end-cfg.othswtime1_start).*rand(catchtr,1,'double')+cfg.othswtime1_start;
    condMat(i,catch1Start:end,12)=othswtime1;clear othswtime1
    othswtime2=(cfg.othswtime2_end-cfg.othswtime2_start).*rand(catchtr,1,'double')+cfg.othswtime2_start;
    condMat(i,catch1Start:end,13)=othswtime2;clear othswtime2
    
    if size(cfg.dsp1,2)==1 %first change
      tmp=cfg.dsp1;
    else
      tmp=(cfg.dsp1(1,1):cfg.dsp1(1,2));
    end
    
    for j=1:(cfg.trialnum/2)%actual trials
      if size(cfg.dsp1,2)==1
        condMat(i,j,14)=tmp;
      else
        condMat(i,j,14)=tmp(1,randi([1,size(tmp,2)],1));
      end
        condMat(i,j+(cfg.trialnum/2),14)=condMat(i,j,14);
    end
    
    
    for j=catch1Start:size(condMat,2) %catch trials
        if size(cfg.dsp1,2)==1
            condMat(i,j,14)=tmp;
        else
            condMat(i,j,14)=tmp(1,randi([1,size(tmp,2)],1));
        end
    end
    clear tmp
    
    
    if size(cfg.dsp2,2)==1 %second change
      tmp=cfg.dsp2;
    else
      tmp=(cfg.dsp2(1,1):cfg.dsp2(1,2));
    end

    for j=1:(cfg.trialnum/2) %actual trials
      if size(cfg.dsp2,2)==1
        condMat(i,j,15)=tmp;
      else
        condMat(i,j,15)=tmp(1,randi([1,size(tmp,2)],1));
      end
      condMat(i,j+(cfg.trialnum/2),15)=condMat(i,j,15);
    end
    
    for j=catch1Start:size(condMat,2) %catch trials
        if size(cfg.dsp2,2)==1
            condMat(i,j,15)=tmp;
        else
            condMat(i,j,15)=tmp(1,randi([1,size(tmp,2)],1));
        end
    end
    clear tmp
    
    w_speed=cfg.switchStartT-condMat(i,(1:(catch1Start-1)),12);w_speed(w_speed<0)=0;
    w_speed= w_speed.*condMat(i,(1:(catch1Start-1)),14);
    
    if size(cfg.dsp1,2)==1 && size(cfg.dsp2,2)==1
      break
    elseif abs(mean(w_speed))<0.001
        break
    end
end %the created 3D matrix = 1st dim - block, 2nd dim - trialnum, 3rd dim 1-4 - position, 3rd dim 5-fixation dot presentation, 3rd dim 6-switchtime
end

clear catch1Start

for i=1:(cfg.blocknum+cfg.pract) %randomize the trials within each block
    idx = (randperm(size(condMat,2)))';
    condMat(i,:,:)=condMat(i,idx,:);
end

