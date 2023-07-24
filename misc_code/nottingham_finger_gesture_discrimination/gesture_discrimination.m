%% Set the paths
%  HISTORY
% 2018.01.09 LR chaged response key to 66 ('b') index finger on box
% 2018.01.09 LR
%%%%%%%%
tmp=matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));

addpath C:\toolbox\

addpath([cd, '\ExpFunctions\']);%add the sub-functions to it
addpath([cd, '\GeneralFunctions\']);%add the more generic sub-functions to it
addpath(genpath('R:\DRS-OPMMEG\Matlab_files\_Paradigms\Jan\'))
% addpath(['C:\Program ' 'Files\MATLAB\R2012a\toolbox\nottsscripts\']) #Change to relevant
% addpath(['C:\Program ' 'Files\MATLAB\R2012a\toolbox\parallel\'])#Change to relevant
% addpath(['C:\Program ' 'Files\MATLAB\'])


%% Basic setup

% clear all;
close all;sca;fclose('all');AssertOpenGL;PsychDefaultSetup(2); Screen('CloseAll')
global psych_default_colormode;
psych_default_colormode = 1;
skypsinc=0; expBasicSetup(skypsinc);

escapeKey = KbName('q');

response_keys = KbName({'w','b','y','g','r'});
% Trigger channels for each colour button
trig_chans = [8 16 32 64 128];
% initialise key press queue
keylist=zeros(1,256);%%create a list of 256 zeros
keylist([response_keys,escapeKey])=1 ;%set keys you interested in to 1
KbQueueCreate(0,keylist);


do_trig = 1;

%% Initialise Parallel Port IO
if do_trig
    global ioObj PortAddress
    PortAddress = 57336;
    ioObj = io64;
    status = io64(ioObj);
    io64(ioObj,PortAddress,0);
    disp('Ports Cleared')
%     on_trig = 32; % Stim side 1
%     off_trig = 64;
    
    trigset_on = {1, 2, 4};
    trig_off = 7;
    
    
end

%% Subject info
cfg=[];
% cfg.subnr=input('Enter subject number: ');
% cfg.sub=input('Enter subject Code: ','s');

%% Main settings

%Path
cfg.exp_dir=[cd,'\'];
cfg.exp_path=[cd,'\stims\'];

%Screen setup
cfg.LPT = 1; %parallel port triggers are enabled
cfg.propixx=0;                %turn on quadRGB mode for 12 simultaneous frames on the ProPixx
cfg.fullscreen=1;               %open a fullscreen window, if not, manual resolution will be used
cfg.manual_resx=1920;         %manual x resolution
cfg.manual_resy=1080;         %manual y resolution
cfg.backprojection=1;           %is a backprojection screen used? (1=yes, 0=no),
cfg.luminance=0.5;              %Projector luminance ([0 1], 1=maximal)

%Physical screen parameters (cm)
% cfg.width=81.5;
% cfg.height=46;
% cfg.dist=100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SCREEN SETTINGS
cfg.width=48;%34
cfg.height=27;%20
cfg.dist=85;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Stimuli
cfg.Vdist = 0; %1               %vertical distance between dots, in degrees
cfg.Hdist = 5;  %5              %horizontal distance between dots, in degrees
cfg.stimSize = 5;               %grating size, in degrees
cfg.FixSize = 0.2;              %fixation dot size, in degrees

cfg.escapeKey = escapeKey;      %This key is checked during the trial, press to abort
cfg.trigger_duration=0.05;

%Trials
cfg.trialnum = 3*60; % Num trials for each condition *5 (5 conditions total)
do_rand = 1; % Randomise trial order (if 0 does left, right, centre)
if do_rand
    rand_trial = randperm(cfg.trialnum);
else
    rand_trial = 1:cfg.trialnum;
end

% before running this script for the first time, set block_gesture = 0
csvwrite(sprintf("gesture_discrimination_hs_hd_%d.csv", block_gesture), rem(rand_trial,3)+1)
block_gesture = block_gesture+1;

%Timing  parameters
cfg.endTime = 2; %stim offset, in sec
cfg.rest_time = 1.5;


try
    %% Initialize Parallel Port IO & triggers
    if do_trig
        disp('Initialise triggers')
        disp('0000 0000')
        io64(ioObj, PortAddress, 0);
    end
    
    % Trigger once on all channels to indicate start of experiment
    if do_trig
        io64(ioObj, PortAddress, 255);
        pause(cfg.trigger_duration)
        io64(ioObj, PortAddress, 0);
    end
    pause(3)
    
    %% Screen setup & open window
    
    cfg=screenSetup(cfg);
    window=cfg.screen.window;
    
    %% Read stimulus images in
    
    %Read the image files in
    tmp=dir([cfg.exp_path,'*.png']);%get the file names
    
    stim_name = 'gesture_square_';
    
    for i=1:3
        stimset{i}=imread(sprintf('%s%s%d.png',cfg.exp_path,stim_name,i));%read the bmp files in
        picTex{1, i} = Screen('MakeTexture', window, stimset{i});
    end
    
%     for i=1:5
%         stimset{i}=imread(sprintf('%s%s%d.png',cfg.exp_path,stim_name,i));%read the bmp files in
%         picTex{1, i} = Screen('MakeTexture', window, stimset{i});
%     end
    

    %Get their size
    [s1,s2,s3]=size(stimset{1, 1});
    
    %% Fixation dot
    
    [fix,fixTex]=createFixDot(cfg);
    
    %% Stimulus Placement
    
    Vpix=round(((tand(cfg.Vdist)*cfg.dist)*(cfg.screen.Ypix/(cfg.height))));
    Hpix=round(((tand(cfg.Hdist)*cfg.dist)*(cfg.screen.Xpix/(cfg.width))));
    
    shift=[-Hpix Vpix ; +Hpix Vpix ; 0 Vpix]; % Add zero shift (central)
    
    stim_width=round(((tand(cfg.stimSize)*cfg.dist)*(cfg.screen.Ypix/cfg.height)));
    recImg=[0 0 stim_width stim_width];%s1 s2];
    recPos1=CenterRectOnPointd(recImg, (cfg.screen.Xpix/2)+shift(1,1), (cfg.screen.Ypix/2)+shift(1,2));
    recPos2=CenterRectOnPointd(recImg, (cfg.screen.Xpix/2)+shift(2,1), (cfg.screen.Ypix/2)+shift(2,2));
    recPos3=CenterRectOnPointd(recImg, (cfg.screen.Xpix/2)+shift(3,1), (cfg.screen.Ypix/2)+shift(3,2));
    
    fixImg=[0 0 size(fix,1) size(fix,2)];
    fixPos=CenterRectOnPointd(fixImg, (cfg.screen.Xpix/2), (cfg.screen.Ypix/2));
    cfg.screen.fixPos=fixPos;
    
    clear Hpix Vpix
    

    %% Setup Propixx 1440 Hz
    
    %setupPropixx(cfg,0);
    
    %%
    cfg.screen.fix=fix;cfg.screen.fixTex=fixTex;
    
    %%
    
    pause(4)
    
    msg1=['  The experiment is about to start'];
    msg2='Focus on the grey dot at all times';
    h_shift=cfg.screen.Xpix/10;v_shift=cfg.screen.Ypix/10;
    drawMyText (cfg, window, fixPos,fixTex, msg1, msg2, h_shift, v_shift,[1 1 1]);
    vbl = Screen('Flip', window);
    
    
    %RestrictKeysForKbCheck(13);qqqqq
    %     KbWait(1);%RestrictKeysForKbCheck([]);
    pause(1)
    clear msg1 msg2
    Screen('FillRect',window,[255 255 255]);%background black
    vbl = Screen('Flip', window);
    
    for j=1:cfg.trialnum
        
        %%
        recPos(1,1:4)=recPos1; %get the trial specific stimulus position
        recPos_other(1,1:4)=recPos2;
        recPos_centre(1,1:4)=recPos3;
        
        %%
        Screen('DrawTexture', window, fixTex, [], fixPos); % present the fixation dot
        vbl = Screen('Flip', window);
        
        %%%%% Run grating
        picnum=rem(rand_trial(j),3)+1;
        window=cfg.screen.window;
        fixTex=cfg.screen.fixTex;
        fixPos=cfg.screen.fixPos;
        Screen('DrawTextures', window, picTex{1,picnum}, [], recPos_centre);
        Screen('DrawTexture', window, fixTex, [], fixPos);
        
        if do_trig
            io64(ioObj, PortAddress, trigset_on{picnum});
            vbl = Screen('Flip', window, vbl + 0.5 * cfg.ifi);
            pause(cfg.trigger_duration)
            io64(ioObj, PortAddress, 0);
        else
            vbl = Screen('Flip', window, vbl + 0.5 * cfg.ifi);
        end

        Screen('DrawTextures', window, picTex{1,picnum}, [], recPos_centre);
        Screen('DrawTexture', window, fixTex, [], fixPos);
        vbl = Screen('Flip', window, vbl + 0.5 * cfg.ifi);
        
        start_time=GetSecs();t=0;
        while 1
            t=GetSecs-start_time;
            
            [key_pressed, seconds, key_code] = KbCheck;
            if (key_pressed)
                if key_code == cfg.escapeKey
                    error('Experiment aborted by user')

                else
                    send_trig = trig_chans(logical(key_code(response_keys)));
                    
                    % send trig
                    io64(ioObj, PortAddress, send_trig);
                    pause(cfg.trigger_duration)
                    io64(ioObj, PortAddress, 0);
                end
            end
                       
            
            if t >cfg.endTime
                disp('End trigger')
                Screen('FillRect',window,[255 255 255]);%background black
                Screen('DrawTexture', window, fixTex, [], fixPos);
                
                vbl = Screen('Flip', window, vbl + 0.5 * cfg.ifi);
                if do_trig
                    io64(ioObj, PortAddress, trig_off);
                    pause(cfg.trigger_duration)
                    io64(ioObj, PortAddress, 0);
                end
                break
            end
        end
        
        %%%%%%%%%%%%%%%%%
        waitMoveOnEsc(cfg,cfg.rest_time)
    end
    
    if do_trig
        io64(ioObj, PortAddress, 255);
        pause(cfg.trigger_duration)
        io64(ioObj, PortAddress, 0);
    end
    
    msg1=['   End of experiment'];
    msg2='';
    h_shift=cfg.screen.Xpix/18;v_shift=cfg.screen.Ypix/18;
    drawMyText (cfg, window, fixPos,fixTex, msg1, msg2, h_shift, v_shift,[1 1 1]);
    vbl = Screen('Flip', window, vbl + 0.5 * cfg.ifi);
    WaitSecs(1);
    sca
    clear screen
%     KbQueueRelease
catch err
%     cleanup(cfg)
    msg1=['   End of experiment'];
    msg2='';
    h_shift=cfg.screen.Xpix/18;v_shift=cfg.screen.Ypix/18;
    drawMyText (cfg, window, fixPos,fixTex, msg1, msg2, h_shift, v_shift,[1 1 1]);
    vbl = Screen('Flip', window, vbl + 0.5 * cfg.ifi);
    WaitSecs(1);
    sca
    clear screen
    rethrow (err)
end

%