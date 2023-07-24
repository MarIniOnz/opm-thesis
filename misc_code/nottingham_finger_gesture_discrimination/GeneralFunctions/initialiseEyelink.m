function cfg=initialiseEyelink(cfg)

try
    if cfg.eylnk==1
        %add eyelink script folder (should be in main experiment folder)
        addpath([cfg.exp_dir, 'Eyelink']);
        
        % Define screen for calibration
        cfg.el_rect=[0 0  cfg.screen.Xpix cfg.screen.Ypix];
        
        %make directory if it doesn't already exist (local computer)
        cfg.el.eyedir = [cfg.exp_dir, 'Eyelink' filesep ];
        if ~exist(cfg.el.eyedir, 'dir'); mkdir(cfg.el.eyedir);end
        
        %check whether files already exist for this subject/session
        if exist([cfg.exp_dir 'Eyelink' filesep cfg.el.edffile '.edf'])>0
            cont=input('Warning! Eyelink file will be overwritten, do you want to continue? (y/n) ','s');
            if cont=='n'
                error('Session aborted')
            end
        end
        
        % Set parameters, start and calibrate eyelink
        cfg = el_Start(cfg);
        
        %Parameters for fixation control
        %     fixWinSize=round(((tand(cfg.el.fixation_window)*cfg.dist)*(cfg.screen.resy/cfg.height)));
        %     cfg.el.fixationWindow = [-fixWinSize -fixWinSize fixWinSize fixWinSize];
        
    else
        warning( 'WARNING: Eyetracker is set to off (see cfg.eylnk)! Eyelink triggers will not be sent!');
        while true
            m=input('Do you want to continue? y/n   ','s');
            if m=='y'
                cfg.eylnk=0;
                break
            elseif m=='n'
                error('The experiment aborted.')
            end
        end
    end
catch
    warning( 'WARNING: Eyetracker setup failed! Eyelink triggers will not be sent!');
    while true
        m=input('Do you want to continue? y/n   ','s');
        if m=='y'
            cfg.eylnk=0;
            break
        elseif m=='n'
            error('The experiment aborted.')
        end
    end
end