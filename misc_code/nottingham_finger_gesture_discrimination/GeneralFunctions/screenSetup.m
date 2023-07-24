function cfg=screenSetup(cfg)

%% Screen setup & open window

%Screen specifics
screens = Screen('Screens');% get the screen numbers
cfg.screenNumber = max(screens);% draw to the external screen if avaliable

%Open a window
if cfg.fullscreen
    [window, windowRect] = PsychImaging('OpenWindow', cfg.screenNumber, 0); 
    HideCursor;%Hide the cursor when fullscreen
else
    offset=50;
    [window, windowRect] = PsychImaging('OpenWindow', cfg.screenNumber, 0, [offset offset cfg.manual_resx+offset cfg.manual_resy+offset]);
end
cfg.screen.window=window;

%Size of the window
[cfg.screen.Xpix, cfg.screen.Ypix]=Screen('WindowSize',window);

%Centre of the window
[cfg.screen.xCentre,cfg.screen.yCentre]=RectCenter(windowRect);

%Alpha blending
Screen('BlendFunction', window, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);%Enable alpha blending
Screen('Flip', window);% Flip to clear

%Query the frame duration
cfg.ifi = Screen('GetFlipInterval', window);
cfg.rRHz= 1/cfg.ifi; %refresh rate (Hz)

%Query the maximum priority level and set it
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);
