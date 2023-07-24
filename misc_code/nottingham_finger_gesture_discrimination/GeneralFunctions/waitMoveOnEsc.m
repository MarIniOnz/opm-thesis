function waitMoveOnEsc(cfg,timeout)

waitstart=GetSecs;moveon=false; window=cfg.screen.window;

while ~moveon
    [ keyIsDown, keyTime, keyCode ] = KbCheck;
    
    if keyIsDown && keyCode(cfg.escapeKey)
        error('Experiment aborted by user')
    end
    if (GetSecs-waitstart)>timeout
        moveon=true;
    end
end
end
