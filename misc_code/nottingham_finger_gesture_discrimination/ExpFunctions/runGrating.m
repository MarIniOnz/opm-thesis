function runGrating(cfg,sendTrigger,picTex,recPos,recPos_other,vbl)

picnum=1; 
window=cfg.screen.window;
fixTex=cfg.screen.fixTex;
fixPos=cfg.screen.fixPos;

KbQueueFlush();%%removes all keyboard presses
KbQueueStart();%%start listening

start_time=GetSecs();t=0;


while 1
    t=GetSecs-start_time;

    if t <cfg.endTime
        picnum=1+mod((picnum-1)+cfg.initspeed,300);
        picnum_other=picnum;
    end

   
    Screen('DrawTextures', window, picTex{1,picnum}, [], recPos);
    Screen('DrawTextures', window, picTex{1,picnum_other}, [], recPos_other);
    Screen('DrawTexture', window, fixTex, [], fixPos);
    
    vbl = Screen('Flip', window, vbl + 0.5 * cfg.ifi);
    
    if t >cfg.endTime
            sendTrignLog(cfg,eylnkMsg,logMsg,LPTtrig,sendTrigger,0);
        break
    end
end

