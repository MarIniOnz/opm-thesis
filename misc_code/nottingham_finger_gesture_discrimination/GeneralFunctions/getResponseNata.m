function [resp,t]=getResponseNata(cfg,timeout)

KbQueueFlush();%%removes all keyboard presses
KbQueueStart();%%start listening

start_resp_time=GetSecs();t=0;
try
    resp_key1=cfg.keys(1,1);resp_key2=cfg.keys(1,2);
catch
    error('error because getResponseNata needs the keycode(s), but they are not specified in the cfg fields (i.e.cfg.keys)')
end

pressed=0;resp=0;
while t<timeout
    [pressed, firstpress] = KbQueueCheck(); %check response
    t=GetSecs-start_resp_time;
    if firstpress(resp_key1)>0
        resp=1;break;%match
    elseif firstpress(resp_key2)>0
        resp=2;break%mismatch
    end
end

