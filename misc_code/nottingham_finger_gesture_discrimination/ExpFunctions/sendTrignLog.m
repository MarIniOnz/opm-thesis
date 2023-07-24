function sendTrignLog(cfg,eylnkMsg,logMsg,LPTtrig,sendTrigger,waitTime)
if cfg.LPT && waitTime>0
  sendTrigger(LPTtrig);
  WaitSecs(waitTime);
  sendTrigger(0);
elseif cfg.LPT && waitTime==0
  sendTrigger(LPTtrig);
else
   WaitSecs(waitTime);
end
if cfg.eylnk %send trial start trigger to eyelink
  Eyelink('Message',eylnkMsg);
end
logw(cfg.fid,logMsg);
end
