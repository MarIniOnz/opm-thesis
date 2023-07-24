function [resp,t] = getResponse(cfg,timeout,noljtime)
if cfg.lj.ljexist
  resp=0;waitstart=GetSecs;t=0;button=1;
  while t<timeout
      [~,cs]  = cfg.lj.ljudObj.eDI(cfg.lj.ljhandle,button+15,1);
      t=GetSecs()-waitstart;
      if cs>0 %exceeds threshold
        resp = button;
        return
      end
  end
else
  waitstart=GetSecs;  
  WaitSecs(noljtime);
  resp=0;
  t=GetSecs()-waitstart;
end
end
