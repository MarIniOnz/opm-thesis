function [resp,t] = getResponse2(cfg)
if cfg.lj.ljexist
    resp=0;t=0;button=1;
        [~,cs]  = cfg.lj.ljudObj.eDI(cfg.lj.ljhandle, button+15, 1); % get current state % check digital bit 16 (CIO0) and digital bit 17 (CIO1)
        t       = GetSecs(); % get time
        if cs > 0 % check if voltage exceeds threshold
            resp = button;
            return
        end        
else
    resp=0;t=999;
end

end