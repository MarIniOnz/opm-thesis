function cfg=CreateLogfile(cfg)
exp_dir=cfg.exp_dir;
%check if logfile directory exists, if not create it
if ~exist([exp_dir 'Logs'],'dir')>0
  mkdir([exp_dir 'Logs']);
end

%Create folder for subject, if not already there
try
  if cfg.subnr<10
    sub_dir=[exp_dir 'Logs' filesep '0' int2str(cfg.subnr) '_' cfg.sub filesep];
  else
    sub_dir=[exp_dir 'Logs' filesep int2str(cfg.subnr) '_' cfg.sub filesep];
  end
  if ~exist(sub_dir,'dir')>0
    mkdir(sub_dir);
  end
  cfg.sub_dir=sub_dir;
catch
  cleanup(cfg);
  error('Subject information is not specified or misspecified')
end
%create file
try
  cfg.LogFile=[sub_dir 'Log_' int2str(cfg.subnr) '_' cfg.sub '_' cfg.stampstr '.txt'];
  cfg.datestamp=cfg.stampstr;
  fid=fopen(cfg.LogFile,'w');
  cfg.fid=fid;
  WaitSecs(1);
  fprintf(fid,'%12s\r\n',['This logfile was created on ' datestr(clock)]);
catch
  cleanup(cfg);
  error('Error whilst trying to create the log file. Log file cannot be created')
end
end
