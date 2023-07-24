function logCfg(cfg)

fprintf(cfg.fid,'%12s\r\n','Config options: \n');
names=fieldnames(cfg);
vals=struct2cell(cfg);
for i=1:length(names)
    if ~isstruct(vals{i})
        fprintf(cfg.fid,'%12s\r\n',[names{i} ': ' num2str(vals{i})]);
    else
        names2=fieldnames(getfield(cfg,names{i}));
        vals2=struct2cell(getfield(cfg,names{i}));
        for i2=1:length(names2)
%            fprintf(cfg.fid,'%12s\r\n',[names2{i2} ': ' num2str(vals2{i2})]);
        end
    end
end
