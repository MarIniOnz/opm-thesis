function [cfg,lj] = initaliseRespButton(cfg)
lj=[];
try
% Initalise Labjack
% Make the UD .NET assembly visible in MATLAB.
lj.ljasm = NET.addAssembly('LJUDDotNet');
lj.ljudObj = LabJack.LabJackUD.LJUD;

% Open the first found LabJack U3.
[lj.ljerror, lj.ljhandle] = lj.ljudObj.OpenLabJackS('LJ_dtU3', 'LJ_ctUSB', '0', true, 0);

% Start by using the pin_configuration_reset IOType so that all pin
% assignments are in the factory default condition.
lj.ljudObj.ePutS(lj.ljhandle, 'LJ_ioPIN_CONFIGURATION_RESET', 0, 0, 0);

% define voltage threshold for button press
lj.volt_thr = 0.1;
cfg.lj=lj;
cfg.lj.ljexist=1;
catch
    warning( 'WARNING: Response button setup failed. No response will be collected.');

    while true
        m=input('Do you want to continue? y/n    ','s');
        if m=='y'
            cfg.lj.ljexist=0;
            break
        elseif m=='n'
            error('The experiment aborted.')
        end
    end   
end
