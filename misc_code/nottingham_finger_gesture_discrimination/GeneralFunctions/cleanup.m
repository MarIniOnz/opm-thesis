function [] = cleanup(cfg)
%close logfile

try
    %Return propixx to normal state
    if cfg.propixx
        if cfg.luminance<1
            for l=1:3 %RGB leds
                Datapixx('SetPropixxLedCurrent',l-1,cfg.screen.current_orig(l));
            end
        end
        Datapixx('RegWrRd');
        Datapixx('SetPropixxDlpSequenceProgram', 0);
        if cfg.backprojection
            Datapixx('DisablePropixxRearProjection');
        end
        Datapixx('RegWrRd');
        Datapixx('close');
    end
catch
    warning('Returing the Propixx to normal state failed.');
end
%lower priority
Priority(0);
%stop eyelink & transfer file

%normal input
% ListenChar(0)

%close screen
sca
end
