function setupPropixx(cfg,propixxsetting)

if cfg.propixx
        try
            Datapixx('Open');
            Datapixx('SetPropixxDlpSequenceProgram', propixxsetting); % 2 for 480, 5 for 1440 Hz, 0 for normal
            if cfg.backprojection
                Datapixx('EnablePropixxRearProjection');
            end
            Datapixx('RegWrRd');
        catch
            cleanup(cfg);
            error('Propixx setup failed. Experiment aborted.')
        end
    end
