function expBasicSetup(skypsinc)
%% Some Basics

Screen('Preference', 'SkipSyncTests', skypsinc); %must be 0 during experiment
if skypsinc==1
    warning( 'WARNING: Sync test is skipped! Do not continue if it is an actual recording!');
    m=input('Do you want to continue? y/n   ','s');
    while true
        if m=='y'
            break
        elseif m=='n'
            error('The experiment aborted.')
        end
    end
end
dummy = GetSecs; clear dummy;% make a dummy call to GetSecs to load the .dll before we need it

%ListenChar(2);

