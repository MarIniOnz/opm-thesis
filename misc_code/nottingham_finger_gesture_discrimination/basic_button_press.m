%% Script for basic button press in OPM lab
% Uses fORP box by Current Designs
clear;

% coloured buttons - blue, yellow, green, red
response_key = KbName({'b','y','g','r'});

% button to start each block (corresponding to index finger button)
blue_key = KbName(response_key(1)); %i.e. '6^'
% initialise key press queue
keylist=zeros(1,256);%%create a list of 256 zeros
keylist(response_key)=1;%%set keys you interested in to 1
KbQueueCreate(0,keylist);

%% Set up triggers
%Clear ports
address = 57336;
io_obj = io64;
% initialize the interface to the inpoutx64 system driver
status = io64(io_obj);
io64(io_obj,address,0);
disp('Ports Cleared')

% Trigger channels for each colour button
trig_chans = [1 2 4 8];

%% Click button to start

disp('Press the blue button to start');
begin = 0;
while (~begin)
    [key_pressed, seconds, key_code] = KbCheck;
    if (key_pressed)
        send_trig = trig_chans(logical(key_code(response_key)));
        % send trig
        io64(io_obj, address, send_trig);
        pause(0.05)
        io64(io_obj, address, 0);
        
        % exit loop if blue key pressed
        begin = find(key_code) == KbName(blue_key);
    end
end
disp('Success!')

%% Some code for during trials that may be useful to someone

% KbQueueFlush(); % removes all keyboard presses
% KbQueueStart();
% 
% while (current_time <= stimulus_duration)
%     [key_pressed, t_first_press, key_code] = KbQueueCheck();
%     if key_pressed
%         response = 1;
%         %                     RT = t_first_press - trial_start_time;
%     end
%     current_time = GetSecs - stim_ontime;
% end
