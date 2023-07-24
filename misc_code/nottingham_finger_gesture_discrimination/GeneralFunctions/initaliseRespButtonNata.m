function[keylist] = InitialiseRespButtonNata(keys,device)

keylist=zeros(1,256);%%create a list of 256 zeros
keylist(keys)=1;%%set keys you interested in to 1 

KbQueueCreate(device,keylist);%%make cue
end
