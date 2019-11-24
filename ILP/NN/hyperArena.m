%%
clear
clc

addpath(genpath(pwd));

%%
training_size=[1e3,2e3,3e3,4e3,5e3];
batch_size=[1e2,2e2,5e2,1e3];
epoch_size=[10,20,30,40];
learning_rate=1e-3;
HID_INDEX=[1,2,3,4];

for ii=1:length(training_size)
    for jj=1:length(batch_size)
        for kk=1:length(epoch_size)
            for ll=1:length(HID_INDEX)
                hyperCandidate(training_size(ii),batch_size(jj),...
                    epoch_size(kk),learning_rate,HID_INDEX(ll));
            end
        end
    end
end
