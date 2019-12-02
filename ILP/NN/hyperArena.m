%%
clear
clc

addpath(genpath(pwd));

%%
% training_size=[1e3,2e3,3e3,4e3,5e3];
training_size=1024;
% batch_size=[1e2,2e2,5e2,1e3];
batch_size=64;
% epoch_size=[10,20,30,40];
epoch_size=1:10;
learning_rate=1e-3;
HID_INDEX=[1,2,3,4,16];

result=cell(length(training_size),length(batch_size),length(epoch_size),length(HID_INDEX));
for ii=1:length(training_size)
    for jj=1:length(batch_size)
        for kk=1:length(epoch_size)
            for ll=1:length(HID_INDEX)
                result{ii,jj,kk,ll}=hyperCandidate(training_size(ii),batch_size(jj),...
                    epoch_size(kk),learning_rate,HID_INDEX(ll));
            end
        end
    end
end

training_accuracy=zeros(length(epoch_size),length(HID_INDEX));
testing_accuracy=training_accuracy;
for kk=1:length(epoch_size)
    for ll=1:length(HID_INDEX)
        training_accuracy(kk,ll)=result{1,1,kk,ll}.training_accuracy;
        testing_accuracy(kk,ll)=result{1,1,kk,ll}.testing_accuracy(1);
    end
end

epoch_plot=1:length(epoch_size);
line_style={'-p','-s','-d','-^','-+'};
color_style={[0.85,0.33,0.10],[0.47,0.67,0.19],[0.30,0.75,0.93],...
    [0.64,0.08,0.18],[0.49,0.18,0.56]};

figure(1);
hold on;
for ll=1:length(HID_INDEX)
    plot(epoch_plot,training_accuracy(:,ll),line_style{ll},'Color',color_style{ll},'LineWidth',3.6);
end
xlabel('Epoch','FontSize',24);
ylabel('Mean Training Accuracy','FontSize',24);
ylim([0.2,1]);
lgd=legend({'1x','2x','3x','4x','16x'},'Location','southeast');
set(gca,'fontsize',24);
lgd.FontSize=24;
grid on;
hold off;

figure(2);
hold on;
for ll=1:length(HID_INDEX)
    plot(epoch_plot,testing_accuracy(:,ll),line_style{ll},'Color',color_style{ll},'LineWidth',3.6);
end
xlabel('Epoch','FontSize',24);
ylabel('Mean Testing Accuracy','FontSize',24);
ylim([0.2,0.8]);
lgd=legend({'1x','2x','3x','4x','16x'},'Location','southeast');
set(gca,'fontsize',24);
lgd.FontSize=24;
grid on;
hold off;