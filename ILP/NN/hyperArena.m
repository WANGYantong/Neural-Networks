%%
clear
clc

addpath(genpath(pwd));

%%
mount_carlo=1:10;
% training_size=[1e3,2e3,3e3,4e3,5e3];
training_size=1024;
% batch_size=[1e2,2e2,5e2,1e3];
batch_size=64;
% epoch_size=[10,20,30,40];
epoch_size=[1,5,10,15,20,25,30,35,40,45,50,55];
% learning_rate=[1e-4,1e-3,1e-2,1e-1,1];
learning_rate=1e-3;
% hid_index=[1,2,3,4,16];
hid_index=1;

% result=cell(length(training_size),length(batch_size),length(epoch_size),length(hid_index));
% for ii=1:length(training_size)
%     for jj=1:length(batch_size)
%         for kk=1:length(epoch_size)
%             for ll=1:length(hid_index)
%                 result{ii,jj,kk,ll}=hyperCandidate(training_size(ii),batch_size(jj),...
%                     epoch_size(kk),learning_rate,hid_index(ll));
%             end
%         end
%     end
% end
% 
IINUM=length(mount_carlo);
JJNUM=length(epoch_size);
result=cell(IINUM,JJNUM);
parfor ii=1:IINUM
    for jj=1:JJNUM
            result{ii,jj}=hyperCandidate(training_size,batch_size,...
                epoch_size(jj),learning_rate,hid_index);
    end
end

%% the effect of depth
% training_accuracy=zeros(length(mount_carlo),length(epoch_size),length(hid_index));
% testing_accuracy=training_accuracy;
% for ii=1:length(mount_carlo)
%     for jj=1:length(epoch_size)
%         for kk=1:length(hid_index)
%             training_accuracy(ii,jj,kk)=result{ii,jj,kk}.training_accuracy;
%             testing_accuracy(ii,jj,kk)=result{ii,jj,kk}.testing_accuracy(1);
%         end
%     end
% end
% 
% training_accuracy_plot=squeeze(mean(training_accuracy));
% testing_accuracy_plot=squeeze(mean(testing_accuracy));
% iteration_plot=16:16:length(epoch_size)*16;
% line_style={'-p','-s','-d','-^','-+'};
% color_style={[0.85,0.33,0.10],[0.47,0.67,0.19],[0.30,0.75,0.93],...
%     [0.64,0.08,0.18],[0.49,0.18,0.56]};
% 
% figure(1);
% hold on;
% for kk=1:length(hid_index)
%     plot(iteration_plot,training_accuracy_plot(:,kk),line_style{kk},'Color',color_style{kk},'LineWidth',3.6);
% end
% xlabel('Iterations','FontSize',24);
% ylabel('Mean Training Accuracy','FontSize',24);
% xlim([16,160]);
% ylim([0.2,1]);
% lgd=legend({'1x','2x','3x','4x','16x'},'Location','south','NumColumns',5);
% set(gca,'fontsize',24);
% lgd.FontSize=24;
% grid on;
% hold off;
% 
% figure(2);
% hold on;
% for kk=1:length(hid_index)
%     plot(iteration_plot,testing_accuracy_plot(:,kk),line_style{kk},'Color',color_style{kk},'LineWidth',3.6);
% end
% xlabel('Iterations','FontSize',24);
% ylabel('Mean Testing Accuracy','FontSize',24);
% xlim([16,160]);
% ylim([0.2,0.8]);
% lgd=legend({'1x','2x','3x','4x','16x'},'Location','south','NumColumns',5);
% set(gca,'fontsize',24);
% lgd.FontSize=24;
% grid on;
% hold off;

%% the effect of batch size
% training_accuracy=zeros(length(mount_carlo),length(batch_size),length(epoch_size));
% testing_accuracy=training_accuracy;
% for ii=1:length(mount_carlo)
%     for jj=1:length(batch_size)
%         for kk=1:length(epoch_size)
%             training_accuracy(ii,jj,kk)=result{ii,jj,kk}.training_accuracy;
%             testing_accuracy(ii,jj,kk)=result{ii,jj,kk}.testing_accuracy(1);
%         end
%     end
% end
% 
% training_accuracy_plot=squeeze(mean(training_accuracy));
% testing_accuracy_plot=squeeze(mean(testing_accuracy));
% epoch_plot=1:length(epoch_size);
% line_style={'-p','-s','-d','-^','-+'};
% color_style={[0.85,0.33,0.10],[0.47,0.67,0.19],[0.30,0.75,0.93],...
%     [0.64,0.08,0.18],[0.49,0.18,0.56]};
% 
% figure(1);
% hold on;
% for jj=1:length(batch_size)
%     plot(epoch_plot,training_accuracy_plot(jj,:),line_style{jj},'Color',color_style{jj},'LineWidth',3.6);
% end
% xlabel('Epoch','FontSize',24);
% ylabel('Mean Training Accuracy','FontSize',24);
% xlim([1,20]);
% % ylim([0.2,1]);
% lgd=legend({'64','256','1024'},'Location','south','NumColumns',3);
% set(gca,'fontsize',24);
% lgd.FontSize=24;
% grid on;
% hold off;
% 
% figure(2);
% hold on;
% for jj=1:length(batch_size)
%     plot(epoch_plot,testing_accuracy_plot(jj,:),line_style{jj},'Color',color_style{jj},'LineWidth',3.6);
% end
% xlabel('Epoch','FontSize',24);
% ylabel('Mean Testing Accuracy','FontSize',24);
% xlim([1,20]);
% % ylim([0.2,0.8]);
% lgd=legend({'64','256','1024'},'Location','south','NumColumns',3);
% set(gca,'fontsize',24);
% lgd.FontSize=24;
% grid on;
% hold off;

%% the effect of epoch
% training_accuracy=zeros(length(mount_carlo),length(epoch_size));
% testing_accuracy=training_accuracy;
% for ii=1:length(mount_carlo)
%     for jj=1:length(epoch_size)
%         training_accuracy(ii,jj)=result{ii,jj}.training_accuracy;
%         testing_accuracy(ii,jj)=result{ii,jj}.testing_accuracy(1);
%     end
% end
% 
% training_accuracy_plot=squeeze(mean(training_accuracy));
% testing_accuracy_plot=squeeze(mean(testing_accuracy));
% line_style={'-p','-s','-d','-^','-+'};
% color_style={[0.85,0.33,0.10],[0.47,0.67,0.19],[0.30,0.75,0.93],...
%     [0.64,0.08,0.18],[0.49,0.18,0.56]};
% 
% figure(1);
% hold on;
% plot(epoch_size,training_accuracy_plot,line_style{1},'Color',color_style{1},'LineWidth',3.6);
% plot(epoch_size,testing_accuracy_plot,line_style{2},'Color',color_style{2},'LineWidth',3.6);
% xlabel('Epoch','FontSize',24);
% ylabel('Mean Accuracy','FontSize',24);
% xlim([1,55]);
% lgd=legend({'Training','Testing'},'Location','south','NumColumns',2);
% set(gca,'fontsize',24);
% lgd.FontSize=24;
% grid on;
% hold off;

%% the effect of learning rate
training_accuracy=zeros(length(mount_carlo),length(learning_rate));
testing_accuracy=training_accuracy;
for ii=1:length(mount_carlo)
    for jj=1:length(learning_rate)
        training_accuracy(ii,jj)=result{ii,jj}.training_accuracy;
        testing_accuracy(ii,jj)=result{ii,jj}.testing_accuracy(1);
    end
end

training_accuracy_plot=squeeze(mean(training_accuracy));
testing_accuracy_plot=squeeze(mean(testing_accuracy));
line_style={'-p','-s','-d','-^','-+'};
color_style={[0.85,0.33,0.10],[0.47,0.67,0.19],[0.30,0.75,0.93],...
    [0.64,0.08,0.18],[0.49,0.18,0.56]};

figure(1);
hold on;
plot(learning_rate,training_accuracy_plot,line_style{1},'Color',color_style{1},'LineWidth',3.6);
plot(learning_rate,testing_accuracy_plot,line_style{2},'Color',color_style{2},'LineWidth',3.6);
x_axes=xlabel('Learning Rate','FontSize',24);
ylabel('Mean Accuracy','FontSize',24);
lgd=legend({'Training','Testing'},'Location','south','NumColumns',2);
set(gca,'fontsize',24,'xscale','log');
lgd.FontSize=24;
grid on;
hold off;