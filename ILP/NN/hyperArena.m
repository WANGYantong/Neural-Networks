%%
clear
clc

addpath(genpath(pwd));

%%
mount_carlo=1:10;
% training_size=[1e3,2e3,3e3,4e3,5e3];
training_size=1024;
% batch_size=[16,32,64,128,256];
batch_size=64;
% epoch_size=[10,20,30,40];
% epoch_size=[1,5,10,15,20,25,30,35,40,45,50,55];
epoch_size=30;
learning_rate=[1e-4,1e-3,1e-2,1e-1,1];
% learning_rate=1e-3;
% hid_index=[1,2,3,4,5,16];
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
JJNUM=length(batch_size);
KKNUM=length(learning_rate);
result=cell(IINUM,JJNUM,KKNUM);
for ii=1:IINUM
    for jj=1:JJNUM
        for kk=1:KKNUM
            result{ii,jj,kk}=hyperCandidate(training_size,batch_size(jj),...
                epoch_size,learning_rate(kk),hid_index);
        end
    end
end

%% the effect of depth
% training_accuracy=zeros(length(mount_carlo),length(epoch_size),length(hid_index));
% training_loss=training_accuracy;
% validation_accuracy=training_accuracy;
% validation_loss=training_accuracy;
% testing_accuracy=training_accuracy;
% training_time=training_accuracy;
% for ii=1:length(mount_carlo)
%     for jj=1:length(epoch_size)
%         for kk=1:length(hid_index)
%             training_time(ii,jj,kk)=result{ii,jj,kk}.training_time;
%             training_accuracy(ii,jj,kk)=result{ii,jj,kk}.training_accuracy;
%             training_loss(ii,jj,kk)=result{ii,jj,kk}.training_loss;
%             validation_accuracy(ii,jj,kk)=result{ii,jj,kk}.validation_accuracy;
%             validation_loss(ii,jj,kk)=result{ii,jj,kk}.validation_loss;
%             testing_accuracy(ii,jj,kk)=result{ii,jj,kk}.testing_accuracy(1);
%         end
%     end
% end
% 
% training_time_plot=squeeze(mean(training_time));
% training_accuracy_plot=squeeze(mean(training_accuracy));
% training_loss_plot=squeeze(mean(training_loss));
% validation_accuracy_plot=squeeze(mean(validation_accuracy));
% validation_loss_plot=squeeze(mean(validation_loss));
% testing_accuracy_plot=squeeze(mean(testing_accuracy));
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% iteration_plot=16:16:length(epoch_size)*16;
% line_style1={'-o','-s','-d','-^','-p','-*'};
% line_style2={':o',':s',':d',':^',':p',':*'};
% color_style={[0.1,0.1,0.1],[0.2,0.2,0.2],[0.3,0.3,0.3],...
%     [0.4,0.4,0.4],[0.5,0.5,0.5],[0.5,0.5,0.5]};
% % color_style=[0.5,0.5,0.5];
% 
% hid_index_plot=[1,2,3,4,6];
% 
% figure(1);
% hold on;
% for kk=hid_index_plot
%     plot(iteration_plot,training_loss_plot(:,kk),line_style1{kk},'Color',color_style{kk},'LineWidth',2,'MarkerSize',16);
% end
% xlabel('Iterations','FontSize',24);
% ylabel('Loss Function','FontSize',24);
% xlim([iteration_plot(1),iteration_plot(end)]);
% % ylim([0.2,1]);
% lgd=legend({'1x','2x','3x','4x','16x'},'Location','north','NumColumns',5);
% set(gca,'fontsize',24);
% lgd.FontSize=24;
% grid on;
% hold off;
% 
% figure(2);
% hold on;
% for kk=hid_index_plot
%     plot(iteration_plot,validation_loss_plot(:,kk),line_style2{kk},'Color',color_style{kk},'LineWidth',2,'MarkerSize',16);
% end
% xlabel('Iterations','FontSize',24);
% ylabel('Loss Function','FontSize',24);
% xlim([iteration_plot(1),iteration_plot(end)]);
% % ylim([0.2,0.9]);
% lgd=legend({'1x','2x','3x','4x','16x'},'Location','north','NumColumns',5);
% set(gca,'fontsize',24);
% lgd.FontSize=24;
% grid on;
% hold off;

%% the effect of batch size
% training_accuracy=zeros(length(mount_carlo),length(batch_size),length(epoch_size));
% training_loss=training_accuracy;
% validation_accuracy=training_accuracy;
% validation_loss=training_accuracy;
% testing_accuracy=training_accuracy;
% training_time=training_accuracy;
% for ii=1:length(mount_carlo)
%     for jj=1:length(batch_size)
%         for kk=1:length(epoch_size)
%             training_time(ii,jj,kk)=result{ii,jj,kk}.training_time;
%             training_accuracy(ii,jj,kk)=result{ii,jj,kk}.training_accuracy;
%             training_loss(ii,jj,kk)=result{ii,jj,kk}.training_loss;
%             validation_accuracy(ii,jj,kk)=result{ii,jj,kk}.validation_accuracy;
%             validation_loss(ii,jj,kk)=result{ii,jj,kk}.validation_loss;
%             testing_accuracy(ii,jj,kk)=result{ii,jj,kk}.testing_accuracy(1);
%         end
%     end
% end
% 
% training_time_plot=squeeze(mean(training_time));
% training_accuracy_plot=squeeze(mean(training_accuracy));
% training_loss_plot=squeeze(mean(training_loss));
% validation_accuracy_plot=squeeze(mean(validation_accuracy));
% validation_loss_plot=squeeze(mean(validation_loss));
% testing_accuracy_plot=squeeze(mean(testing_accuracy));

% epoch_plot=epoch_size;
% line_style1={'-o','-s','-d','-^','-p','-*'};
% line_style2={':o',':s',':d',':^',':p',':*'};
% color_style={[0.1,0.1,0.1],[0.2,0.2,0.2],[0.3,0.3,0.3],...
%     [0.4,0.4,0.4],[0.5,0.5,0.5],[0.5,0.5,0.5]};
% 
% figure(1);
% hold on;
% for jj=1:length(batch_size)
%     plot(epoch_plot,training_loss_plot(jj,:),line_style1{jj},'Color',color_style{jj},'LineWidth',2,'MarkerSize',16);
% end
% xlabel('Epoch','FontSize',24);
% ylabel('Loss Function','FontSize',24);
% xlim([epoch_plot(1),epoch_plot(end)]);
% % ylim([0.2,1]);
% lgd=legend({'16','32','64','256','1024'},'Location','north','NumColumns',5);
% set(gca,'fontsize',24);
% lgd.FontSize=24;
% grid on;
% hold off;
% 
% figure(2);
% hold on;
% for jj=1:length(batch_size)
%     plot(epoch_plot,validation_loss_plot(jj,:),line_style2{jj},'Color',color_style{jj},'LineWidth',2,'MarkerSize',16);
% end
% xlabel('Epoch','FontSize',24);
% ylabel('Loss Function','FontSize',24);
% xlim([epoch_plot(1),epoch_plot(end)]);
% % ylim([0.2,0.8]);
% lgd=legend({'16','32','64','256','1024'},'Location','north','NumColumns',5);
% set(gca,'fontsize',24);
% lgd.FontSize=24;
% grid on;
% hold off;

%% the effect of epoch

% training_accuracy=zeros(length(mount_carlo),length(batch_size),length(epoch_size));
% training_loss=training_accuracy;
% validation_accuracy=training_accuracy;
% validation_loss=training_accuracy;
% testing_accuracy=training_accuracy;
% training_time=training_accuracy;
% for ii=1:length(mount_carlo)
%     for jj=1:length(batch_size)
%         for kk=1:length(epoch_size)
%             training_time(ii,jj,kk)=result{ii,jj,kk}.training_time;
%             training_accuracy(ii,jj,kk)=result{ii,jj,kk}.training_accuracy;
%             training_loss(ii,jj,kk)=result{ii,jj,kk}.training_loss;
%             validation_accuracy(ii,jj,kk)=result{ii,jj,kk}.validation_accuracy;
%             validation_loss(ii,jj,kk)=result{ii,jj,kk}.validation_loss;
%             testing_accuracy(ii,jj,kk)=result{ii,jj,kk}.testing_accuracy(1);
%         end
%     end
% end
% 
% training_time_plot=squeeze(mean(training_time));
% training_accuracy_plot=squeeze(mean(training_accuracy));
% training_loss_plot=squeeze(mean(training_loss));
% validation_accuracy_plot=squeeze(mean(validation_accuracy));
% validation_loss_plot=squeeze(mean(validation_loss));
% testing_accuracy_plot=squeeze(mean(testing_accuracy));

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
line_style1={'-o','-s','-d','-^','-p','-*'};
line_style2={':o',':s',':d',':^',':p',':*'};
color_style={[0.1,0.1,0.1],[0.2,0.2,0.2],[0.3,0.3,0.3],...
    [0.4,0.4,0.4],[0.5,0.5,0.5],[0.5,0.5,0.5]};

figure(1);
hold on;
plot(epoch_size,training_loss_plot,line_style1{1},'Color',color_style{1},'LineWidth',2,'MarkerSize',16);
plot(epoch_size,validation_loss_plot,line_style2{2},'Color',color_style{2},'LineWidth',2,'MarkerSize',16);
xlabel('Epoch','FontSize',24);
ylabel('Loss Function','FontSize',24);
xlim([epoch_size(1),epoch_size(end)]);
lgd=legend({'Training','Validation'},'Location','north','NumColumns',2);
set(gca,'fontsize',24);
lgd.FontSize=24;
grid on;
hold off;

%% the effect of learning rate

% training_accuracy=zeros(length(mount_carlo),length(batch_size),length(learning_rate));
% training_loss=training_accuracy;
% validation_accuracy=training_accuracy;
% validation_loss=training_accuracy;
% testing_accuracy=training_accuracy;
% training_time=training_accuracy;
% for ii=1:length(mount_carlo)
%     for jj=1:length(batch_size)
%         for kk=1:length(learning_rate)
%             training_time(ii,jj,kk)=result{ii,jj,kk}.training_time;
%             training_accuracy(ii,jj,kk)=result{ii,jj,kk}.training_accuracy;
%             training_loss(ii,jj,kk)=result{ii,jj,kk}.training_loss;
%             validation_accuracy(ii,jj,kk)=result{ii,jj,kk}.validation_accuracy;
%             validation_loss(ii,jj,kk)=result{ii,jj,kk}.validation_loss;
%             testing_accuracy(ii,jj,kk)=result{ii,jj,kk}.testing_accuracy(1);
%         end
%     end
% end
% 
% training_time_plot=squeeze(mean(training_time));
% training_accuracy_plot=squeeze(mean(training_accuracy));
% training_loss_plot=squeeze(mean(training_loss));
% validation_accuracy_plot=squeeze(mean(validation_accuracy));
% validation_loss_plot=squeeze(mean(validation_loss));
% testing_accuracy_plot=squeeze(mean(testing_accuracy));


line_style1={'-o','-s','-d','-^','-p','-*'};
line_style2={':o',':s',':d',':^',':p',':*'};
color_style={[0.1,0.1,0.1],[0.2,0.2,0.2],[0.3,0.3,0.3],...
    [0.4,0.4,0.4],[0.5,0.5,0.5],[0.5,0.5,0.5]};

figure(1);
hold on;
plot(learning_rate,training_loss_plot,line_style1{1},'Color',color_style{1},'LineWidth',2,'MarkerSize',16);
plot(learning_rate,validation_loss_plot,line_style2{2},'Color',color_style{2},'LineWidth',2,'MarkerSize',16);
x_axes=xlabel('Learning Rate','FontSize',24);
ylabel('Loss Function','FontSize',24);
lgd=legend({'Training','Validation'},'Location','north','NumColumns',2);
set(gca,'fontsize',24,'xscale','log');
lgd.FontSize=24;
grid on;
hold off;