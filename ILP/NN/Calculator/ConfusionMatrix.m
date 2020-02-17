function [TP,FP,TN,FN] = ConfusionMatrix(condition,prediction)

% reshape matrix to vector
condition=condition(:);
prediction=prediction(:);

C = confusionmat(condition,prediction);
NUM=sum(C,'all');

TP=zeros(length(C),1);
FP=TP;
TN=TP;
FN=TP;

for ii=1:length(C)
    TP(ii)=C(ii,ii);
    FP(ii)=sum(C(:,ii))-TP(ii);
    FN(ii)=sum(C(ii,:))-TP(ii);
    TN(ii)=NUM-TP(ii)-FP(ii)-FN(ii);
end

end
