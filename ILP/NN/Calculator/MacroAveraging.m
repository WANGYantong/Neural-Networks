function [Accuracy,Precision,Recall,F1Score] = MacroAveraging(TP,FP,TN,FN)

Accuracy_each=(TP+TN)./(TP+FP+TN+FN);
Precision_each=TP./(TP+FP);
Recall_each=TP./(TP+FN);
F1Score_each=2*TP./(2*TP+FN+FP);

for ii=1:length(TP)
    if (TP(ii)==0) && (FP(ii)==0) && (FN(ii)==0)
        Precision_each(ii)=1;
        Recall_each(ii)=1;
        F1Score_each(ii)=1;
    end
    
    if (TP(ii)==0 && FP(ii)==0 && FN(ii) ~= 0) || ...
            (TP(ii)==0 && FP(ii)~=0 && FN(ii) == 0)
        Precision_each(ii)=0;
        Recall_each(ii)=0;
        F1Score_each(ii)=0;
    end
end

Accuracy=mean(Accuracy_each);
Precision=mean(Precision_each);
Recall=mean(Recall_each);
F1Score=mean(F1Score_each);

end

