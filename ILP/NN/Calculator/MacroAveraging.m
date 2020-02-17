function [Accuracy,Precision,Recall,F1Score] = MacroAveraging(TP,FP,TN,FN)

Accuracy_each=(TP+TN)./(TP+FP+TN+FN);
Precision_each=TP./(TP+FP);
Recall_each=TP./(TP+FN);
F1Score_each=2*TP./(2*TP+FN+FP);

Accuracy=mean(Accuracy_each);
Precision=mean(Precision_each);
Recall=mean(Recall_each);
F1Score=mean(F1Score_each);

end

