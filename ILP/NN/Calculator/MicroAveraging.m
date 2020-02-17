function [Accuracy,Precision,Recall,F1Score] = MicroAveraging(TP,FP,TN,FN)

TP_sum=sum(TP);
FP_sum=sum(FP);
TN_sum=sum(TN);
FN_sum=sum(FN);

Accuracy=(TP_sum+TN_sum)/(TP_sum+FP_sum+TN_sum+FN_sum);
Precision=TP_sum/(TP_sum+FP_sum);
Recall=TP_sum/(TP_sum+FN_sum);
F1Score=2*TP_sum/(2*TP_sum+FN_sum+FP_sum);

end

