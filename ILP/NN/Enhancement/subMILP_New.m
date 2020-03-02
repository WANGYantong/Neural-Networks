function [final_state,TC_CNNMILP,saveDV_CNNMILP,whistle] = subMILP_New(img,init_state,score,Net,whistle)

NUMTEST=size(score,1);

result_CNNMILP=cell(NUMTEST,1);
alloc_CNNMILP=cell(NUMTEST,1);
TC_CNNMILP=zeros(NUMTEST,1);
saveDV_CNNMILP=zeros(NUMTEST,1);
% time_CNNMILP=zeros(NUMTEST,1);

% computation time & mean TC & number of decision variables
for ii=1:NUMTEST
    result_CNNMILP{ii}=subMILP(img(:,:,:,ii), init_state(ii,:), score(ii,:),Net);
     if isempty(result_CNNMILP{ii}.fval)
        result_CNNMILP{ii}=subMILP(img(:,:,:,ii), init_state(ii,:), score(ii,:)+1,Net);
        whistle=whistle+1;
    end
    alloc_CNNMILP{ii}=result_CNNMILP{ii}.allocations;
    TC_CNNMILP(ii)=result_CNNMILP{ii}.fval;
    saveDV_CNNMILP(ii)=result_CNNMILP{ii}.num_var;
%     time_CNNMILP(ii)=result_CNNMILP{ii}.time;
end

final_state=mat2cell([alloc_CNNMILP{1:end}]',NUMTEST,ones(1,5))';

end

