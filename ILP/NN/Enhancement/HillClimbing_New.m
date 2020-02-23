function final_state= HillClimbing_New(img,init_state,score,Net)

NUMTEST=size(score,1);
alloc_HCLS=cell(NUMTEST,1);

for ii=1:NUMTEST
    [buff_HCLS,~,~]=...
        HillClimbing(img(:,:,:,ii), init_state(ii,:), score(ii,:),Net);
    alloc_HCLS{ii}=buff_HCLS';
end

final_state=mat2cell([alloc_HCLS{1:end}]',NUMTEST,ones(1,5))';

end

