function accuracy=ErrorCalc(imgLabels, score, opt)
% calculate top x accuracy

counter=0;
for ii=1:opt.NT
    for jj=1:opt.NF
        [~,idx] = sort(score{jj}(ii,:),'descend');
        candidates=categorical(idx(1:opt.mode));
        if ismember(imgLabels(ii,jj),candidates)
            counter=counter+1;
        end
    end
end
accuracy=counter/(opt.NF*opt.NT);

end

