function labelUpdate = combiner(labelOriginal,score)

if nargin < 2
    opts=0; %  totally randomized
else
    opts=1; % based on prediction score
end

NF=length(labelOriginal);
NE=length(score)/NF;

if opts
    
    scoreRe=reshape(score,[NE,NF])';
    
    % get the nonzero prediction
    % emumerate each prediction by 1-bit flip
        % space check
        % link check
        % object compare
    
end

end

