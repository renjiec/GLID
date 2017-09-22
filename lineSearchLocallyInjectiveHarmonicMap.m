function [ls_t, minGlobal_sigma2] = lineSearchLocallyInjectiveHarmonicMap(phipsy, dpp, fzgz0, dfzgz, ls_t, fillDistanceSegments, v, E2, L, nextSampleInSameCage)

%% compute second order differentail first, with reduced variable set for multiply connected domains
fzzgzz   = E2*phipsy;
dfzzgzz  = E2*dpp;

%%
if iscell(v)
    %% special process for holes
    cageSizes = cellfun(@numel, v);
    inextv = [2:sum(cageSizes) 1];
    iprevv = inextv([end-1:end 1:end-2]);
    inextv( cumsum(cageSizes) ) = 1+cumsum([0 cageSizes(1:end-1)]);
    iprevv( 1+cumsum([0 cageSizes(1:end-1)]) ) = cumsum(cageSizes);
    v = cat(1, v{:});

    nVarAll = numel(v)+numel(cageSizes)-1;
    isFreeVar = ~full( sparse(1, reshape( cumsum(cageSizes(1:end-1)), [], 1 ) + [1 2], true, 1, nVarAll) );
    mfree2full = full( sparse(find(isFreeVar), 1:sum(isFreeVar), 1, nVarAll, size(phipsy,1)) );
    
    phipsy = mfree2full*phipsy;
    dpp = mfree2full*dpp;
    
    assert( nargin>9 );
else
    inextv = [2:size(phipsy, 1) 1];   % next phipsy in the same cage
    iprevv = inextv([end-1:end 1:end-2]);  % previous phipsy in the same cage
    nextSampleInSameCage = [2:size(fzgz0,1) 1];
end

normdpp = norm(dpp);

%% non-vanishing fz
% this check combined with the condition that min(|fz|)>0 for all segments is sufficient condition to assure that fz does not vanish inside the domain
% proof in the paper is slightly weaker, as it requires max(|fz0|, |fz1|) > L_fz*len
argument_princial_approx = inf;
while ls_t*normdpp>1e-20
    fzgzt = fzgz0 + ls_t*dfzgz;

    dtheta = angle( fzgzt(nextSampleInSameCage,1)./fzgzt(:,1) );
    argument_princial_approx = sum( dtheta );
    if argument_princial_approx < 1
        break;
    end

    ls_t = ls_t/2;
end

%%
fAvgOverSamplePairs = @(x) (x+x(nextSampleInSameCage,:))/2;
fAvgAbsOverSamplePairs = @(x) fAvgOverSamplePairs( abs(x) );
fDiff  = @(x) x(inextv, :) - x;
fDiffP = @(x) x - x(iprevv, :);


nv = numel(v);
dS  = fDiffP( fDiff(phipsy(1:nv,:))./fDiff(v) );
ddS = fDiffP( fDiff(   dpp(1:nv,:))./fDiff(v) );

% add Lipschitz for holes
logTermIdxs = nv+1:size(phipsy,1);
if ~isempty(logTermIdxs)
    dS = [dS;   phipsy(logTermIdxs,:)];
    ddS= [ddS;  dpp(logTermIdxs,:)];
end

delta_L_fzgz0 = L*abs(dS);
delta_L_fzgz1 = L*abs(dS+ls_t*ddS);
ls_t0 = ls_t;

minGlobal_sigma2 = -inf;


while true
    sufficientstep = ls_t*normdpp>1e-12;
    
%     L_fzgz    = fAvgAbsOverPairs(fzzgzz+ls_t*dfzzgzz) + (L*abs(dSdz+ls_t*ddSdz)).*fillDistanceSegments;

    delta_L_fzgz = delta_L_fzgz0 + (delta_L_fzgz1-delta_L_fzgz0)*(ls_t/ls_t0);
    L_fzgz    = fAvgAbsOverSamplePairs(fzzgzz+ls_t*dfzzgzz) + delta_L_fzgz.*fillDistanceSegments;
    fzgzt = fzgz0 + ls_t*dfzgz;

%     fprintf('L_fz: %.2f, L_fzbar: %.2f\n', max(L_fzgz));

    %% local injectivity, sigma2>0
    min_absfz = fAvgAbsOverSamplePairs( fzgzt(:,1) ) - L_fzgz(:,1).*fillDistanceSegments;
    max_absgz = fAvgAbsOverSamplePairs( fzgzt(:,2) ) + L_fzgz(:,2).*fillDistanceSegments;
    minGlobal_sigma2 = min(min_absfz - max_absgz);
    
    if sufficientstep && minGlobal_sigma2 < 0
        ls_t = ls_t/2;
        continue;
    end

    max_absfz = fAvgAbsOverSamplePairs( fzgzt(:,1) ) + L_fzgz(:,1).*fillDistanceSegments;
    maxGlobal_sigma1 = max(max_absfz + max_absgz);
    maxGlobal_k = max( max_absgz ./ min_absfz );

    maxOnSamples_sigma1 = max(abs(fzgzt)*[1; 1]);
    minOnSamples_sigma2 = min(abs(fzgzt)*[1; -1]);
    maxOnSamples_k = max( abs(fzgzt(:,2))./abs(fzgzt(:,1)) );

    
    fprintf('t: %.5f, sigma1: (%.3f, %.3f), sigma2: (%.3f, %.3f), k: (%.3f, %.3f)\n', ls_t, maxOnSamples_sigma1, maxGlobal_sigma1, minOnSamples_sigma2, minGlobal_sigma2, maxOnSamples_k, maxGlobal_k);
%     assert(maxGlobal_sigma1>0);

    %% all condition satisfied or step too small
    break;
end

% argument principal must be satisfied: contour integral fz'/fz = 0 <=> fz non vanishing
% assert( argument_princial_approx < 1 );
