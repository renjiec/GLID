function [phi_valid, psy_valid, t] = validateMapBoundsV2(L, v, fillDistanceSegments, phi_prev, phi_next, psy_prev, psy_next, DerivativeOfCauchyCoordinates, SODerivativeOfCauchyCoordinates, ...
sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, maxBisectionIterations, unrefinedInwardCage, use_Cauchy_argument_principle, LocallyInjectiveOnly)
%given a previous solution (phi_prev, psy_prev) and a new solution (phi_next, psy_next), this function checks whether the new solution
%is valid (satisfy the requiered bounds). The function assumes that the previous solution is valid.
%if the new solution is not valid a line search is being performed until a valid solution is reached.
%   phi_valid, psy_valid - the valid solutions
%   t - a parameter between 0 and 1. t=0 means the prev solution is being returned.
%       t=1 means the next solution is valid and is being returned.
%       any value smaller than 1 indicates that the obtained valid solution was found on the boundary of the feasible domain.
%
%   inwardOffsetCage - can be the inward offset cage without any virtual vertices (for better speed)

%debug - right now I don't search the upper part of the space for a solution. this actually means that the obtained solution will not be on the boundary of the feasible domain.
%I think that this way there is less chance to get stuck

%debug - maybe expose this to the user?

if iscell(v) && numel(v)==1, v = v{1}; end

assert(~iscell(v), 'multi-connected domain is not supported yet');

if ~exist('LocallyInjectiveOnly', 'var'), LocallyInjectiveOnly = false; end
if LocallyInjectiveOnly
    sigma2_lower_bound = 0;
    sigma1_upper_bound = inf;
    k_upper_bound = inf;
end


sigma2_min_allowed = 0.7*sigma2_lower_bound;
sigma1_max_allowed = 1.3*sigma1_upper_bound;
k_max_allowed = min(1, 1.2*k_upper_bound);


t = 1;

for i=1:maxBisectionIterations
    
    phi_valid = (1-t)*phi_prev + t*phi_next;
    psy_valid = (1-t)*psy_prev + t*psy_next;
    
    %return; %for debugging - this disables the validation
    
    if hasGPUComputing
        phi_valid = gpuArray(phi_valid);
        psy_valid = gpuArray(psy_valid);
    end
    
    [fz_does_not_vanish, lowerBoundGlobal_sigma2, upperBoundGlobal_sigma1, upperBoundGlobal_k, minOnSamples_sigma2, maxOnSamples_sigma1, maxOnSamples_k] = computeBoundsOnAllTypeOfDistortion(L, v, fillDistanceSegments, DerivativeOfCauchyCoordinates, SODerivativeOfCauchyCoordinates, phi_valid, psy_valid, unrefinedInwardCage, use_Cauchy_argument_principle);
    
    fprintf('t: %.5f, sigma1: (%.3f, %.3f, %.3f), sigma2: (%.3f, %.3f, %.3f), k: (%.3f, %.3f, %.3f)\n', ...
        t, ...
        sigma1_upper_bound, maxOnSamples_sigma1, upperBoundGlobal_sigma1, ...
        sigma2_lower_bound, minOnSamples_sigma2, lowerBoundGlobal_sigma2, ...
        k_upper_bound, maxOnSamples_k, upperBoundGlobal_k);

%     if (fz_does_not_vanish)    
    if (fz_does_not_vanish && ...
            lowerBoundGlobal_sigma2 >= sigma2_min_allowed && ...
            upperBoundGlobal_sigma1 <= sigma1_max_allowed && ...
            upperBoundGlobal_k <= k_max_allowed)
        
        return; %everything is ok. map is locally injective with all bounds satisfied
    else
        fprintf('Map corresponding to t: %.5f is invalid. Trying to reduce t...\n', t);
        if(i == maxBisectionIterations-1) %last iteration
            fprintf('Line search failed to progress. Reverting to previous map (t=0)\n');
            t = 0;
        else
            t = t/2;
        end
    end
end
%     %if we reached here it means that we didn't manage to find a valid solution after maxBisectionIterations iterations
%     t = 0;
%     phi_valid = phi_prev;
%     psy_valid = psy_prev;
end



function [fz_does_not_vanish, lowerBoundGlobal_sigma2, upperBoundGlobal_sigma1, upperBoundGlobal_k, minOnSamples_sigma2, maxOnSamples_sigma1, maxOnSamples_k] ...
    = computeBoundsOnAllTypeOfDistortion(L, v, fillDistanceSegments, DerivativeOfCauchyCoordinates, SODerivativeOfCauchyCoordinates, phi, psy, unrefinedInwardCage, use_Cauchy_argument_principle)

tic;

fz = DerivativeOfCauchyCoordinates*phi;

[~, thetaDiffs] = extractBestAngle(fz);

abs_fz = abs(fz);

dphidz = (phi([2:end 1]) - phi)./(v([2:end 1]) - v);
dpsydz = (psy([2:end 1]) - psy)./(v([2:end 1]) - v);
fzz   = abs(SODerivativeOfCauchyCoordinates*phi);
fzbzb = abs(SODerivativeOfCauchyCoordinates*psy);
fAvgOverPairs = @(x) (x+x([2:end 1]))/2;
L_fz    = fAvgOverPairs(fzz  ) + (L*abs( dphidz - dphidz([end 1:end-1]) )).*fillDistanceSegments;
L_fzbar = fAvgOverPairs(fzbzb) + (L*abs( dpsydz - dpsydz([end 1:end-1]) )).*fillDistanceSegments;

fprintf('L_fz: %.2f, L_fzbar: %.2f\n', max(L_fz), max(L_fzbar));

deltaTheta = abs(thetaDiffs);

%this is the sufficient condition to assure that fz does not vanish inside the domain
% if(all(1e-5 + (2 + deltaTheta).*L_fz.*fillDistanceSegments < (2 - deltaTheta).*(abs_fz + abs_fz([2:end 1]))))  % TODO: why 1e-5?
if(all((2 + deltaTheta).*L_fz.*fillDistanceSegments*2 < (2 - deltaTheta).*(abs_fz + abs_fz([2:end 1]))))  % TODO: why 1e-5?

    fz_does_not_vanish = true;
%     fprintf('***********verified! with L_fz= %f, l= %f***************\n', max(L_fz), max(fillDistanceSegments));
else %sufficient conditions failed but there is still a chance that fz does not vanish since this is not a necessary condition

    if(use_Cauchy_argument_principle)
        %we use the Cauchy argument principle to get a sharp answer to whether fz vanishes or not

        fprintf('\n');
        numZeros = computeNumOfZerosOf_fz_inside(v, phi, unrefinedInwardCage);
        fprintf('numZeros of fz inside the domain: %d.\n', numZeros);
        if(numZeros == 0)
            fz_does_not_vanish = true;
        else
            fz_does_not_vanish = false;
        end
        fprintf('\n');
    else
        fz_does_not_vanish = false;
    end
end

if(fz_does_not_vanish)
    %now that we know that fz does not vanish in the domain we can bound sigma2, sigma1, and k.
    upperBoundOnEachSegment_abs_fz = computeUpperBounds(L_fz, fillDistanceSegments, abs_fz);

    abs_fzbar = abs(DerivativeOfCauchyCoordinates*psy);
    
    upperBoundOnEachSegment_abs_fzbar = computeUpperBounds(L_fzbar, fillDistanceSegments, abs_fzbar);
    lowerBoundOnEachSegment_abs_fz = computeLowerBounds(L_fz, fillDistanceSegments, abs_fz);
    
    lowerBoundGlobal_sigma2 = min(lowerBoundOnEachSegment_abs_fz - upperBoundOnEachSegment_abs_fzbar);
    upperBoundGlobal_sigma1 = max(upperBoundOnEachSegment_abs_fz + upperBoundOnEachSegment_abs_fzbar);
    upperBoundGlobal_k = max(upperBoundOnEachSegment_abs_fzbar ./ lowerBoundOnEachSegment_abs_fz);
    
    minOnSamples_sigma2 = min(abs_fz - abs_fzbar);
    maxOnSamples_sigma1 = max(abs_fz + abs_fzbar);
    maxOnSamples_k = max(abs_fzbar ./ abs_fz);
    
    % 	lowerBoundGlobal_sigma2_notTight = lowerBoundGlobal_abs_fz - upperBoundGlobal_abs_fzbar;
    %     upperBoundGlobal_sigma1_notTight = upperBoundGlobal_abs_fz + upperBoundGlobal_abs_fzbar;
    %     upperBoundGlobal_k_notTight = upperBoundGlobal_abs_fzbar / lowerBoundGlobal_abs_fz;
    
else %fz vanishes

    fprintf('fz vanishes inside the domain!\n');
    fz_does_not_vanish = false;
    lowerBoundGlobal_sigma2 = -Inf;
    minOnSamples_sigma2 = -Inf;
    upperBoundGlobal_sigma1 = Inf;
    maxOnSamples_sigma1 = Inf;
    upperBoundGlobal_k = Inf;
    maxOnSamples_k = Inf;
end

fprintf('computeBoundsOnAllTypeOfDistortion time: %.5f\n', toc);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   the following two functions compute a lower or upper bounds for a real function f on a polygon.
%
%   L_segments - the Lipschitz constant of the function f on each segment.
%   fillDistanceSegments - length of segments divided by 2.
%   f - the known values of the function (must be real) at the samples (vertices of the polygon).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lowerBoundOnEachSegment, lowerBoundGlobal] = computeLowerBounds(L_segments, fillDistanceSegments, f)

avg_val = 1/2*(f + f([2:end 1]));

lowerBoundOnEachSegment = avg_val - L_segments.*fillDistanceSegments;

if(nargout > 1)
    lowerBoundGlobal = min(lowerBoundOnEachSegment);
end
end


function [upperBoundOnEachSegment, upperBoundGlobal] = computeUpperBounds(L_segments, fillDistanceSegments, f)

avg_val = 1/2*(f + f([2:end 1]));

upperBoundOnEachSegment = avg_val + L_segments.*fillDistanceSegments;

if(nargout > 1)
    upperBoundGlobal = max(upperBoundOnEachSegment);
end
end
