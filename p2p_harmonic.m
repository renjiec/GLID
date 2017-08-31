fprintf('\n\n');

if hasGPUComputing
    myGPUArray = @(x) gpuArray(x);
else
    myGPUArray = @(x) x;
end
        
%preprocessing - done once
if needsPreprocessing
    v = cellfun(myGPUArray, v, 'UniformOutput', false);

    if ~exist('Phi', 'var') || numel(Phi) ~= numel(vv)
        Phi = gather(vv);
        Psy = Phi*0;
    end

    energySamples = myGPUArray(energySamples);
    denseEvaluationSamples = myGPUArray(denseEvaluationSamples);

    %% preprocess for Modulus of Continuity
	[L, indexOfMaxL] = computeLipschitzConstantOfDerivativeOfCauchy(v, denseEvaluationSamples);

    [~, SoDerivativeOfCauchyCoordinatesAtEnergySamples] = derivativesOfCauchyCoord(v, energySamples, holeCenters);

    catv = myGPUArray( cat(1, v{:}) );
    L2 = myGPUArray(zeros(numel(energySamples), numel(catv)+numel(holeCenters)));
    for i=1:numel(catv)
        L2(:, i) = distancePointToSegment(catv(i), energySamples, energySamples(nextSampleInSameCage)).^-2/2/pi;
    end

    %% Lipschitz for log basis in multiconeccted case
    for i=1:numel(holeCenters)
        L2(:, end-numel(holeCenters)+i) = distancePointToSegment(holeCenters(i), energySamples, energySamples(nextSampleInSameCage)).^-3*2;
    end

    
    %%
    DerivativeOfCauchyCoordinatesAtFixedSamples = gather(derivativesOfCauchyCoord(v, fixedSamples, holeCenters));      %copmute on gpu and transfer to cpu
    DerivativeOfCauchyCoordinatesAtEnergySamples = derivativesOfCauchyCoord(v, energySamples, holeCenters);    %copmute on gpu and transfer to cpu
    DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples = gather(derivativesOfCauchyCoord(v, activeSetPoolSamples, holeCenters)); %copmute on gpu and transfer to cpu
    DerivativeOfCauchyCoordinatesAtDenseEvaluationSamples = derivativesOfCauchyCoord(v, denseEvaluationSamples, holeCenters); %no gather - keep on the gpu

    fillDistanceSegments = myGPUArray( abs(energySamples-energySamples(nextSampleInSameCage))/2 );

    Q = DerivativeOfCauchyCoordinatesAtEnergySamples'*DerivativeOfCauchyCoordinatesAtEnergySamples;
    
    mydiag = @(x) sparse(1:numel(x), 1:numel(x), x);
    [E, vec] = eig(gather(Q));
    vec = diag(vec); vec = vec(2:end);
    ARAP_q = mydiag(vec.^0.5)*E(:,2:end)';
    clear Q;
    
    needsPreprocessing = false;

    forceConformalMode = false;
end

total_time = tic;

switch solver_type
    case {'CVX', 'Direct Mosek'}
        active_set_is_on = true; %should we use the active set approach or not
        if active_set_is_on

            if(~exist('activeSetSigma1', 'var'))
                activeSetSigma1 = [];
            end
            if(~exist('activeSetSigma2', 'var'))
                activeSetSigma2 = [];
            end
            if(~exist('activeSet_k', 'var'))
                activeSet_k = [];
            end

            abs_fz = abs(DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples*Phi);
            abs_fzbar = abs(DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples*Psy);

            sigma1 = gather(abs_fz + abs_fzbar);
            sigma2 = gather(abs_fz - abs_fzbar);
            k = gather(abs_fzbar ./ abs_fz);

            upperThresholdSigma1 = 0.95*sigma1_upper_bound;
            lowerThresholdSigma1 = 0.945*sigma1_upper_bound;

            lowerThresholdSigma2 = 1.15*sigma2_lower_bound;
            upperThresholdSigma2 = 1.2*sigma2_lower_bound;

            upperThreshold_k = 0.95*k_upper_bound;
            lowerThreshold_k = 0.945*k_upper_bound;

            warning('off', 'signal:findpeaks:largeMinPeakHeight');

            [~, indicesToBeAddedSigma1] = findpeaks(sigma1, 'MinPeakHeight', upperThresholdSigma1);
            indicesToBeAddedSigma1 = union(indicesToBeAddedSigma1, find(sigma1 > sigma1_upper_bound));
            [~, indicesToBeAddedSigma2] = findpeaks(-sigma2, 'MinPeakHeight', -lowerThresholdSigma2); %we use minus to find local minimum
            indicesToBeAddedSigma2 = union(indicesToBeAddedSigma2, find(sigma2 < sigma2_lower_bound));
            [~, indicesToBeAdded_k] = findpeaks(k, 'MinPeakHeight', upperThreshold_k);
            indicesToBeAdded_k = union(indicesToBeAdded_k, find(k > k_upper_bound));


            activeSetSigma1 = union(activeSetSigma1, indicesToBeAddedSigma1);
            indicesToBeRemovedSigma1 = activeSetSigma1(sigma1(activeSetSigma1) < lowerThresholdSigma1);
            activeSetSigma1 = setdiff(activeSetSigma1, indicesToBeRemovedSigma1);

            activeSetSigma2 = union(activeSetSigma2, indicesToBeAddedSigma2);
            indicesToBeRemovedSigma2 = activeSetSigma2(sigma2(activeSetSigma2) > upperThresholdSigma2);
            activeSetSigma2 = setdiff(activeSetSigma2, indicesToBeRemovedSigma2);

            activeSet_k = union(activeSet_k, indicesToBeAdded_k);
            indicesToBeRemoved_k = activeSet_k(k(activeSet_k) < lowerThreshold_k);
            activeSet_k = setdiff(activeSet_k, indicesToBeRemoved_k);

            numActiveSetPool = size(DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples, 1);
        else
            activeSetSigma1 = [];
            activeSetSigma2 = [];
            activeSet_k = [];
        end

        DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1 = DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples(activeSetSigma1, 1:end);
        DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2 = DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples(activeSetSigma2, 1:end);
        DerivativeOfCauchyCoordinatesAtActiveSamples_k = DerivativeOfCauchyCoordinatesAtActiveSetPoolSamples(activeSet_k, 1:end);

        frames_sigma2 = calc_frames(DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2, Phi);
        frames_k = calc_frames(DerivativeOfCauchyCoordinatesAtActiveSamples_k, Phi);

        frames_fixed = calc_frames(DerivativeOfCauchyCoordinatesAtFixedSamples, Phi);

        frames_energy = calc_frames(DerivativeOfCauchyCoordinatesAtEnergySamples, Phi);

        arap_frames_vector = gather( transpose(frames_energy)*DerivativeOfCauchyCoordinatesAtEnergySamples ); %note the use of transpose rather than '
        ARAP_g = (arap_frames_vector*E(:,2:end)).*reshape(vec.^-0.5, 1, []);
end


optimizationTimeOnly = tic;

numVirtualVertices = size(CauchyCoordinatesAtP2Phandles, 2);
numFixedSamples = size(DerivativeOfCauchyCoordinatesAtFixedSamples, 1);
numEnergySamples = size(DerivativeOfCauchyCoordinatesAtEnergySamples, 1);
numDenseEvaluationSamples = size(DerivativeOfCauchyCoordinatesAtDenseEvaluationSamples, 1);


P2P_Deformation_Converged = 0;

switch solver_type
    case 'CVX' 
        [solverStatus, Energy_total, E_ISO, E_POSITIONAL, phi, psy] = cvx_p2p_harmonic( ...
            CauchyCoordinatesAtP2Phandles, DerivativeOfCauchyCoordinatesAtFixedSamples, ...
            DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1, DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2, DerivativeOfCauchyCoordinatesAtActiveSamples_k, ...
            P2PCurrentPositions, ARAP_g, ARAP_q, frames_fixed, frames_sigma2, frames_k, ...
            p2p_weight, sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, ...
            numVirtualVertices, numFixedSamples, numEnergySamples, ...
            no_output, forceConformalMode);

    case 'Direct Mosek'
        if forceConformalMode
           error('Conformal mode is currently supported only in CVX. Switch to CVX or try to reduce bound on k to a small number in order to approximate conformal');
        end

        tic
        [solverStatus, Energy_total, E_ISO, E_POSITIONAL, phi, psy] = mosek_p2p_harmonic( ...
            CauchyCoordinatesAtP2Phandles, DerivativeOfCauchyCoordinatesAtFixedSamples, ...
            DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1, DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2, DerivativeOfCauchyCoordinatesAtActiveSamples_k, ...
            P2PCurrentPositions, ARAP_g, ARAP_q, frames_fixed, frames_sigma2, frames_k, ...
            p2p_weight, sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, ...
            numVirtualVertices, numFixedSamples, numEnergySamples, ...
            true);

        
        statsAll = zeros(1, 8);
        statsAll(1, [5 7 8]) = [toc*1000 E_POSITIONAL Energy_total];

    case 'AQP'
        [XP2PDeform, statsAll] = meshAQP(X, T, P2PVtxIds, P2PCurrentPositions, XP2PDeform, AQP_nIterPerViz);
        
    case 'SLIM'
        [XP2PDeform, statsAll] = meshSLIM(X, T, P2PVtxIds, P2PCurrentPositions, XP2PDeform, AQP_nIterPerViz, p2p_weight, energy_type, energy_parameter);
        
    otherwise         %{'Gradient Descent', 'LBFGS', 'Newton', 'cuNewton_SPDH', ...}

        if ~exist('NLO_preprocessed','var') || ~NLO_preprocessed
            D2 = myGPUArray( DerivativeOfCauchyCoordinatesAtEnergySamples );
            NLO_preprocessed = true;
            nPhiPsyIters = 2;

            phi = Phi;
            psy = Psy;
        end
        
        if isempty(phipsyIters)
            phipsyIters = repmat( myGPUArray( [Phi Psy] ), 1, nPhiPsyIters );
        end

        C2 = myGPUArray(CauchyCoordinatesAtP2Phandles);
       
        if isempty(P2PCurrentPositions)
            Energy_total = 0; E_POSITIONAL = 0;
        else
            [phipsyIters, statsAll] = nlo_p2p_harmonic(myGPUArray(1i), D2, C2, myGPUArray(P2PCurrentPositions), softP2P, p2p_weight, ...
                phipsyIters, energy_parameter, AQP_kappa, AQP_nIterPerViz, solver_type, energy_type, nextSampleInSameCage, ...
                hessianSampleRate, fillDistanceSegments, v, SoDerivativeOfCauchyCoordinatesAtEnergySamples, L2);


            ppdif = norm( phipsyIters(:,1:2)-[phi psy], 'fro' )
            P2P_Deformation_Converged = gather(1*( ppdif < 1e-300 ));

            statsAll(:, end-1:end) = statsAll(:, end-1:end)/numEnergySamples; % energy normalization
        end
        
        phi = double(phipsyIters(:,1));
        psy = double(phipsyIters(:,2));
        solverStatus = 'solved';
end

if isempty(P2PCurrentPositions)
    Energy_total = 0; E_POSITIONAL = 0;
else
    E_POSITIONAL = statsAll(:,end-1);
    Energy_total = statsAll(:,end);
    E_ISO = Energy_total - E_POSITIONAL;
end


fprintf('Optimization time only time:%.4f\n', toc(optimizationTimeOnly));

if any(strcmpi(solver_type, {'CVX', 'Direct Mosek'}))
    validationTime = tic;
    
    LocallyInjectiveOnly = false;

    [Phi, Psy, t] = validateMapBoundsV2(L2, v, fillDistanceSegments, Phi, phi, Psy, psy, DerivativeOfCauchyCoordinatesAtEnergySamples, SoDerivativeOfCauchyCoordinatesAtEnergySamples, ...
                                       sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, 10, unrefinedInwardCage, false, LocallyInjectiveOnly);

    fprintf('Validation time:%.4f\n', toc(validationTime));
    
    
    E_POSITIONAL = p2p_weight*norm( CauchyCoordinatesAtP2Phandles*Phi+conj(CauchyCoordinatesAtP2Phandles*Psy) - P2PCurrentPositions )^2;
    statsAll(1, 7) = gather(E_POSITIONAL);
else
    t = 1; Phi = phi; Psy = psy;
end

DeformedP2PhandlePositions = CauchyCoordinatesAtP2Phandles*Phi + conj(CauchyCoordinatesAtP2Phandles*Psy);

Phi = gather(Phi); Psy = gather(Psy);

if ~strcmpi(solver_type, 'AQP') && ~strcmpi(solver_type, 'SLIM')
    XP2PDeform = gather(C*Phi + conj(C*Psy));
end

fprintf('E_ISO: %8.3e, E_POS: %8.3e, E: %8.3e\n', [E_ISO E_POSITIONAL Energy_total]');
fprintf('Total script time:%.4f\n', toc(total_time));
fprintf('Vertices:%d, Energy_samples:%d, Evaluation_samples:%d\n', numVirtualVertices, numEnergySamples, numDenseEvaluationSamples);
