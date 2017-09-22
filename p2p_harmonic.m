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

    %% preprocess for Modulus of Continuity
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

    DerivativeOfCauchyCoordinatesAtEnergySamples = derivativesOfCauchyCoord(v, energySamples, holeCenters);    %copmute on gpu and transfer to cpu
    fillDistanceSegments = myGPUArray( abs(energySamples-energySamples(nextSampleInSameCage))/2 );
    
    nextSampleInSameCage = myGPUArray(nextSampleInSameCage);
    
    needsPreprocessing = false;
    
    phi = Phi; psy = Psy;
end

total_time = tic;

optimizationTimeOnly = tic;

numVirtualVertices = size(CauchyCoordinatesAtP2Phandles, 2);
numEnergySamples = size(DerivativeOfCauchyCoordinatesAtEnergySamples, 1);

P2P_Deformation_Converged = 0;

switch solver_type
    case 'AQP'
        [XP2PDeform, statsAll] = meshAQP(X, T, P2PVtxIds, P2PCurrentPositions, XP2PDeform, numIterations);

    case 'SLIM'
        [XP2PDeform, statsAll] = meshSLIM(X, T, P2PVtxIds, P2PCurrentPositions, XP2PDeform, numIterations, p2p_weight, energy_type, energy_parameter);
        
    otherwise
             %{'Newton', 'Newton SPDH'}
        if ~exist('NLO_preprocessed','var') || ~NLO_preprocessed
            D2 = myGPUArray( DerivativeOfCauchyCoordinatesAtEnergySamples );

            NLO_preprocessed = true;
            nPhiPsyIters = 2;

            phipsyIters = repmat( myGPUArray( [phi psy] ), 1, nPhiPsyIters );
        end
        
        C2 = myGPUArray(CauchyCoordinatesAtP2Phandles);
        
        if ~isempty(P2PCurrentPositions)
            [phipsyIters, statsAll] = nlo_p2p_harmonic(D2, C2, myGPUArray(P2PCurrentPositions), p2p_weight, ...
                phipsyIters, energy_parameter, numIterations, solver_type, energy_type, nextSampleInSameCage, ...
                hessianSampleRate, fillDistanceSegments, v, SoDerivativeOfCauchyCoordinatesAtEnergySamples, L2);

            ppdif = norm( phipsyIters(:,1:2)-[phi psy], 'fro' )
            P2P_Deformation_Converged = gather(1*( ppdif < 1e-300 ));
            P2P_Converged_settings = struct('energy', energy_type, 'p2p_weight', p2p_weight, 'energy_param', energy_parameter);
            
            statsAll(:, end-1:end) = statsAll(:, end-1:end)/numEnergySamples; % energy normalization
        end

        phi = double(phipsyIters(:,1));
        psy = double(phipsyIters(:,2));
end

if isempty(P2PCurrentPositions)
    Energy_total = 0; E_POSITIONAL = 0;
else
    E_POSITIONAL = statsAll(:,end-1);
    Energy_total = statsAll(:,end);
    E_ISO = Energy_total - E_POSITIONAL;
end

fprintf('Optimization time: %.4f\n', toc(optimizationTimeOnly));

if ~any(strcmpi(solver_type, {'AQP', 'SLIM'}))
    XP2PDeform = gather(C*phi + conj(C*psy));
    DeformedP2PhandlePositions = CauchyCoordinatesAtP2Phandles*phi + conj(CauchyCoordinatesAtP2Phandles*psy);
end

if ~isempty(P2PCurrentPositions)
    fprintf('E_ISO: %8.3e, E_POS: %8.3e, E_def: %8.3e\n', [E_ISO E_POSITIONAL Energy_total]');
    fprintf('Total script time:%.4f\n', toc(total_time));
end
